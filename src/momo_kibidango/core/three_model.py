"""Three-model pyramid speculative decoder.

Adds a qualifier (mid-tier) model between draft and target.  The pipeline
runs in three stages:

1. **Draft** -- small model generates candidate tokens quickly.
2. **Qualify** -- mid-tier model filters candidates using stage-1 threshold.
   If the qualifier uses a different tokenizer, a ``TokenizerBridge``
   translates token IDs transparently.
3. **Verify** -- large target model performs final verification on the
   surviving candidates using stage-2 threshold.

This layered approach reduces expensive target-model forward passes while
maintaining output quality.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import psutil
import torch

from momo_kibidango.config.settings import DecoderSettings
from momo_kibidango.core.adaptive import AdaptiveThreshold
from momo_kibidango.core.decoder import BaseDecoder, GenerationRequest, GenerationResult
from momo_kibidango.core.kv_cache import KVCacheManager
from momo_kibidango.exceptions import ModelLoadError, ModelNotLoadedError
from momo_kibidango.models.registry import ModelRegistry, ModelTier
from momo_kibidango.models.loader import LoadedModel, ModelLoader
from momo_kibidango.models.tokenizer_bridge import TokenizerBridge
from momo_kibidango.monitoring import MetricsCollector

logger = logging.getLogger(__name__)


class ThreeModelDecoder(BaseDecoder):
    """Speculative decoder with draft + qualifier + target pyramid."""

    def __init__(
        self,
        settings: DecoderSettings,
        registry: ModelRegistry,
        loader: ModelLoader,
        metrics: MetricsCollector,
        adaptive: AdaptiveThreshold | None = None,
    ) -> None:
        self._settings = settings
        self._registry = registry
        self._loader = loader
        self._metrics = metrics
        self._adaptive = adaptive
        self._kv_cache = KVCacheManager()

        self._draft: Optional[LoadedModel] = None
        self._qualifier: Optional[LoadedModel] = None
        self._target: Optional[LoadedModel] = None

        # Bridges created after load() when tokenizers differ
        self._bridge_draft_to_qual: Optional[TokenizerBridge] = None
        self._bridge_qual_to_target: Optional[TokenizerBridge] = None

        self._loaded = False

    # -- BaseDecoder interface -----------------------------------------------

    @property
    def mode(self) -> str:  # noqa: D401
        return "3model"

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        """Load all three models and create tokenizer bridges if needed."""
        try:
            draft_spec = self._registry.get_tier(ModelTier.DRAFT)
            qualifier_spec = self._registry.get_tier(ModelTier.QUALIFIER)
            target_spec = self._registry.get_tier(ModelTier.TARGET)
            device = self._settings.resolve_device()

            logger.info("Loading draft model: %s", draft_spec.model_id)
            self._draft = self._loader.load(draft_spec, device=device)

            logger.info("Loading qualifier model: %s", qualifier_spec.model_id)
            self._qualifier = self._loader.load(qualifier_spec, device=device)

            logger.info("Loading target model: %s", target_spec.model_id)
            self._target = self._loader.load(target_spec, device=device)

            # Create tokenizer bridges when vocabularies differ
            if (
                draft_spec.effective_tokenizer_id
                != qualifier_spec.effective_tokenizer_id
            ):
                self._bridge_draft_to_qual = TokenizerBridge(
                    self._draft.tokenizer, self._qualifier.tokenizer,
                )
                logger.info(
                    "TokenizerBridge created: draft -> qualifier",
                )

            if (
                qualifier_spec.effective_tokenizer_id
                != target_spec.effective_tokenizer_id
            ):
                self._bridge_qual_to_target = TokenizerBridge(
                    self._qualifier.tokenizer, self._target.tokenizer,
                )
                logger.info(
                    "TokenizerBridge created: qualifier -> target",
                )

            self._loaded = True
            logger.info("ThreeModelDecoder loaded successfully")
        except Exception as exc:
            self._loaded = False
            raise ModelLoadError(
                f"Failed to load 3-model pipeline: {exc}"
            ) from exc

    def unload(self) -> None:
        """Release all model memory."""
        self._loader.unload_all()
        self._draft = None
        self._qualifier = None
        self._target = None
        self._bridge_draft_to_qual = None
        self._bridge_qual_to_target = None
        self._kv_cache.invalidate()
        self._loaded = False
        logger.info("ThreeModelDecoder unloaded")

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Run three-stage speculative decoding: draft -> qualify -> verify."""
        if not self._loaded:
            raise ModelNotLoadedError(
                "ThreeModelDecoder.generate() called before load()"
            )

        assert self._draft is not None
        assert self._qualifier is not None
        assert self._target is not None

        device = self._settings.resolve_device()
        tokenizer = self._draft.tokenizer
        eos_token_id = tokenizer.eos_token_id

        input_ids = tokenizer.encode(request.prompt, return_tensors="pt").to(device)

        generated_tokens: list[int] = []
        total_draft_attempts = 0
        total_accepted = 0
        stage1_accepted_total = 0
        stage1_total = 0
        stage2_accepted_total = 0
        stage2_total = 0
        memory_samples: list[float] = []

        t_start = time.perf_counter()

        temperature = request.temperature or self._settings.temperature
        max_draft = self._settings.max_draft_tokens
        stage1_thresh = (
            self._adaptive.stage1_threshold
            if self._adaptive
            else self._settings.stage1_threshold
        )
        stage2_thresh = (
            self._adaptive.stage2_threshold
            if self._adaptive
            else self._settings.stage2_threshold
        )

        with torch.no_grad():
            while len(generated_tokens) < request.max_new_tokens:
                # -- memory snapshot -----------------------------------------
                mem_gb = psutil.Process().memory_info().rss / (1024 ** 3)
                memory_samples.append(mem_gb)

                # -- Stage 1: draft ------------------------------------------
                draft_tokens, _draft_probs = self._draft_step(
                    input_ids, max_draft, temperature, device,
                )
                total_draft_attempts += len(draft_tokens)

                # -- Stage 2: qualify ----------------------------------------
                qualified_tokens, n_qual_accepted = self._qualify_step(
                    input_ids, draft_tokens, temperature, stage1_thresh, device,
                )
                stage1_total += len(draft_tokens)
                stage1_accepted_total += n_qual_accepted

                if self._adaptive and len(draft_tokens) > 0:
                    self._adaptive.update("stage1", n_qual_accepted, len(draft_tokens))
                    stage1_thresh = self._adaptive.stage1_threshold

                # -- Fallback: if nothing survived qualifier, sample from target
                if not qualified_tokens:
                    fallback_token = self._target_sample(input_ids, temperature, device)
                    generated_tokens.append(fallback_token)
                    total_accepted += 1

                    new_ids = torch.tensor(
                        [fallback_token], device=device,
                    ).unsqueeze(0)
                    input_ids = torch.cat([input_ids, new_ids], dim=1)

                    if fallback_token == eos_token_id:
                        break
                    continue

                # -- Stage 3: verify -----------------------------------------
                accepted, n_ver_accepted = self._verify_step(
                    input_ids, qualified_tokens, temperature, stage2_thresh, device,
                )
                stage2_total += len(qualified_tokens)
                stage2_accepted_total += n_ver_accepted
                total_accepted += n_ver_accepted

                if self._adaptive and len(qualified_tokens) > 0:
                    self._adaptive.update("stage2", n_ver_accepted, len(qualified_tokens))
                    stage2_thresh = self._adaptive.stage2_threshold

                # Append accepted tokens
                for tok in accepted[:n_ver_accepted]:
                    generated_tokens.append(tok)
                    if tok == eos_token_id or len(generated_tokens) >= request.max_new_tokens:
                        break

                # Update input_ids
                if accepted:
                    new_ids = torch.tensor(
                        accepted[:n_ver_accepted], device=device,
                    ).unsqueeze(0)
                    input_ids = torch.cat([input_ids, new_ids], dim=1)

                # Check stop conditions
                if generated_tokens and generated_tokens[-1] == eos_token_id:
                    break
                if any(
                    seq in tokenizer.decode(generated_tokens)
                    for seq in request.stop_sequences
                ):
                    break

        t_end = time.perf_counter()
        elapsed = t_end - t_start

        text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        tokens_generated = len(generated_tokens)
        tps = tokens_generated / elapsed if elapsed > 0 else 0.0
        overall_acceptance = (
            total_accepted / total_draft_attempts if total_draft_attempts > 0 else 0.0
        )
        s1_rate = stage1_accepted_total / stage1_total if stage1_total > 0 else 0.0
        s2_rate = stage2_accepted_total / stage2_total if stage2_total > 0 else 0.0
        peak_mem = max(memory_samples) if memory_samples else 0.0

        # Record to metrics collector
        self._metrics.record_inference(
            duration_seconds=elapsed,
            tokens_generated=tokens_generated,
            model_mode=self.mode,
            acceptance_rate=overall_acceptance,
            stage_rates={"stage1": s1_rate, "stage2": s2_rate},
        )

        return GenerationResult(
            text=text,
            tokens_generated=tokens_generated,
            elapsed_seconds=elapsed,
            tokens_per_second=tps,
            acceptance_rate=overall_acceptance,
            stage_acceptance_rates={"stage1": s1_rate, "stage2": s2_rate},
            peak_memory_gb=peak_mem,
            mode=self.mode,
            draft_attempts=total_draft_attempts,
            accepted_tokens=total_accepted,
        )

    # -- Internal helpers ----------------------------------------------------

    def _draft_step(
        self,
        input_ids: torch.Tensor,
        n_tokens: int,
        temperature: float,
        device: str,
    ) -> tuple[list[int], list[torch.Tensor]]:
        """Generate candidate tokens using the draft model."""
        assert self._draft is not None

        draft_tokens: list[int] = []
        draft_probs: list[torch.Tensor] = []
        current_ids = input_ids.clone()

        for _ in range(n_tokens):
            outputs = self._draft.model(current_ids)
            logits = outputs.logits[:, -1, :]
            probs = torch.softmax(logits / max(temperature, 1e-8), dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze()

            draft_tokens.append(next_token.item())
            draft_probs.append(probs)

            current_ids = torch.cat(
                [current_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1,
            )

        return draft_tokens, draft_probs

    def _qualify_step(
        self,
        input_ids: torch.Tensor,
        draft_tokens: list[int],
        temperature: float,
        threshold: float,
        device: str,
    ) -> tuple[list[int], int]:
        """Filter draft tokens through the qualifier model.

        Uses a TokenizerBridge when draft and qualifier tokenizers differ.
        Returns the list of tokens that passed qualification (in draft-tokenizer
        space) and the number accepted.
        """
        assert self._qualifier is not None

        if not draft_tokens:
            return [], 0

        # Translate token IDs if tokenizers differ
        qual_tokens = draft_tokens
        if self._bridge_draft_to_qual is not None:
            qual_tokens = self._bridge_draft_to_qual.map_tokens(draft_tokens)

        qual_ids = torch.tensor(qual_tokens, device=device).unsqueeze(0)
        extended_ids_qual = self._prepare_qualifier_input(input_ids, qual_ids, device)

        outputs = self._qualifier.model(extended_ids_qual)

        accepted: list[int] = []
        n_accepted = 0

        for i, (draft_tok, qual_tok) in enumerate(zip(draft_tokens, qual_tokens)):
            qual_logits = outputs.logits[:, input_ids.size(1) + i - 1, :]
            qual_probs = torch.softmax(
                qual_logits / max(temperature, 1e-8), dim=-1,
            )

            if qual_probs[0, qual_tok].item() > threshold:
                accepted.append(draft_tok)
                n_accepted += 1
            else:
                # First rejection ends the qualified sequence
                break

        return accepted, n_accepted

    def _prepare_qualifier_input(
        self,
        input_ids: torch.Tensor,
        qual_draft_ids: torch.Tensor,
        device: str,
    ) -> torch.Tensor:
        """Build qualifier input, bridging tokenizers if necessary."""
        if self._bridge_draft_to_qual is not None:
            assert self._draft is not None and self._qualifier is not None
            # Re-encode the prompt text with the qualifier's tokenizer
            prompt_text = self._draft.tokenizer.decode(
                input_ids[0].tolist(), skip_special_tokens=True,
            )
            qual_input = self._qualifier.tokenizer.encode(
                prompt_text, return_tensors="pt",
            ).to(device)
            return torch.cat([qual_input, qual_draft_ids], dim=1)
        return torch.cat([input_ids, qual_draft_ids], dim=1)

    def _verify_step(
        self,
        input_ids: torch.Tensor,
        qualified_tokens: list[int],
        temperature: float,
        threshold: float,
        device: str,
    ) -> tuple[list[int], int]:
        """Verify qualified tokens against the target model.

        If draft and target use different tokenizers, tokens are bridged
        through ``_bridge_qual_to_target`` (or directly if qualifier and target
        share the same vocabulary).
        """
        assert self._target is not None

        if not qualified_tokens:
            return [], 0

        # Translate tokens to target tokenizer space if needed
        target_tokens = qualified_tokens
        if self._bridge_qual_to_target is not None:
            target_tokens = self._bridge_qual_to_target.map_tokens(qualified_tokens)
        elif self._bridge_draft_to_qual is None:
            # Draft and target share tokenizer; use draft tokens directly
            target_tokens = qualified_tokens

        target_draft_ids = torch.tensor(target_tokens, device=device).unsqueeze(0)

        # Build target input — re-encode prompt if tokenizers differ
        if self._bridge_qual_to_target is not None or self._bridge_draft_to_qual is not None:
            assert self._draft is not None
            prompt_text = self._draft.tokenizer.decode(
                input_ids[0].tolist(), skip_special_tokens=True,
            )
            target_input = self._target.tokenizer.encode(
                prompt_text, return_tensors="pt",
            ).to(device)
            extended_ids = torch.cat([target_input, target_draft_ids], dim=1)
            base_len = target_input.size(1)
        else:
            extended_ids = torch.cat([input_ids, target_draft_ids], dim=1)
            base_len = input_ids.size(1)

        outputs = self._target.model(extended_ids)

        accepted: list[int] = []
        n_accepted = 0

        for i, (orig_tok, tgt_tok) in enumerate(
            zip(qualified_tokens, target_tokens)
        ):
            target_logits = outputs.logits[:, base_len + i - 1, :]
            target_probs = torch.softmax(
                target_logits / max(temperature, 1e-8), dim=-1,
            )

            if target_probs[0, tgt_tok].item() > threshold:
                accepted.append(orig_tok)
                n_accepted += 1
            else:
                # Reject — resample from target distribution
                new_token_tgt = torch.multinomial(
                    target_probs, num_samples=1,
                ).squeeze().item()
                # Map back to draft tokenizer space if needed
                if self._bridge_qual_to_target is not None:
                    mapped = self._bridge_qual_to_target.reverse_map_token(new_token_tgt)
                    accepted.append(mapped)
                elif self._bridge_draft_to_qual is not None:
                    mapped = self._bridge_draft_to_qual.reverse_map_token(new_token_tgt)
                    accepted.append(mapped)
                else:
                    accepted.append(new_token_tgt)
                n_accepted += 1
                break

        return accepted, n_accepted

    def _target_sample(
        self, input_ids: torch.Tensor, temperature: float, device: str,
    ) -> int:
        """Sample a single token from the target model (fallback path)."""
        assert self._target is not None

        # Re-encode if tokenizers differ
        if self._bridge_draft_to_qual is not None or self._bridge_qual_to_target is not None:
            assert self._draft is not None
            prompt_text = self._draft.tokenizer.decode(
                input_ids[0].tolist(), skip_special_tokens=True,
            )
            target_input = self._target.tokenizer.encode(
                prompt_text, return_tensors="pt",
            ).to(device)
        else:
            target_input = input_ids

        outputs = self._target.model(target_input)
        logits = outputs.logits[:, -1, :]
        probs = torch.softmax(logits / max(temperature, 1e-8), dim=-1)
        token = torch.multinomial(probs, num_samples=1).squeeze().item()

        # Map back to draft tokenizer if needed
        if self._bridge_qual_to_target is not None:
            token = self._bridge_qual_to_target.reverse_map_token(token)
        elif self._bridge_draft_to_qual is not None:
            token = self._bridge_draft_to_qual.reverse_map_token(token)

        return token
