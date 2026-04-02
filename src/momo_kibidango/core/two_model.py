"""Two-model speculative decoder using dependency injection.

Draft model proposes candidate tokens; target model verifies them in a single
forward pass.  Tokens whose target probability exceeds the stage-2 threshold
are accepted; the first rejected position triggers a resample from the target
distribution and ends the current draft round.
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
from momo_kibidango.monitoring import MetricsCollector

logger = logging.getLogger(__name__)


class TwoModelDecoder(BaseDecoder):
    """Speculative decoder with a draft + target model pair."""

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
        self._target: Optional[LoadedModel] = None
        self._loaded = False

    # -- BaseDecoder interface -----------------------------------------------

    @property
    def mode(self) -> str:  # noqa: D401
        return "2model"

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        """Load draft and target models via the injected ModelLoader."""
        try:
            draft_spec = self._registry.get_tier(ModelTier.DRAFT)
            target_spec = self._registry.get_tier(ModelTier.TARGET)

            logger.info("Loading draft model: %s", draft_spec.model_id)
            self._draft = self._loader.load(draft_spec, device=self._settings.resolve_device())

            logger.info("Loading target model: %s", target_spec.model_id)
            self._target = self._loader.load(target_spec, device=self._settings.resolve_device())

            self._loaded = True
            logger.info("TwoModelDecoder loaded successfully")
        except Exception as exc:
            self._loaded = False
            raise ModelLoadError(f"Failed to load 2-model pipeline: {exc}") from exc

    def unload(self) -> None:
        """Release all model memory."""
        self._loader.unload_all()
        self._draft = None
        self._target = None
        self._kv_cache.invalidate()
        self._loaded = False
        logger.info("TwoModelDecoder unloaded")

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Run speculative decoding: draft then verify in a loop."""
        if not self._loaded:
            raise ModelNotLoadedError(
                "TwoModelDecoder.generate() called before load()"
            )

        assert self._draft is not None
        assert self._target is not None

        device = self._settings.resolve_device()
        tokenizer = self._draft.tokenizer
        eos_token_id = tokenizer.eos_token_id

        # Tokenize prompt
        input_ids = tokenizer.encode(request.prompt, return_tensors="pt").to(device)

        generated_tokens: list[int] = []
        total_draft_attempts = 0
        total_accepted = 0
        memory_samples: list[float] = []

        t_start = time.perf_counter()

        temperature = request.temperature or self._settings.temperature
        max_draft = self._settings.max_draft_tokens
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

                # -- draft step ----------------------------------------------
                draft_tokens, draft_probs = self._draft_step(
                    input_ids, max_draft, temperature, device,
                )
                total_draft_attempts += len(draft_tokens)

                # -- verify step ---------------------------------------------
                accepted, n_accepted = self._verify_step(
                    input_ids, draft_tokens, temperature, stage2_thresh, device,
                )
                total_accepted += n_accepted

                # Adaptive update
                if self._adaptive and len(draft_tokens) > 0:
                    self._adaptive.update("stage2", n_accepted, len(draft_tokens))
                    stage2_thresh = self._adaptive.stage2_threshold

                # Append accepted tokens
                for tok in accepted[:n_accepted]:
                    generated_tokens.append(tok)
                    if tok == eos_token_id or len(generated_tokens) >= request.max_new_tokens:
                        break

                # Update input_ids
                if accepted:
                    new_ids = torch.tensor(
                        accepted[:n_accepted], device=device,
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
        acceptance_rate = (
            total_accepted / total_draft_attempts if total_draft_attempts > 0 else 0.0
        )
        peak_mem = max(memory_samples) if memory_samples else 0.0

        # Record to metrics collector
        self._metrics.record_inference(
            duration_seconds=elapsed,
            tokens_generated=tokens_generated,
            model_mode=self.mode,
            acceptance_rate=acceptance_rate,
            stage_rates={"stage2": acceptance_rate},
        )

        return GenerationResult(
            text=text,
            tokens_generated=tokens_generated,
            elapsed_seconds=elapsed,
            tokens_per_second=tps,
            acceptance_rate=acceptance_rate,
            stage_acceptance_rates={"stage2": acceptance_rate},
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

    def _verify_step(
        self,
        input_ids: torch.Tensor,
        draft_tokens: list[int],
        temperature: float,
        threshold: float,
        device: str,
    ) -> tuple[list[int], int]:
        """Verify draft tokens against the target model.

        Returns the list of accepted (or resampled) tokens and the count of
        tokens that were genuinely accepted from the draft.
        """
        assert self._target is not None

        if not draft_tokens:
            return [], 0

        accepted_tokens: list[int] = []
        draft_ids = torch.tensor(draft_tokens, device=device).unsqueeze(0)
        extended_ids = torch.cat([input_ids, draft_ids], dim=1)

        outputs = self._target.model(extended_ids)

        for i, draft_token in enumerate(draft_tokens):
            target_logits = outputs.logits[:, input_ids.size(1) + i - 1, :]
            target_probs = torch.softmax(
                target_logits / max(temperature, 1e-8), dim=-1,
            )

            if target_probs[0, draft_token].item() > threshold:
                accepted_tokens.append(draft_token)
            else:
                # Reject — resample from target distribution
                new_token = torch.multinomial(target_probs, num_samples=1).squeeze().item()
                accepted_tokens.append(new_token)
                break

        return accepted_tokens, len(accepted_tokens)
