"""Cross-tokenizer token mapping for multi-model pipelines."""

from __future__ import annotations
import logging
from typing import Any

logger = logging.getLogger(__name__)


class TokenizerBridge:
    """Maps tokens between different tokenizer vocabularies (e.g., Qwen <-> Phi-2).

    Accepts pre-loaded tokenizer objects (not IDs) to avoid redundant loading.
    When both tokenizers are the same object, all mapping calls short-circuit
    to identity operations.
    """

    def __init__(self, source_tokenizer: Any, target_tokenizer: Any) -> None:
        self._source = source_tokenizer
        self._target = target_tokenizer
        self._same = source_tokenizer is target_tokenizer

    @property
    def requires_mapping(self) -> bool:
        """Return ``True`` when the source and target tokenizers differ."""
        return not self._same

    def map_tokens(
        self,
        token_ids: list[int],
        direction: str = "source_to_target",
    ) -> list[int]:
        """Map a sequence of token IDs between vocabularies.

        Parameters
        ----------
        token_ids:
            Token IDs in the source (or target) vocabulary.
        direction:
            ``"source_to_target"`` or ``"target_to_source"``.

        Returns
        -------
        list[int]
            Token IDs in the opposite vocabulary.
        """
        if self._same:
            return token_ids
        if direction == "source_to_target":
            text = self._source.decode(token_ids, skip_special_tokens=True)
            return self._target.encode(text, add_special_tokens=False)
        else:
            text = self._target.decode(token_ids, skip_special_tokens=True)
            return self._source.encode(text, add_special_tokens=False)

    def map_single_token(
        self,
        token_id: int,
        direction: str = "source_to_target",
    ) -> int:
        """Map a single token ID, returning the first mapped token.

        Falls back to the original *token_id* if mapping produces an empty
        sequence (e.g., whitespace-only tokens that one tokenizer absorbs).
        """
        mapped = self.map_tokens([token_id], direction)
        return mapped[0] if mapped else token_id

    def map_text_to_target(self, text: str) -> list[int]:
        """Encode raw *text* directly with the target tokenizer."""
        return self._target.encode(text, add_special_tokens=False)

    def reverse_map_token(self, token_id: int) -> int:
        """Map a single token from target space back to source space."""
        return self.map_single_token(token_id, direction="target_to_source")

    def decode_source(self, token_ids: list[int]) -> str:
        """Decode *token_ids* using the source tokenizer."""
        return self._source.decode(token_ids, skip_special_tokens=True)
