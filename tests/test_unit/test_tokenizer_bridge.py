"""Unit tests for TokenizerBridge."""

import pytest
from unittest.mock import MagicMock

from momo_kibidango.models.tokenizer_bridge import TokenizerBridge


@pytest.fixture
def source_tokenizer():
    tok = MagicMock()
    tok.decode = MagicMock(return_value="hello world")
    tok.encode = MagicMock(return_value=[10, 20, 30])
    return tok


@pytest.fixture
def target_tokenizer():
    tok = MagicMock()
    tok.decode = MagicMock(return_value="hello world")
    tok.encode = MagicMock(return_value=[100, 200, 300])
    return tok


class TestTokenizerBridgeSame:
    """Test behavior when source and target are the same object."""

    def test_same_tokenizer_passthrough(self, source_tokenizer):
        bridge = TokenizerBridge(source_tokenizer, source_tokenizer)
        assert bridge.requires_mapping is False

    def test_same_tokenizer_map_tokens_identity(self, source_tokenizer):
        bridge = TokenizerBridge(source_tokenizer, source_tokenizer)
        ids = [1, 2, 3]
        result = bridge.map_tokens(ids, direction="source_to_target")
        assert result is ids  # Same object, no mapping
        # decode/encode should NOT have been called
        source_tokenizer.decode.assert_not_called()
        source_tokenizer.encode.assert_not_called()

    def test_same_tokenizer_map_tokens_reverse(self, source_tokenizer):
        bridge = TokenizerBridge(source_tokenizer, source_tokenizer)
        ids = [4, 5, 6]
        result = bridge.map_tokens(ids, direction="target_to_source")
        assert result is ids


class TestTokenizerBridgeDifferent:
    """Test behavior when source and target are different tokenizers."""

    def test_requires_mapping(self, source_tokenizer, target_tokenizer):
        bridge = TokenizerBridge(source_tokenizer, target_tokenizer)
        assert bridge.requires_mapping is True

    def test_map_tokens_source_to_target(self, source_tokenizer, target_tokenizer):
        bridge = TokenizerBridge(source_tokenizer, target_tokenizer)
        input_ids = [1, 2, 3]
        result = bridge.map_tokens(input_ids, direction="source_to_target")

        # Should decode with source, encode with target
        source_tokenizer.decode.assert_called_once_with(input_ids, skip_special_tokens=True)
        target_tokenizer.encode.assert_called_once_with("hello world", add_special_tokens=False)
        assert result == [100, 200, 300]

    def test_map_tokens_target_to_source(self, source_tokenizer, target_tokenizer):
        bridge = TokenizerBridge(source_tokenizer, target_tokenizer)
        input_ids = [100, 200, 300]
        result = bridge.map_tokens(input_ids, direction="target_to_source")

        # Should decode with target, encode with source
        target_tokenizer.decode.assert_called_once_with(input_ids, skip_special_tokens=True)
        source_tokenizer.encode.assert_called_once_with("hello world", add_special_tokens=False)
        assert result == [10, 20, 30]


class TestTokenizerBridgeMapSingleToken:
    """Test map_single_token."""

    def test_map_single_token(self, source_tokenizer, target_tokenizer):
        bridge = TokenizerBridge(source_tokenizer, target_tokenizer)
        result = bridge.map_single_token(1, direction="source_to_target")
        # map_single_token wraps in list, maps, returns first element
        assert result == 100

    def test_map_single_token_empty_result_fallback(self, source_tokenizer, target_tokenizer):
        """When mapping produces empty list, falls back to original token_id."""
        target_tokenizer.encode = MagicMock(return_value=[])
        bridge = TokenizerBridge(source_tokenizer, target_tokenizer)
        result = bridge.map_single_token(42, direction="source_to_target")
        assert result == 42

    def test_map_single_token_same_tokenizer(self, source_tokenizer):
        bridge = TokenizerBridge(source_tokenizer, source_tokenizer)
        result = bridge.map_single_token(7)
        assert result == 7


class TestTokenizerBridgeHelpers:
    """Test convenience methods."""

    def test_map_text_to_target(self, source_tokenizer, target_tokenizer):
        bridge = TokenizerBridge(source_tokenizer, target_tokenizer)
        result = bridge.map_text_to_target("some text")
        target_tokenizer.encode.assert_called_once_with("some text", add_special_tokens=False)
        assert result == [100, 200, 300]

    def test_decode_source(self, source_tokenizer, target_tokenizer):
        bridge = TokenizerBridge(source_tokenizer, target_tokenizer)
        result = bridge.decode_source([1, 2])
        source_tokenizer.decode.assert_called_once_with([1, 2], skip_special_tokens=True)
        assert result == "hello world"
