"""Unit tests for the momo-kibidango exception hierarchy."""

import pytest

from momo_kibidango.exceptions import (
    MomoError,
    ModelLoadError,
    ModelNotLoadedError,
    TokenizerMismatchError,
    ResourceExhaustedError,
    RateLimitExceededError,
    InvalidPromptError,
    GenerationTimeoutError,
    ConfigurationError,
)


# All custom exception classes that should inherit from MomoError
ALL_EXCEPTIONS = [
    ModelLoadError,
    ModelNotLoadedError,
    TokenizerMismatchError,
    ResourceExhaustedError,
    RateLimitExceededError,
    InvalidPromptError,
    GenerationTimeoutError,
    ConfigurationError,
]


class TestMomoError:
    """Test base MomoError."""

    def test_momo_error_is_exception(self):
        assert issubclass(MomoError, Exception)

    def test_momo_error_can_be_raised(self):
        with pytest.raises(MomoError):
            raise MomoError("test error")

    def test_momo_error_message(self):
        err = MomoError("something went wrong")
        assert str(err) == "something went wrong"

    def test_momo_error_caught_as_exception(self):
        with pytest.raises(Exception):
            raise MomoError("caught as base Exception")


class TestExceptionHierarchy:
    """Verify all exceptions inherit from MomoError."""

    @pytest.mark.parametrize("exc_class", ALL_EXCEPTIONS)
    def test_all_exceptions_inherit_momo_error(self, exc_class):
        assert issubclass(exc_class, MomoError)
        assert issubclass(exc_class, Exception)

    @pytest.mark.parametrize("exc_class", ALL_EXCEPTIONS)
    def test_exception_can_be_raised_and_caught_as_momo_error(self, exc_class):
        with pytest.raises(MomoError):
            raise exc_class("test")

    @pytest.mark.parametrize("exc_class", ALL_EXCEPTIONS)
    def test_exception_message(self, exc_class):
        msg = f"Error from {exc_class.__name__}"
        err = exc_class(msg)
        assert str(err) == msg

    @pytest.mark.parametrize("exc_class", ALL_EXCEPTIONS)
    def test_exception_caught_by_own_type(self, exc_class):
        with pytest.raises(exc_class):
            raise exc_class("specific catch")


class TestSpecificExceptions:
    """Test specific exception use cases."""

    def test_model_load_error(self):
        with pytest.raises(ModelLoadError, match="OOM"):
            raise ModelLoadError("OOM while loading model")

    def test_model_not_loaded_error(self):
        with pytest.raises(ModelNotLoadedError, match="not loaded"):
            raise ModelNotLoadedError("Model not loaded yet")

    def test_tokenizer_mismatch_error(self):
        with pytest.raises(TokenizerMismatchError, match="mismatch"):
            raise TokenizerMismatchError("Vocabulary mismatch")

    def test_resource_exhausted_error(self):
        with pytest.raises(ResourceExhaustedError, match="GPU"):
            raise ResourceExhaustedError("GPU memory exhausted")

    def test_rate_limit_exceeded_error(self):
        with pytest.raises(RateLimitExceededError, match="limit"):
            raise RateLimitExceededError("Rate limit exceeded")

    def test_invalid_prompt_error(self):
        with pytest.raises(InvalidPromptError, match="too long"):
            raise InvalidPromptError("Prompt too long")

    def test_generation_timeout_error(self):
        with pytest.raises(GenerationTimeoutError, match="timeout"):
            raise GenerationTimeoutError("Generation timeout after 300s")

    def test_configuration_error(self):
        with pytest.raises(ConfigurationError, match="missing"):
            raise ConfigurationError("Config key missing")
