"""Integration tests for the REST API server."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from momo_kibidango.api.server import InferenceServer
from momo_kibidango.core.decoder import BaseDecoder, GenerationRequest, GenerationResult
from momo_kibidango.monitoring.metrics import MetricsCollector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_canned_result(text: str = "Generated response") -> GenerationResult:
    """Return a canned GenerationResult for testing."""
    return GenerationResult(
        text=text,
        tokens_generated=10,
        elapsed_seconds=0.5,
        tokens_per_second=20.0,
        acceptance_rate=0.8,
        stage_acceptance_rates={"stage2": 0.8},
        peak_memory_gb=2.0,
        mode="2model",
        draft_attempts=12,
        accepted_tokens=10,
    )


def _make_server(is_loaded: bool = True) -> tuple[InferenceServer, MagicMock]:
    """Build an InferenceServer with a mocked decoder."""
    decoder = MagicMock(spec=BaseDecoder)
    decoder.is_loaded = is_loaded
    decoder.mode = "2model"
    decoder.generate.return_value = _make_canned_result()

    metrics = MetricsCollector()
    server = InferenceServer(decoder=decoder, metrics=metrics)
    return server, decoder


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_health_endpoint(self):
        server, _ = _make_server()
        client = server.app.test_client()

        response = client.get("/health")
        data = response.get_json()

        assert response.status_code == 200
        assert data["status"] == "healthy"


class TestReadyEndpoint:
    def test_ready_when_loaded(self):
        server, _ = _make_server(is_loaded=True)
        client = server.app.test_client()

        response = client.get("/ready")
        data = response.get_json()

        assert response.status_code == 200
        assert data["status"] == "ready"

    def test_ready_when_not_loaded(self):
        server, _ = _make_server(is_loaded=False)
        client = server.app.test_client()

        response = client.get("/ready")
        data = response.get_json()

        assert response.status_code == 503
        assert data["status"] == "not_ready"


class TestInferEndpoint:
    def test_infer_endpoint(self):
        server, decoder = _make_server()
        client = server.app.test_client()

        response = client.post(
            "/infer",
            data=json.dumps({"prompt": "Hello world"}),
            content_type="application/json",
        )
        data = response.get_json()

        assert response.status_code == 200
        assert data["text"] == "Generated response"
        assert data["tokens_generated"] == 10
        assert data["mode"] == "2model"
        decoder.generate.assert_called_once()

    def test_infer_with_parameters(self):
        server, decoder = _make_server()
        client = server.app.test_client()

        response = client.post(
            "/infer",
            data=json.dumps({
                "prompt": "Write code",
                "max_tokens": 128,
                "temperature": 0.5,
            }),
            content_type="application/json",
        )

        assert response.status_code == 200
        call_args = decoder.generate.call_args[0][0]
        assert isinstance(call_args, GenerationRequest)
        assert call_args.prompt == "Write code"
        assert call_args.max_new_tokens == 128
        assert call_args.temperature == 0.5

    def test_infer_missing_prompt(self):
        server, _ = _make_server()
        client = server.app.test_client()

        response = client.post(
            "/infer",
            data=json.dumps({"max_tokens": 100}),
            content_type="application/json",
        )

        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data

    def test_infer_empty_prompt(self):
        server, _ = _make_server()
        client = server.app.test_client()

        response = client.post(
            "/infer",
            data=json.dumps({"prompt": ""}),
            content_type="application/json",
        )

        assert response.status_code == 400

    def test_infer_no_json_body(self):
        server, _ = _make_server()
        client = server.app.test_client()

        response = client.post("/infer", data="not json")

        assert response.status_code == 400


class TestMetricsEndpoint:
    def test_metrics_endpoint(self):
        server, _ = _make_server()
        client = server.app.test_client()

        response = client.get("/metrics")
        data = response.get_json()

        assert response.status_code == 200
        assert "inference_count" in data
        assert "total_tokens" in data
        assert "throughput" in data

    def test_metrics_after_inference(self):
        server, _ = _make_server()
        client = server.app.test_client()

        # Make an inference call first
        client.post(
            "/infer",
            data=json.dumps({"prompt": "Test"}),
            content_type="application/json",
        )

        response = client.get("/metrics")
        data = response.get_json()

        # Metrics should reflect the inference call
        assert data["inference_count"] >= 1


class TestBatchEndpoint:
    def test_batch_endpoint(self):
        server, decoder = _make_server()
        client = server.app.test_client()

        response = client.post(
            "/batch",
            data=json.dumps({
                "prompts": ["Hello", "World", "Test"],
            }),
            content_type="application/json",
        )
        data = response.get_json()

        assert response.status_code == 200
        assert data["batch_size"] == 3
        assert len(data["results"]) == 3
        assert decoder.generate.call_count == 3

        for result in data["results"]:
            assert "text" in result
            assert result["text"] == "Generated response"

    def test_batch_empty_list(self):
        server, _ = _make_server()
        client = server.app.test_client()

        response = client.post(
            "/batch",
            data=json.dumps({"prompts": []}),
            content_type="application/json",
        )

        assert response.status_code == 400

    def test_batch_missing_prompts(self):
        server, _ = _make_server()
        client = server.app.test_client()

        response = client.post(
            "/batch",
            data=json.dumps({"data": "wrong"}),
            content_type="application/json",
        )

        assert response.status_code == 400
