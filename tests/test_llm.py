import json
from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from app import llm
from app.config import settings

SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}


class _FakeStream:
    def __init__(self, response):
        self._response = response

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self._response


def _response(text, stop_reason="end_turn"):
    return SimpleNamespace(stop_reason=stop_reason, content=[SimpleNamespace(type="text", text=text)])


def _client(stream):
    return SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(stream=stream)))


def test_valid_json_response(monkeypatch):
    monkeypatch.setattr(
        llm, "_get_anthropic_client",
        lambda: _client(lambda **kw: _FakeStream(_response(json.dumps({"answer": "ok"})))),
    )
    assert llm.generate_json("sys", "q?", SCHEMA) == {"answer": "ok"}


def test_request_shape(monkeypatch):
    captured = {}

    def stream(**kwargs):
        captured.update(kwargs)
        return _FakeStream(_response(json.dumps({"answer": "ok"})))

    monkeypatch.setattr(llm, "_get_anthropic_client", lambda: _client(stream))
    llm.generate_json("sys", "q?", SCHEMA, effort="low")

    assert captured["model"] == settings.anthropic_model
    assert captured["system"] == "sys"
    assert captured["output_config"] == {
        "effort": "low",
        "format": {"type": "json_schema", "schema": {**SCHEMA, "additionalProperties": False}},
    }
    # Opted into server-side refusal fallbacks.
    assert captured["fallbacks"] == "default"
    assert captured["betas"] == ["server-side-fallback-2026-07-01"]


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
def test_unusable_stop_reasons_raise(monkeypatch, stop_reason):
    monkeypatch.setattr(
        llm, "_get_anthropic_client",
        lambda: _client(lambda **kw: _FakeStream(_response("{}", stop_reason))),
    )
    with pytest.raises(llm.LLMGenerationError):
        llm.generate_json("sys", "q?", SCHEMA)


def test_invalid_json_raises_schema_error(monkeypatch):
    monkeypatch.setattr(
        llm, "_get_anthropic_client", lambda: _client(lambda **kw: _FakeStream(_response("not json")))
    )
    with pytest.raises(llm.SchemaValidationError):
        llm.generate_json("sys", "q?", SCHEMA)


def test_authentication_error_maps_to_llm_generation_error(monkeypatch):
    def raise_auth_error(**kwargs):
        request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
        raise anthropic.AuthenticationError(
            "invalid key", response=httpx2.Response(401, request=request), body=None
        )

    monkeypatch.setattr(llm, "_get_anthropic_client", lambda: _client(raise_auth_error))
    with pytest.raises(llm.LLMGenerationError):
        llm.generate_json("sys", "q?", SCHEMA)


def test_strict_schema_recurses_into_nested_schemas():
    schema = {
        "type": "object",
        "properties": {
            "nested": {"type": "object", "properties": {"x": {"type": "string"}}},
            "already_set": {"type": "object", "additionalProperties": True},
            "items": {"type": "array", "items": {"type": "object", "properties": {}}},
        },
    }

    result = llm.strict_schema(schema)

    assert result["additionalProperties"] is False
    assert result["properties"]["nested"]["additionalProperties"] is False
    assert result["properties"]["already_set"]["additionalProperties"] is True
    assert result["properties"]["items"]["items"]["additionalProperties"] is False
    assert "additionalProperties" not in schema
