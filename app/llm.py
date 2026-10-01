"""Claude calls that must return schema-valid JSON.

Same backend as Newspaper_ingestion: the Claude API with native structured
output (`output_config.format`), validated again with `jsonschema` as a
defensive check. Every feature module (tagging, analysis, quizzes...) builds
its own prompt and schema and calls `generate_json`.

Requests opt into server-side refusal fallbacks (`fallbacks: "default"`): if
the safety classifiers decline a request - which can happen on hard news
such as attacks or security incidents - the API re-runs it on Anthropic's
recommended fallback model inside the same call instead of failing.
"""
from __future__ import annotations

import json

import anthropic
import jsonschema

from app.config import settings

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMGenerationError(RuntimeError):
    """Raised when Claude can't be reached or returns an unusable response."""


class SchemaValidationError(RuntimeError):
    """Raised when Claude's output doesn't validate against the schema."""


def strict_schema(schema):
    """Deep-copy `schema`, adding `"additionalProperties": false` to every
    object subschema that doesn't already specify it - Claude's
    `output_config.format.schema` rejects object subschemas without it."""
    if isinstance(schema, dict):
        result = {k: strict_schema(v) for k, v in schema.items()}
        if result.get("type") == "object" and "additionalProperties" not in result:
            result["additionalProperties"] = False
        return result
    if isinstance(schema, list):
        return [strict_schema(item) for item in schema]
    return schema


_anthropic_client: anthropic.Anthropic | None = None


def _get_anthropic_client() -> anthropic.Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        try:
            # Resolves credentials from ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN
            # / an `ant auth login` profile - never hardcode a key here.
            headers = (
                {"anthropic-workspace-id": settings.anthropic_workspace_id}
                if settings.anthropic_workspace_id
                else None
            )
            _anthropic_client = anthropic.Anthropic(default_headers=headers)
        except Exception as exc:
            raise LLMGenerationError(
                "Could not create an Anthropic client. Set ANTHROPIC_API_KEY "
                f"(or run `ant auth login`). Original error: {exc}"
            ) from exc
    return _anthropic_client


def _call_claude(system_prompt: str, prompt: str, output_schema: dict, effort: str) -> str:
    client = _get_anthropic_client()
    try:
        # Streamed so long outputs (a 20-question quiz) can't hit the HTTP
        # timeout; get_final_message() assembles the whole response.
        with client.beta.messages.stream(
            model=settings.anthropic_model,
            max_tokens=settings.anthropic_max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
            output_config={
                "effort": effort,
                "format": {"type": "json_schema", "schema": output_schema},
            },
            betas=[FALLBACK_BETA],
            fallbacks="default",
        ) as stream:
            response = stream.get_final_message()
    except anthropic.AuthenticationError as exc:
        raise LLMGenerationError(
            "Anthropic authentication failed - set ANTHROPIC_API_KEY (or run "
            f"`ant auth login`). Original error: {exc}"
        ) from exc
    except anthropic.NotFoundError as exc:
        raise LLMGenerationError(
            f"Anthropic model '{settings.anthropic_model}' was not found: {exc}"
        ) from exc
    except anthropic.RateLimitError as exc:
        raise LLMGenerationError(f"Anthropic rate limit hit: {exc}") from exc
    except anthropic.APIStatusError as exc:
        raise LLMGenerationError(f"Anthropic API error ({exc.status_code}): {exc.message}") from exc
    except anthropic.APIConnectionError as exc:
        raise LLMGenerationError(f"Could not reach the Anthropic API: {exc}") from exc

    if response.stop_reason == "refusal":
        raise LLMGenerationError("Claude declined this request, and so did the fallback model.")
    if response.stop_reason == "max_tokens":
        raise LLMGenerationError(
            "Claude's response was cut off at ANTHROPIC_MAX_TOKENS; raise it and retry."
        )
    text = next((block.text for block in response.content if block.type == "text"), None)
    if not text:
        raise LLMGenerationError(f"Claude returned no text content: {response.content}")
    return text


def generate_json(system_prompt: str, prompt: str, output_schema: dict, effort: str = "medium") -> dict:
    """Ask Claude for a JSON object matching `output_schema`."""
    output_schema = strict_schema(output_schema)
    raw_content = _call_claude(system_prompt, prompt, output_schema, effort)

    # `output_config.format` already guarantees schema-valid JSON, but we
    # validate anyway as a defensive check rather than trusting it blindly.
    try:
        parsed = json.loads(raw_content)
        jsonschema.validate(instance=parsed, schema=output_schema)
        return parsed
    except (json.JSONDecodeError, jsonschema.exceptions.ValidationError) as exc:
        raise SchemaValidationError(
            f"Claude's structured output unexpectedly failed schema validation: {exc}"
        ) from exc
