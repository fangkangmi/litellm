import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

import litellm
from litellm.litellm_core_utils.internal_params import INTERNAL_REQUEST_PARAMS
from litellm.llms.bedrock.chat.invoke_transformations.base_invoke_transformation import (
    AmazonInvokeConfig,
)
from litellm.llms.bedrock.common_utils import BedrockError
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from tests._support.stream_chunk_size import DEFAULT_CHUNKING_REQUESTS, ROUTER_CHUNK_SIZE_CASES, keys_at_every_depth


def test_mistral_completion_keeps_mcp_state_out_of_signed_body() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content) == {"prompt": "<s>[INST] hi [/INST]\n", "max_tokens": 10}
        assert request.headers["authorization"].startswith("AWS4-HMAC-SHA256 ")
        return httpx.Response(200, json={"outputs": [{"text": "hello", "stop_reason": "stop"}]})

    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        result: Final = litellm.completion(
            model="bedrock/invoke/mistral.mistral-7b-instruct-v0:2",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=10,
            skip_mcp_handler=True,
            _skip_mcp_handler=True,
            mcp_handler_context={"request_id": "internal"},
            client=HTTPHandler(client=transport),
            aws_access_key_id="fake",
            aws_secret_access_key="fake",
            aws_region_name="us-east-1",
        )
    assert result.choices[0].message.content == "hello"


def test_titan_transform_filters_internal_inference_parameters() -> None:
    parameters: Final = {
        "maxTokenCount": 10,
        **dict.fromkeys(INTERNAL_REQUEST_PARAMS, True),
        "cache_control_injection_points": [],
    }
    result: Final = AmazonInvokeConfig().transform_request(
        model="amazon.titan-text-express-v1",
        messages=[{"role": "user", "content": "hi"}],
        optional_params=parameters,
        litellm_params={},
        headers={},
    )
    assert result == {"inputText": "\n\nUser: hi\n\nBot: ", "textGenerationConfig": {"maxTokenCount": 10}}
    assert parameters == {
        "maxTokenCount": 10,
        **dict.fromkeys(INTERNAL_REQUEST_PARAMS, True),
        "cache_control_injection_points": [],
    }


@pytest.mark.parametrize(
    "model",
    ("anthropic.claude-v2:1", "us.amazon.nova-2-lite-v1:0", "twelvelabs.pegasus-1-2-v1:0", "openai.gpt-oss-120b-1:0"),
)
def test_delegated_invoke_transforms_filter_internal_parameters(model: str) -> None:
    config: Final = AmazonInvokeConfig()
    clean: Final = config.transform_request(
        model=model,
        messages=[{"role": "user", "content": "hi"}],
        optional_params={"max_tokens": 10},
        litellm_params={},
        headers={},
    )
    result: Final = config.transform_request(
        model=model,
        messages=[{"role": "user", "content": "hi"}],
        optional_params={
            "max_tokens": 10,
            **dict.fromkeys(INTERNAL_REQUEST_PARAMS, True),
            "cache_control_injection_points": [],
        },
        litellm_params={},
        headers={},
    )
    assert result == clean


def test_titan_extra_body_filters_only_the_inference_container() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content) == {
            "inputText": "\n\nUser: hi\n\nBot: ",
            "textGenerationConfig": {"maxTokenCount": 12},
            "extension": {"fake_stream": "user data"},
        }
        return httpx.Response(
            200, json={"results": [{"outputText": "hello", "tokenCount": 1}], "inputTextTokenCount": 1}
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        result: Final = litellm.completion(
            model="bedrock/invoke/amazon.titan-text-express-v1",
            messages=[{"role": "user", "content": "hi"}],
            extra_body={
                **dict.fromkeys(INTERNAL_REQUEST_PARAMS, True),
                "textGenerationConfig": {"maxTokenCount": 12, **dict.fromkeys(INTERNAL_REQUEST_PARAMS, True)},
                "extension": {"fake_stream": "user data"},
            },
            client=HTTPHandler(client=transport),
            api_key="synthetic-bearer",
            aws_region_name="us-east-1",
        )
    assert result.choices[0].message.content == "hello"


@pytest.mark.parametrize(
    "model",
    [
        "anthropic.claude-sonnet-4-6",
        "amazon.titan-text-express-v1",
        "mistral.mistral-7b-instruct-v0:2",
    ],
)
def test_completion_keeps_stream_chunk_size_out_of_invoke_bodies(model: str) -> None:
    send: Final = MagicMock(return_value=httpx.Response(200))
    client: Final = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(send)))

    litellm.completion(
        model=f"bedrock/invoke/{model}",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        max_tokens=10,
        client=client,
        aws_access_key_id="fake",
        aws_secret_access_key="fake",
        aws_region_name="us-east-1",
        stream_chunk_size=2048,
    )

    request: Final = send.call_args.args[0]
    assert "stream_chunk_size" not in keys_at_every_depth(json.loads(request.content)), request.content


def test_validate_environment_maps_guardrail_config_to_invoke_headers():
    """The InvokeModel API takes the guardrail identifier/version/trace as
    X-Amzn-Bedrock-* request headers, unlike Converse which takes them in the
    body. https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_InvokeModel.html"""
    optional_params = {
        "guardrailConfig": {
            "guardrailIdentifier": "ff6ujrregl1q",
            "guardrailVersion": "DRAFT",
            "trace": "enabled",
        },
        "max_tokens": 10,
    }

    headers = AmazonInvokeConfig().validate_environment(
        headers={},
        model="anthropic.claude-3-sonnet-20240229-v1:0",
        messages=[{"role": "user", "content": "hi"}],
        optional_params=optional_params,
        litellm_params={},
    )

    assert headers["X-Amzn-Bedrock-GuardrailIdentifier"] == "ff6ujrregl1q"
    assert headers["X-Amzn-Bedrock-GuardrailVersion"] == "DRAFT"
    assert headers["X-Amzn-Bedrock-Trace"] == "ENABLED"
    assert "guardrailConfig" not in optional_params


def test_validate_environment_without_guardrail_config_leaves_headers_untouched():
    headers = AmazonInvokeConfig().validate_environment(
        headers={"foo": "bar"},
        model="anthropic.claude-3-sonnet-20240229-v1:0",
        messages=[{"role": "user", "content": "hi"}],
        optional_params={"max_tokens": 10},
        litellm_params={},
    )

    assert headers == {"foo": "bar"}


def test_validate_environment_skips_absent_guardrail_fields():
    headers = AmazonInvokeConfig().validate_environment(
        headers={},
        model="amazon.titan-text-express-v1",
        messages=[{"role": "user", "content": "hi"}],
        optional_params={"guardrailConfig": {"guardrailIdentifier": "gr-id", "guardrailVersion": "1"}},
        litellm_params={},
    )

    assert headers == {
        "X-Amzn-Bedrock-GuardrailIdentifier": "gr-id",
        "X-Amzn-Bedrock-GuardrailVersion": "1",
    }


def test_validate_environment_does_not_clobber_explicit_guardrail_headers():
    """Users worked around the missing guardrailConfig support by passing the
    AWS headers directly; an explicit header must keep winning over
    guardrailConfig regardless of casing."""
    headers = AmazonInvokeConfig().validate_environment(
        headers={"x-amzn-bedrock-guardrailidentifier": "explicit-id"},
        model="anthropic.claude-3-sonnet-20240229-v1:0",
        messages=[{"role": "user", "content": "hi"}],
        optional_params={
            "guardrailConfig": {"guardrailIdentifier": "config-id", "guardrailVersion": "2"},
        },
        litellm_params={},
    )

    assert headers["x-amzn-bedrock-guardrailidentifier"] == "explicit-id"
    assert "X-Amzn-Bedrock-GuardrailIdentifier" not in headers
    assert headers["X-Amzn-Bedrock-GuardrailVersion"] == "2"


@pytest.mark.parametrize(
    "bad_guardrail_config",
    [
        {"guardrailIdentifier": "gr-id", "trace": "verbose"},
        {"guardrailIdentifier": ["gr-id"]},
        "gr-id",
        {},
        {"trace": "enabled"},
    ],
)
def test_validate_environment_rejects_malformed_guardrail_config(bad_guardrail_config):
    with pytest.raises(BedrockError) as excinfo:
        AmazonInvokeConfig().validate_environment(
            headers={},
            model="anthropic.claude-3-sonnet-20240229-v1:0",
            messages=[{"role": "user", "content": "hi"}],
            optional_params={"guardrailConfig": bad_guardrail_config},
            litellm_params={},
        )

    assert excinfo.value.status_code == 400
    assert "guardrailConfig" in str(excinfo.value)


@pytest.mark.parametrize(
    "model",
    [
        "anthropic.claude-3-sonnet-20240229-v1:0",
        "amazon.titan-text-express-v1",
        "mistral.mistral-7b-instruct-v0:2",
        "meta.llama3-8b-instruct-v1:0",
    ],
)
def test_guardrail_config_flows_to_headers_not_request_body(model):
    """Mirrors the handler flow (validate_environment then transform_request):
    guardrailConfig must end up in the signed headers and never leak into the
    request body, where Bedrock rejects it as an extra input."""
    config = AmazonInvokeConfig()
    optional_params = {
        "guardrailConfig": {
            "guardrailIdentifier": "ff6ujrregl1q",
            "guardrailVersion": "DRAFT",
            "trace": "disabled",
        },
        "max_tokens": 10,
    }
    messages = [{"role": "user", "content": "hi"}]

    headers = config.validate_environment(
        headers={},
        model=model,
        messages=messages,
        optional_params=optional_params,
        litellm_params={},
    )
    request_body = config.transform_request(
        model=model,
        messages=messages,
        optional_params=optional_params,
        litellm_params={},
        headers=headers,
    )

    assert "guardrailConfig" not in json.dumps(request_body)
    assert headers["X-Amzn-Bedrock-GuardrailIdentifier"] == "ff6ujrregl1q"
    assert headers["X-Amzn-Bedrock-GuardrailVersion"] == "DRAFT"
    assert headers["X-Amzn-Bedrock-Trace"] == "DISABLED"


def test_get_error_class_preserves_provider_headers():
    """The invoke handler path hands real provider headers to get_error_class (LIT-5428)."""
    error = AmazonInvokeConfig().get_error_class(
        error_message="Amazon Bedrock is unable to process your request.",
        status_code=500,
        headers={"x-amzn-RequestId": "req-invoke-500"},
    )

    assert isinstance(error, BedrockError)
    assert error.headers == {"x-amzn-RequestId": "req-invoke-500"}
    assert error.response.headers["x-amzn-requestid"] == "req-invoke-500"


def test_transform_response_hands_json_mode_to_nova():
    """The invoke dispatcher forwards its json_mode argument to Nova instead of dropping it."""
    from litellm.types.utils import ModelResponse

    response_json = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "tooluse_nova_json",
                            "name": "json_tool_call",
                            "input": {"city": "Paris", "temperature": 21},
                        }
                    }
                ],
            }
        },
        "stopReason": "tool_use",
        "usage": {"inputTokens": 5, "outputTokens": 4, "totalTokens": 9},
    }
    raw_response = httpx.Response(200, json=response_json, request=httpx.Request("POST", "https://bedrock"))

    result = AmazonInvokeConfig().transform_response(
        model="invoke/amazon.nova-lite-v1:0",
        raw_response=raw_response,
        model_response=ModelResponse(),
        logging_obj=MagicMock(),
        request_data={},
        messages=[{"role": "user", "content": "weather"}],
        optional_params={},
        litellm_params={},
        encoding=None,
        api_key=None,
        json_mode=True,
    )

    assert result.choices[0].message.tool_calls is None
    assert json.loads(result.choices[0].message.content) == {"city": "Paris", "temperature": 21}


def _stream_invoke_completion_with_spied_client(**kwargs: object) -> tuple[MagicMock, MagicMock]:
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.iter_bytes = MagicMock(return_value=iter([]))
    client = HTTPHandler()
    client.post = MagicMock(return_value=mock_response)

    litellm.completion(
        model="bedrock/invoke/anthropic.claude-haiku-4-5-20251001-v1:0",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        client=client,
        aws_access_key_id="fake",
        aws_secret_access_key="fake",
        aws_region_name="us-east-1",
        **kwargs,
    )
    return mock_response.iter_bytes, client.post


def test_completion_stream_chunk_size_reaches_iter_bytes_but_not_invoke_body() -> None:
    iter_bytes_spy, post_spy = _stream_invoke_completion_with_spied_client(stream_chunk_size=64)

    iter_bytes_spy.assert_called_once_with(chunk_size=64)
    data: Final = post_spy.call_args.kwargs["data"]
    assert "stream_chunk_size" not in keys_at_every_depth(json.loads(data)), data


@pytest.mark.parametrize("request_kwargs", DEFAULT_CHUNKING_REQUESTS)
def test_completion_uses_default_chunking_unless_a_valid_size_is_requested(
    request_kwargs: Mapping[str, object],
) -> None:
    iter_bytes_spy, _ = _stream_invoke_completion_with_spied_client(**request_kwargs)

    iter_bytes_spy.assert_called_once_with(chunk_size=None)


async def _astream_invoke_completion_with_spied_client(**kwargs: object) -> tuple[MagicMock, AsyncMock]:
    async def _no_bytes():
        return
        yield b""

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.aiter_bytes = MagicMock(return_value=_no_bytes())
    aiter_bytes_spy = mock_response.aiter_bytes
    client = AsyncHTTPHandler()
    client.post = AsyncMock(return_value=mock_response)

    await litellm.acompletion(
        model="bedrock/invoke/anthropic.claude-haiku-4-5-20251001-v1:0",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        client=client,
        aws_access_key_id="fake",
        aws_secret_access_key="fake",
        aws_region_name="us-east-1",
        **kwargs,
    )
    return aiter_bytes_spy, client.post


@pytest.mark.asyncio
async def test_acompletion_stream_chunk_size_reaches_aiter_bytes_but_not_invoke_body() -> None:
    aiter_bytes_spy, post_spy = await _astream_invoke_completion_with_spied_client(stream_chunk_size=64)

    aiter_bytes_spy.assert_called_once_with(chunk_size=64)
    data: Final = post_spy.call_args.kwargs["data"]
    assert "stream_chunk_size" not in keys_at_every_depth(json.loads(data)), data


@pytest.mark.asyncio
@pytest.mark.parametrize("request_kwargs", DEFAULT_CHUNKING_REQUESTS)
async def test_acompletion_uses_default_chunking_unless_a_valid_size_is_requested(
    request_kwargs: Mapping[str, object],
) -> None:
    aiter_bytes_spy, _ = await _astream_invoke_completion_with_spied_client(**request_kwargs)

    aiter_bytes_spy.assert_called_once_with(chunk_size=None)


INVOKE_DEPLOYMENT: Final = MappingProxyType(
    {
        "model": "bedrock/invoke/anthropic.claude-haiku-4-5-20251001-v1:0",
        "aws_access_key_id": "fake",
        "aws_secret_access_key": "fake",
        "aws_region_name": "us-east-1",
    }
)


@pytest.mark.parametrize("deployment_extras,expected_chunk_size", ROUTER_CHUNK_SIZE_CASES)
def test_router_deployment_stream_chunk_size_reaches_iter_bytes(
    deployment_extras: Mapping[str, object], expected_chunk_size: int | None
) -> None:
    mock_response: Final = MagicMock()
    mock_response.status_code = 200
    mock_response.iter_bytes = MagicMock(return_value=iter([]))
    client: Final = HTTPHandler()
    client.post = MagicMock(return_value=mock_response)
    router: Final = litellm.Router(
        model_list=[{"model_name": "invoke-chunked", "litellm_params": {**INVOKE_DEPLOYMENT, **deployment_extras}}]
    )

    router.completion(
        model="invoke-chunked",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        client=client,
    )

    mock_response.iter_bytes.assert_called_once_with(chunk_size=expected_chunk_size)
    data: Final = client.post.call_args.kwargs["data"]
    assert "stream_chunk_size" not in keys_at_every_depth(json.loads(data)), data


def test_invoke_stream_rejects_non_int_stream_chunk_size_before_calling_bedrock() -> None:
    send: Final = MagicMock(return_value=httpx.Response(200))
    client: Final = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(send)))

    with pytest.raises(litellm.BadRequestError):
        litellm.completion(
            model="bedrock/invoke/anthropic.claude-haiku-4-5-20251001-v1:0",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
            client=client,
            aws_access_key_id="fake",
            aws_secret_access_key="fake",
            aws_region_name="us-east-1",
            stream_chunk_size="sixty-four",
        )

    send.assert_not_called()


def test_router_deployment_with_a_non_numeric_stream_chunk_size_gets_a_400_before_calling_bedrock() -> None:
    send: Final = MagicMock(return_value=httpx.Response(200))
    client: Final = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(send)))
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "invoke-chunked",
                "litellm_params": {**INVOKE_DEPLOYMENT, "stream_chunk_size": "sixty-four"},
            }
        ]
    )

    with pytest.raises(litellm.BadRequestError) as exc_info:
        router.completion(
            model="invoke-chunked",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
            client=client,
        )

    assert exc_info.value.status_code == 400
    send.assert_not_called()


def test_delegated_nova_keeps_inline_cache_and_ignores_tool_cache_injection() -> None:
    model: Final = "us.amazon.nova-2-lite-v1:0"
    tools: Final = [
        {"type": "function", "function": {"name": "inspect", "parameters": {"type": "object", "properties": {}}}}
    ]
    messages: Final = [
        {"role": "system", "content": [{"type": "text", "text": "system", "cache_control": {"type": "ephemeral"}}]},
        {"role": "user", "content": "hi"},
    ]
    expected: Final = litellm.AmazonInvokeNovaConfig().transform_request(
        model=model,
        messages=messages,
        optional_params={"tools": tools, "maxTokens": 10},
        litellm_params={},
        headers={},
    )
    result: Final = AmazonInvokeConfig().transform_request(
        model=model,
        messages=messages,
        optional_params={
            **dict.fromkeys(INTERNAL_REQUEST_PARAMS, True),
            "tools": tools,
            "maxTokens": 10,
            "cache_control_injection_points": [{"location": "tool_config"}],
        },
        litellm_params={},
        headers={},
    )
    assert result == expected
    assert result["system"] == [{"text": "system", "cachePoint": {"type": "default"}}]
    assert len(result["toolConfig"]["tools"]) == len(tools)
