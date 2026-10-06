from collections.abc import Mapping
from typing import Final, TypeVar

MCP_INTERNAL_PARAMS: Final = frozenset(("skip_mcp_handler", "_skip_mcp_handler", "mcp_handler_context"))
INTERNAL_REQUEST_PARAMS: Final = MCP_INTERNAL_PARAMS | frozenset(
    ("stream_chunk_size", "fake_stream", "cache_control_injection_points")
)
_PRE_TRANSFORM_PARAMS: Final = INTERNAL_REQUEST_PARAMS - frozenset(("cache_control_injection_points",))
_Value: Final = TypeVar("_Value")


def filter_chat_params(
    params: Mapping[str, _Value],
) -> dict[str, _Value]:  # mutable-ok: provider transforms consume a fresh dict
    return {key: value for key, value in params.items() if key not in _PRE_TRANSFORM_PARAMS}


def filter_provider_body(
    params: Mapping[str, _Value],
) -> dict[str, _Value]:  # mutable-ok: request handlers require a fresh dict for signing and dispatch
    return {key: value for key, value in params.items() if key not in INTERNAL_REQUEST_PARAMS}
