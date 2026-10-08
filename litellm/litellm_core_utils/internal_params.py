from collections.abc import Mapping
from typing import Final, TypeVar

INTERNAL_REQUEST_PARAMS: Final = frozenset(
    (
        "skip_mcp_handler",
        "_skip_mcp_handler",
        "mcp_handler_context",
        "stream_chunk_size",
        "fake_stream",
        "cache_control_injection_points",
    )
)
_Value: Final = TypeVar("_Value")


def filter_provider_body(
    params: Mapping[str, _Value],
) -> dict[str, _Value]:  # mutable-ok: request handlers require a fresh dict for signing and dispatch
    return {key: value for key, value in params.items() if key not in INTERNAL_REQUEST_PARAMS}
