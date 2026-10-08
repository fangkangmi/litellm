from types import MappingProxyType
from typing import Final

from litellm.litellm_core_utils.internal_params import INTERNAL_REQUEST_PARAMS, filter_provider_body


def test_provider_filter_preserves_options_and_nested_user_content() -> None:
    nested: Final = MappingProxyType({key: "user data" for key in INTERNAL_REQUEST_PARAMS})
    retained: Final = {"dimensions": 256, "encoding_format": "float", "extension": nested}
    parameters: Final = MappingProxyType({**retained, **dict.fromkeys(INTERNAL_REQUEST_PARAMS, True)})

    result: Final = filter_provider_body(parameters)

    assert result == retained
    assert result["extension"] is nested
    assert parameters == {**retained, **dict.fromkeys(INTERNAL_REQUEST_PARAMS, True)}
