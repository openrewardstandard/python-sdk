from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Literal, Union

from typing_extensions import TypeAliasType

JSONValue = TypeAliasType("JSONValue", Union[Mapping[str, Any], Sequence[Any], str, int, float, bool, None])
JSONObject = Mapping[str, JSONValue]


@dataclass
class Task:
    """Represents a task from an ORS environment."""
    environment_name: str
    task_spec: JSONObject


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: Optional[JSONObject]


@dataclass
class TextBlock:
    text: str
    detail: Optional[JSONObject] = None
    type: Literal["text"] = "text"


@dataclass
class ImageBlock:
    data: str
    mimeType: str
    detail: Optional[JSONObject] = None
    type: Literal["image"] = "image"


@dataclass
class ToolOutput:
    blocks: list[Union[TextBlock, ImageBlock]]
    metadata: Optional[JSONObject] = None
    reward: Optional[float] = None
    finished: bool = False


class ToolCallError(Exception):
    pass


class AuthenticationError(Exception):
    """Raised when API authentication fails (401 Unauthorized)"""
    pass


Provider = Literal["openai", "anthropic", "google", "openrouter"]
