from typing import Any, Literal, Mapping, Optional, Sequence, Union

from pydantic import BaseModel, Field, RootModel, model_validator
from typing_extensions import TypeAliasType

JSONValue = TypeAliasType("JSONValue", Union[Mapping[str, Any], Sequence[Any], str, int, float, bool, None])
JSONObject = Mapping[str, JSONValue]


class TextBlock(BaseModel, extra="forbid"):
    text: str
    detail: Optional[JSONObject] = None
    type: Literal["text"] = "text"

class ImageBlock(BaseModel, extra="forbid"):
    data: str
    mimeType: str
    detail: Optional[JSONObject] = None
    type: Literal["image"] = "image"

Blocks = TypeAliasType("Blocks", Sequence[Union[TextBlock, ImageBlock]])

class ToolOutput(BaseModel, extra="forbid"):
    blocks: Blocks
    metadata: Optional[JSONObject] = None
    reward: Optional[float] = None
    finished: bool = False

class RunToolSuccess(BaseModel, extra="forbid"):
    ok: Literal[True] = True
    output: ToolOutput

class RunToolError(BaseModel, extra="forbid"):
    ok: Literal[False] = False
    error: str

class RunToolOutput(RootModel[Union[RunToolSuccess, RunToolError]]):
    root: Union[RunToolSuccess, RunToolError] = Field(discriminator="ok")

class ToolSpec(BaseModel, extra="forbid"):
    name: str
    description: str
    input_schema: Optional[JSONObject]

class ListToolsOutput(BaseModel, extra="forbid"):
    tools: list[ToolSpec]

class ToolCall(BaseModel, extra="forbid"):
    name: str
    input: JSONObject
    task_id: Optional[str] = None

class CreateSession(BaseModel, extra="forbid"):
    env_name: Optional[str] = None
    task_spec: Optional[JSONObject] = None
    split: Optional[str] = None
    index: Optional[int] = None
    secrets: Optional[Mapping[str, str]] = None

    @model_validator(mode="after")
    def check_task_source(self) -> "CreateSession":
        """Exactly one of task_spec or (split, index) must be provided."""
        has_spec = self.task_spec is not None
        has_index = self.split is not None and self.index is not None
        if has_spec == has_index:
            raise ValueError("Provide either task_spec or both split and index, not both/neither")
        if (self.split is None) != (self.index is None):
            raise ValueError("split and index must both be provided together")
        return self

class ListTasks(BaseModel, extra="forbid"):
    split: str

class NumTasks(BaseModel, extra="forbid"):
    split: str

class GetTask(BaseModel, extra="forbid"):
    split: str
    index: int

class GetTaskRange(BaseModel, extra="forbid"):
    split: str
    start: Optional[int] = None
    stop: Optional[int] = None

SplitType = Literal["train", "validation", "test"]

class Split(BaseModel, extra="forbid"):
    name: str
    type: SplitType
