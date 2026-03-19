"""Tests for core ORS data types."""

import pytest
from ors.types import (
    TextBlock,
    ImageBlock,
    ToolOutput,
    ToolSpec,
    Split,
    RunToolSuccess,
    RunToolError,
    RunToolOutput,
    CreateSession,
    ListToolsOutput,
    ToolCall,
    ListTasks,
    NumTasks,
    GetTask,
    GetTaskRange,
)


class TestTextBlock:
    def test_basic(self):
        block = TextBlock(text="hello")
        assert block.text == "hello"
        assert block.type == "text"
        assert block.detail is None

    def test_with_detail(self):
        block = TextBlock(text="hello", detail={"key": "value"})
        assert block.detail == {"key": "value"}

    def test_serialize(self):
        block = TextBlock(text="hello")
        d = block.model_dump()
        assert d == {"text": "hello", "detail": None, "type": "text"}


class TestImageBlock:
    def test_basic(self):
        block = ImageBlock(data="base64data", mimeType="image/png")
        assert block.data == "base64data"
        assert block.mimeType == "image/png"
        assert block.type == "image"

    def test_serialize(self):
        block = ImageBlock(data="abc", mimeType="image/jpeg")
        d = block.model_dump()
        assert d["type"] == "image"
        assert d["data"] == "abc"
        assert d["mimeType"] == "image/jpeg"


class TestToolOutput:
    def test_defaults(self):
        out = ToolOutput(blocks=[TextBlock(text="ok")])
        assert out.reward is None
        assert out.finished is False
        assert out.metadata is None

    def test_with_reward(self):
        out = ToolOutput(
            blocks=[TextBlock(text="correct")],
            reward=1.0,
            finished=True,
            metadata={"score": 100},
        )
        assert out.reward == 1.0
        assert out.finished is True
        assert out.metadata == {"score": 100}


class TestRunToolOutput:
    def test_success(self):
        success = RunToolSuccess(output=ToolOutput(blocks=[TextBlock(text="ok")]))
        result = RunToolOutput(root=success)
        d = result.model_dump()
        assert d["ok"] is True
        assert d["output"]["blocks"][0]["text"] == "ok"

    def test_error(self):
        error = RunToolError(error="something went wrong")
        result = RunToolOutput(root=error)
        d = result.model_dump()
        assert d["ok"] is False
        assert d["error"] == "something went wrong"


class TestToolSpec:
    def test_basic(self):
        spec = ToolSpec(name="submit", description="Submit answer", input_schema=None)
        assert spec.name == "submit"
        assert spec.input_schema is None

    def test_with_schema(self):
        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
        spec = ToolSpec(name="submit", description="Submit", input_schema=schema)
        assert spec.input_schema is not None


class TestSplit:
    def test_types(self):
        for t in ["train", "validation", "test"]:
            s = Split(name=t, type=t)
            assert s.type == t

    def test_invalid_type(self):
        with pytest.raises(Exception):
            Split(name="custom", type="invalid")


class TestCreateSession:
    def test_with_task_spec(self):
        cs = CreateSession(task_spec={"question": "2+2?"})
        assert cs.task_spec is not None

    def test_with_split_index(self):
        cs = CreateSession(split="train", index=0)
        assert cs.split == "train"
        assert cs.index == 0

    def test_neither_fails(self):
        with pytest.raises(Exception):
            CreateSession()

    def test_both_fails(self):
        with pytest.raises(Exception):
            CreateSession(task_spec={"q": "?"}, split="train", index=0)

    def test_partial_split_index_fails(self):
        with pytest.raises(Exception):
            CreateSession(split="train")


class TestToolCall:
    def test_basic(self):
        tc = ToolCall(name="submit", input={"answer": "42"})
        assert tc.name == "submit"
        assert tc.task_id is None

    def test_with_task_id(self):
        tc = ToolCall(name="submit", input={}, task_id="abc123")
        assert tc.task_id == "abc123"


class TestListToolsOutput:
    def test_empty(self):
        lo = ListToolsOutput(tools=[])
        assert len(lo.tools) == 0

    def test_with_tools(self):
        lo = ListToolsOutput(tools=[
            ToolSpec(name="t1", description="Tool 1", input_schema=None),
            ToolSpec(name="t2", description="Tool 2", input_schema={"type": "object"}),
        ])
        assert len(lo.tools) == 2
