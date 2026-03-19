"""Tests for the Environment ABC and @tool decorator."""

import pytest
from pydantic import BaseModel

from ors import Environment, tool, ToolOutput, TextBlock, Split
from ors.types import ListToolsOutput


class SubmitInput(BaseModel):
    answer: str


class SimpleEnv(Environment):
    @classmethod
    def list_splits(cls):
        return [Split(name="test", type="test")]

    @classmethod
    def list_tasks(cls, split: str):
        return [{"question": "What is 2+2?", "answer": "4"}]

    def get_prompt(self):
        return [TextBlock(text=self.task_spec["question"])]

    @tool
    def submit(self, params: SubmitInput) -> ToolOutput:
        """Submit your answer"""
        correct = params.answer.strip() == self.task_spec["answer"]
        return ToolOutput(
            blocks=[TextBlock(text="Correct!" if correct else "Incorrect")],
            reward=1.0 if correct else 0.0,
            finished=True,
        )


class NoParamToolEnv(Environment):
    @classmethod
    def list_splits(cls):
        return ["test"]

    @classmethod
    def list_tasks(cls, split: str):
        return [{"value": 42}]

    def get_prompt(self):
        return [TextBlock(text="Call get_value")]

    @tool
    def get_value(self) -> ToolOutput:
        """Get the value"""
        return ToolOutput(
            blocks=[TextBlock(text=str(self.task_spec["value"]))],
        )


class TestToolDecorator:
    def test_marks_function(self):
        @tool
        def my_tool(self) -> ToolOutput:
            pass
        assert getattr(my_tool, "_env_tool", False) is True
        assert getattr(my_tool, "_env_tool_shared", False) is True

    def test_shared_false(self):
        @tool(shared=False)
        def my_tool(self) -> ToolOutput:
            pass
        assert getattr(my_tool, "_env_tool", False) is True
        assert getattr(my_tool, "_env_tool_shared", True) is False


class TestEnvironment:
    def test_list_tools(self):
        tools = SimpleEnv.list_tools()
        assert isinstance(tools, ListToolsOutput)
        assert len(tools.tools) == 1
        assert tools.tools[0].name == "submit"
        assert tools.tools[0].description == "Submit your answer"
        assert tools.tools[0].input_schema is not None

    def test_list_tools_no_params(self):
        tools = NoParamToolEnv.list_tools()
        assert len(tools.tools) == 1
        assert tools.tools[0].name == "get_value"
        assert tools.tools[0].input_schema is None

    def test_name(self):
        assert SimpleEnv.name() == "SimpleEnv"

    def test_list_splits(self):
        splits = SimpleEnv.list_splits()
        assert len(splits) == 1

    def test_list_tasks(self):
        tasks = SimpleEnv.list_tasks("test")
        assert len(tasks) == 1
        assert tasks[0]["question"] == "What is 2+2?"

    def test_get_prompt(self):
        env = SimpleEnv(task_spec={"question": "What is 2+2?", "answer": "4"})
        prompt = env.get_prompt()
        assert len(prompt) == 1
        assert prompt[0].text == "What is 2+2?"

    @pytest.mark.asyncio
    async def test_call_tool_correct(self):
        env = SimpleEnv(task_spec={"question": "What is 2+2?", "answer": "4"})
        result = await env._call_tool("submit", {"answer": "4"})
        d = result.model_dump()
        assert d["ok"] is True
        assert d["output"]["reward"] == 1.0
        assert d["output"]["finished"] is True

    @pytest.mark.asyncio
    async def test_call_tool_incorrect(self):
        env = SimpleEnv(task_spec={"question": "What is 2+2?", "answer": "4"})
        result = await env._call_tool("submit", {"answer": "5"})
        d = result.model_dump()
        assert d["ok"] is True
        assert d["output"]["reward"] == 0.0

    @pytest.mark.asyncio
    async def test_call_tool_not_found(self):
        env = SimpleEnv(task_spec={"question": "x", "answer": "y"})
        result = await env._call_tool("nonexistent", {})
        d = result.model_dump()
        assert d["ok"] is False
        assert "not a valid tool" in d["error"]

    @pytest.mark.asyncio
    async def test_call_tool_validation_error(self):
        env = SimpleEnv(task_spec={"question": "x", "answer": "y"})
        result = await env._call_tool("submit", {"wrong_field": "value"})
        d = result.model_dump()
        assert d["ok"] is False
        assert "validation error" in d["error"].lower()

    @pytest.mark.asyncio
    async def test_num_tasks(self):
        count = await SimpleEnv.num_tasks("test")
        assert count == 1

    @pytest.mark.asyncio
    async def test_get_task(self):
        task = await SimpleEnv.get_task("test", 0)
        assert task["question"] == "What is 2+2?"

    @pytest.mark.asyncio
    async def test_get_task_range(self):
        tasks = await SimpleEnv.get_task_range("test", 0, 1)
        assert len(tasks) == 1
