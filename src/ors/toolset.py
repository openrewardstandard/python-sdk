"""Base class for toolsets that provide reusable collections of tools."""

from abc import ABC
from typing import Any


class Toolset(ABC):
    """
    Base class for toolsets. Toolsets are collections of related tools that
    can be easily reused across different environments.

    They follow the same @tool decorator pattern as Environment methods.

    Usage:
        from ors import Toolset, tool, ToolOutput, TextBlock

        class MyToolset(Toolset):
            @tool
            async def my_tool(self) -> ToolOutput:
                return ToolOutput(blocks=[TextBlock(text="Hello")])

        class MyEnv(Environment):
            toolsets = [MyToolset]
    """

    def __init__(self, env: Any):
        """
        Initialize toolset with environment reference.

        Args:
            env: The environment instance that owns this toolset
        """
        self.env = env
