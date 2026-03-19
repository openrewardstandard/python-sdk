import asyncio
import base64
import json
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Optional, Union, overload

import aiohttp
from ors.client._http import (
    request_retryable,
    resumable_sse,
)
from ors.client._session import BaseAsyncSession, SessionTerminatedError
from ors.client._types import (
    ImageBlock,
    JSONObject,
    Mapping,
    Provider,
    Task,
    TextBlock,
    ToolCallError,
    ToolOutput,
    ToolSpec,
)
from ors.client.environment import convert_tool_response


def _build_secrets_header(secrets: dict[str, str]) -> str:
    """Build X-Secrets header from simple key-value pairs."""
    payload = {k: {"value": v} for k, v in secrets.items()}
    return base64.b64encode(json.dumps(payload).encode()).decode()


class AsyncSession(BaseAsyncSession):

    def __init__(
        self,
        env: "AsyncEnvironment",
        task: Optional[Task] = None,
        secrets: Optional[dict[str, str]] = None,
        api_key: Optional[str] = None,
        split: Optional[str] = None,
        index: Optional[int] = None,
    ):
        has_task = task is not None
        has_index = split is not None and index is not None
        if has_task == has_index:
            raise ValueError("Provide either task or both split and index, not both/neither")
        if (split is None) != (index is None):
            raise ValueError("split and index must both be provided together")

        creation_headers: Optional[dict[str, str]] = None
        if secrets:
            creation_headers = {"X-Secrets": _build_secrets_header(secrets)}

        super().__init__(
            base_url=str(env.client._base_url),
            api_key=api_key,
            creation_endpoint="/create_session",
            creation_payload={},
            client=env.client,
            creation_headers=creation_headers,
        )

        self._secrets_headers = creation_headers
        self.env = env
        self.task = task
        self.split = split
        self.index = index

        self._has_task_tools: bool = True

    def _env_path(self, suffix: str) -> str:
        """Build URL path using the environment name."""
        return f"/{self.env.name}{suffix}"

    async def _post_create(self) -> None:
        """POST /create with task payload after SID is obtained."""
        create_payload: dict[str, Any] = {}
        if self.task is not None:
            create_payload["task_spec"] = self.task.task_spec
            create_payload["env_name"] = self.task.environment_name
        else:
            create_payload["split"] = self.split
            create_payload["index"] = self.index

        await request_retryable(
            self.client,
            "POST",
            "/create",
            expect_json=True,
            sid=self.sid,
            json=create_payload,
            token=self.api_key,
            extra_headers=self._secrets_headers,
        )

    async def _pre_delete(self) -> None:
        """POST /delete to tear down the environment on the server."""
        if self.sid:
            await request_retryable(
                self.client,
                "POST",
                "/delete",
                expect_json=False,
                sid=self.sid,
                token=self.api_key,
            )

    async def get_prompt(self) -> list[Union[TextBlock, ImageBlock]]:
        res = await self._run_or_die(
            request_retryable(
                self.client,
                "GET",
                self._env_path("/prompt"),
                expect_json=True,
                sid=self.sid,
                token=self.api_key,
            )
        )
        blocks: list[Union[TextBlock, ImageBlock]] = []
        for block in res:
            if block["type"] == "text":
                blocks.append(TextBlock(text=block["text"], detail=block.get("detail")))
            elif block["type"] == "image":
                blocks.append(ImageBlock(mimeType=block["mimeType"], detail=block.get("detail"), data=block["data"]))
        return blocks

    @overload
    async def list_tools(self, format: None = None) -> list[ToolSpec]: ...

    @overload
    async def list_tools(self, format: Provider) -> list[dict]: ...

    async def list_tools(self, format: Optional[Provider] = None) -> Union[list[ToolSpec], list[dict]]:
        if self._has_task_tools:
            try:
                res = await self._run_or_die(
                    request_retryable(
                        self.client,
                        "GET",
                        self._env_path("/task_tools"),
                        expect_json=True,
                        sid=self.sid,
                        token=self.api_key,
                    )
                )
                return convert_tool_response(res, format=format)
            except aiohttp.ClientResponseError as e:
                if e.status == 404:
                    self._has_task_tools = False
                else:
                    raise
        res = await self._run_or_die(
            request_retryable(
                self.client,
                "GET",
                self._env_path("/tools"),
                expect_json=True,
                sid=self.sid,
                token=self.api_key,
            )
        )
        return convert_tool_response(res, format=format)

    async def call_tool(self, tool_name: str, input: JSONObject = {}) -> ToolOutput:
        if not isinstance(input, Mapping):
            raise ToolCallError(f"Tool input must be a dictionary, got {type(input).__name__}")

        if not all(isinstance(k, str) for k in input.keys()):
            non_string_keys = [k for k in input.keys() if not isinstance(k, str)]
            raise ToolCallError(f"All keys in tool input must be strings. Found non-string keys: {non_string_keys}")

        res = await self._run_or_die(
            resumable_sse(
                self.client,
                self._env_path("/call"),
                sid=self.sid,
                token=self.api_key,
                json={"name": tool_name, "input": input},
                max_retries=5,
            )
        )

        if res["ok"]:
            blocks: list[Union[TextBlock, ImageBlock]] = []
            for block in res["output"]["blocks"]:
                if block["type"] == "text":
                    blocks.append(TextBlock(
                        text=block["text"],
                        detail=block.get("detail")
                    ))
                elif block["type"] == "image":
                    blocks.append(ImageBlock(
                        mimeType=block["mimeType"],
                        detail=block.get("detail"),
                        data=block["data"]
                    ))
            return ToolOutput(
                blocks=blocks,
                metadata=res["output"]["metadata"],
                reward=res["output"]["reward"],
                finished=res["output"]["finished"]
            )
        else:
            raise ToolCallError(res["error"])


class Session:
    """Synchronous wrapper around AsyncSession."""

    def __init__(self, async_session: AsyncSession, loop: asyncio.AbstractEventLoop):
        self._async = async_session
        self._loop = loop

    @property
    def sid(self) -> Optional[str]:
        return self._async.sid

    @property
    def task(self) -> Optional[Task]:
        return self._async.task

    def __enter__(self) -> "Session":
        self._loop.run_until_complete(self._async.__aenter__())
        return self

    def __exit__(self, *exc):
        self._loop.run_until_complete(self._async.__aexit__(*exc))

    def get_prompt(self) -> list[Union[TextBlock, ImageBlock]]:
        return self._loop.run_until_complete(self._async.get_prompt())

    @overload
    def list_tools(self, format: None = None) -> list[ToolSpec]: ...

    @overload
    def list_tools(self, format: Provider) -> list[dict]: ...

    def list_tools(self, format: Optional[Provider] = None) -> Union[list[ToolSpec], list[dict]]:
        return self._loop.run_until_complete(self._async.list_tools(format))

    def call_tool(self, tool_name: str, input: JSONObject = {}) -> ToolOutput:
        return self._loop.run_until_complete(self._async.call_tool(tool_name, input))


# Avoid circular import: AsyncEnvironment is imported at type-check time only
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ors.client.environment import AsyncEnvironment
