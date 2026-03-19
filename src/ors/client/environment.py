import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Optional, Union, overload

import aiohttp
from ors.client._http import (
    _raise_for_status,
    request_retryable,
    resumable_sse,
    _finalize_session,
)
from ors.client._types import (
    ImageBlock,
    JSONObject,
    Mapping,
    Provider,
    Task,
    TextBlock,
    ToolSpec,
)


# ── Schema sanitizers for provider-specific tool format conversion ──

GOOGLE_UNSUPPORTED_SCHEMA_KEYS = {
    "additionalProperties",
    "additional_properties",
    "title",
    "default",
    "examples",
    "example",
    "patternProperties",
    "oneOf",
    "allOf",
    "anyOf",
    "not",
}

OPENAI_UNSUPPORTED_SCHEMA_KEYS = {
    "additionalProperties",
    "patternProperties",
    "oneOf",
    "allOf",
    "anyOf",
    "not",
}


def _sanitize_google_schema(x: Any) -> Any:
    """Recursively remove schema keys that Gemini/Google function calling rejects."""
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            if k in GOOGLE_UNSUPPORTED_SCHEMA_KEYS:
                continue
            if k == "$ref":
                k = "ref"
            elif k == "$defs":
                k = "defs"
            out[k] = _sanitize_google_schema(v)
        return out
    if isinstance(x, list):
        return [_sanitize_google_schema(i) for i in x]
    return x


def _sanitize_openai_schema(x: Any) -> Any:
    """Recursively sanitize schema for OpenAI function calling."""
    if isinstance(x, dict):
        if "anyOf" in x:
            options = x["anyOf"]
            for option in options:
                if not (isinstance(option, dict) and option.get("type") == "null"):
                    return _sanitize_openai_schema(option)
            if options:
                return _sanitize_openai_schema(options[0])

        if "oneOf" in x:
            options = x["oneOf"]
            if options:
                return _sanitize_openai_schema(options[0])

        if "allOf" in x:
            options = x["allOf"]
            if options:
                return _sanitize_openai_schema(options[0])

        out = {}
        for k, v in x.items():
            if k in OPENAI_UNSUPPORTED_SCHEMA_KEYS:
                continue
            out[k] = _sanitize_openai_schema(v)

        if out.get("type") == "array" and "items" not in out:
            out["items"] = {}

        return out

    if isinstance(x, list):
        return [_sanitize_openai_schema(i) for i in x]

    return x


def _strip_titles(value: Any) -> Any:
    """Recursively remove JSON schema ``title`` keys."""
    if isinstance(value, dict):
        return {
            k: _strip_titles(v)
            for k, v in value.items()
            if k != "title"
        }
    if isinstance(value, list):
        return [_strip_titles(item) for item in value]
    return value


@overload
def convert_tool_response(res: Mapping[str, Any], format: None = None) -> list[ToolSpec]: ...

@overload
def convert_tool_response(res: Mapping[str, Any], format: Provider = ...) -> list[dict[str, Any]]: ...

def convert_tool_response(
    res: Mapping[str, Any],
    format: Optional[Provider] = None,
) -> Union[list[ToolSpec], list[dict[str, Any]]]:
    if format is not None:
        if format == "openai":
            return [
                {
                    "type": "function",
                    **{
                        k: _strip_titles(v)
                        for k, v in tool.items()
                        if k not in {"input_schema", "title"}
                    },
                    "parameters": (
                        _sanitize_openai_schema(_strip_titles(tool["input_schema"]))
                        if tool.get("input_schema")
                        else None
                    ),
                }
                for tool in res["tools"]
            ]
        elif format == "openrouter":
            return [
                {
                    "type": "function",
                    "function": {
                        k: _strip_titles(v)
                        for k, v in tool.items()
                        if k not in {"input_schema", "title"}
                    },
                    "parameters": (
                        _sanitize_openai_schema(_strip_titles(tool["input_schema"]))
                        if tool.get("input_schema")
                        else None
                    ),
                }
                for tool in res["tools"]
            ]
        elif format == "anthropic":
            return [
                {
                    "type": "custom",
                    **{
                        k: _strip_titles(v)
                        for k, v in tool.items()
                        if k not in {"input_schema", "title"}
                    },
                    "input_schema": _strip_titles(tool["input_schema"]) if tool.get("input_schema") else {"type": "object", "properties": {}}
                }
                for tool in res["tools"]
            ]
        elif format == "google":
            return [
                {
                    **{
                        k: _strip_titles(v)
                        for k, v in tool.items()
                        if k not in {"input_schema", "title"}
                    },
                    "parameters": (
                        _sanitize_google_schema(_strip_titles(tool["input_schema"]))
                        if tool.get("input_schema")
                        else None
                    ),
                }
                for tool in res["tools"]
            ]
        else:
            raise ValueError(f"Invalid format: {format!r}")

    return [ToolSpec(**tool) for tool in res["tools"]]


# ── SID provider for stateless environment queries ──

@asynccontextmanager
async def _sid_provider(client: aiohttp.ClientSession, token: Optional[str]) -> AsyncGenerator[str, None]:
    """Ephemeral SID provider using SSE-based /create_session, cleanup via /delete_session."""
    sid: Optional[str] = None

    def on_event(event: str, data: str) -> None:
        nonlocal sid
        if event == "task_id":
            sid = data.strip()

    await resumable_sse(
        client,
        "/create_session",
        token=token,
        max_retries=3,
        on_event=on_event,
    )

    assert sid is not None, "No SID returned from /create_session"
    try:
        yield sid
    finally:
        try:
            await request_retryable(client, "POST", "/delete_session", sid=sid, expect_json=False, token=token)
        except Exception:
            pass


# ── Async Environment ──

class AsyncEnvironment:

    def __init__(
        self,
        name: str,
        client: aiohttp.ClientSession,
        api_key: Optional[str],
    ) -> None:
        self.name = name
        self.client = client
        self.api_key = api_key

    async def list_splits(self) -> list[str]:
        async with _sid_provider(self.client, self.api_key) as sid:
            path = f"/{self.name}/splits"
            res = await request_retryable(self.client, "GET", path, expect_json=True, sid=sid, token=self.api_key)
            return [s["name"] for s in res]

    async def list_tasks(self, split: str) -> list[Task]:
        async with _sid_provider(self.client, self.api_key) as sid:
            path = f"/{self.name}/tasks"
            res = await request_retryable(self.client, "POST", path, expect_json=True, sid=sid, json={"split": split}, token=self.api_key)
            return [Task(environment_name=res.get("env_name", self.name), task_spec=task) for task in res["tasks"]]

    async def num_tasks(self, split: str) -> int:
        """Get the number of tasks for a given split."""
        async with _sid_provider(self.client, self.api_key) as sid:
            path = f"/{self.name}/num_tasks"
            res = await request_retryable(self.client, "POST", path, expect_json=True, sid=sid, json={"split": split}, token=self.api_key)
            return res["num_tasks"]

    async def get_task(self, split: str, index: int) -> Task:
        """Get a single task by split and index."""
        async with _sid_provider(self.client, self.api_key) as sid:
            path = f"/{self.name}/task"
            res = await request_retryable(self.client, "POST", path, expect_json=True, sid=sid, json={"split": split, "index": index}, token=self.api_key)
            return Task(environment_name=self.name, task_spec=res["task"])

    async def get_task_range(self, split: str, start: Optional[int] = None, stop: Optional[int] = None) -> list[Task]:
        """Get tasks for indices in range(start, stop). Supports negative and None indices."""
        async with _sid_provider(self.client, self.api_key) as sid:
            path = f"/{self.name}/task_range"
            payload: dict[str, Any] = {"split": split}
            if start is not None:
                payload["start"] = start
            if stop is not None:
                payload["stop"] = stop
            res = await request_retryable(self.client, "POST", path, expect_json=True, sid=sid, json=payload, token=self.api_key)
            return [Task(environment_name=self.name, task_spec=task) for task in res["tasks"]]

    async def list_tools(self, format: Optional[Provider] = None) -> Union[list[ToolSpec], list[dict]]:
        path = f"/{self.name}/tools"
        async with _sid_provider(self.client, self.api_key) as sid:
            res = await request_retryable(self.client, "GET", path, expect_json=True, sid=sid, token=self.api_key)
            return convert_tool_response(res, format=format)

    def session(
        self,
        task: Optional[Task] = None,
        secrets: Optional[dict[str, str]] = None,
        *,
        split: Optional[str] = None,
        index: Optional[int] = None,
    ) -> "AsyncSession":
        """Create a session from a Task object or from split/index."""
        from ors.client.session import AsyncSession
        return AsyncSession(self, task=task, secrets=secrets, api_key=self.api_key, split=split, index=index)


# ── Async Environments API ──

class AsyncEnvironmentsAPI:

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = aiohttp.ClientTimeout(total=None)

        self._connector: Optional[aiohttp.TCPConnector] = None
        self._clients: dict[str, aiohttp.ClientSession] = {}

    def _get_connector(self) -> aiohttp.TCPConnector:
        """Lazily create the connector when inside a running event loop."""
        if self._connector is None or self._connector.closed:
            self._connector = aiohttp.TCPConnector(limit=1_000_000)
        return self._connector

    def get(self, name: str, base_url: Optional[str] = None) -> AsyncEnvironment:
        if base_url is None:
            base_url = self.base_url

        if base_url not in self._clients:
            self._clients[base_url] = aiohttp.ClientSession(
                base_url=base_url,
                timeout=self.timeout,
                connector=self._get_connector(),
                trust_env=True,
            )
        client = self._clients[base_url]

        return AsyncEnvironment(
            name=name,
            client=client,
            api_key=self.api_key,
        )

    def __del__(self):
        for client in self._clients.values():
            _finalize_session(client)


# ── Sync wrappers ──

class Session:
    """Synchronous wrapper around AsyncSession (from session.py)."""

    def __init__(self, async_session: "AsyncSession", loop: asyncio.AbstractEventLoop):
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

    def list_tools(self, format: Optional[Provider] = None) -> Union[list[ToolSpec], list[dict]]:
        return self._loop.run_until_complete(self._async.list_tools(format))

    def call_tool(self, tool_name: str, input: JSONObject = {}) -> "ToolOutput":
        from ors.client._types import ToolOutput
        return self._loop.run_until_complete(self._async.call_tool(tool_name, input))


class Environment:
    """Synchronous wrapper around AsyncEnvironment."""

    def __init__(self, async_env: AsyncEnvironment, loop: asyncio.AbstractEventLoop):
        self._async = async_env
        self._loop = loop

    @property
    def name(self) -> str:
        return self._async.name

    def list_splits(self) -> list[str]:
        return self._loop.run_until_complete(self._async.list_splits())

    def list_tasks(self, split: str) -> list[Task]:
        return self._loop.run_until_complete(self._async.list_tasks(split))

    def num_tasks(self, split: str) -> int:
        return self._loop.run_until_complete(self._async.num_tasks(split))

    def get_task(self, split: str, index: int) -> Task:
        return self._loop.run_until_complete(self._async.get_task(split, index))

    def get_task_range(self, split: str, start: Optional[int] = None, stop: Optional[int] = None) -> list[Task]:
        return self._loop.run_until_complete(self._async.get_task_range(split, start, stop))

    def list_tools(self, format: Optional[Provider] = None) -> Union[list[ToolSpec], list[dict]]:
        return self._loop.run_until_complete(self._async.list_tools(format))

    def session(
        self,
        task: Optional[Task] = None,
        secrets: Optional[dict[str, str]] = None,
        *,
        split: Optional[str] = None,
        index: Optional[int] = None,
    ) -> Session:
        """Create a session from a Task object or from split/index."""
        from ors.client.session import AsyncSession
        async_session = AsyncSession(self._async, task=task, secrets=secrets, api_key=self._async.api_key, split=split, index=index)
        return Session(async_session, self._loop)


class EnvironmentsAPI:
    """Synchronous wrapper around AsyncEnvironmentsAPI."""

    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self._loop = asyncio.new_event_loop()
        self._async = AsyncEnvironmentsAPI(base_url, api_key)

    def get(self, name: str, base_url: Optional[str] = None) -> Environment:
        async def _get():
            return self._async.get(name, base_url)
        async_env = self._loop.run_until_complete(_get())
        return Environment(async_env, self._loop)

    def close(self):
        """Clean up resources."""
        async def _close_all():
            for client in self._async._clients.values():
                if not client.closed:
                    await client.close()
        self._loop.run_until_complete(_close_all())
        self._loop.run_until_complete(self._loop.shutdown_asyncgens())
        self._loop.close()

    def __enter__(self) -> "EnvironmentsAPI":
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        if not self._loop.is_closed():
            self.close()


# For TYPE_CHECKING
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ors.client.session import AsyncSession
    from ors.client._types import ToolOutput
