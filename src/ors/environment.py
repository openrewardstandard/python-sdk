import functools
import inspect
import time
from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable, ClassVar, Optional, Sequence, TypeVar, Union, get_type_hints, overload

from pydantic import BaseModel, ValidationError

from ors._log import get_logger
from .types import (Blocks, JSONObject, ListToolsOutput, RunToolError,
                    RunToolOutput, RunToolSuccess, Split, ToolOutput, ToolSpec)
from ._utils import maybe_await

T = TypeVar("T")

logger = get_logger("ors.environment")


@overload
def tool(fn: Callable[..., Any]) -> Callable[..., Any]: ...
@overload
def tool(*, shared: bool = True) -> Callable[[Callable[..., Any]], Callable[..., Any]]: ...
def tool(fn: Optional[Callable[..., Any]] = None, *, shared: bool = True) -> Callable[..., Any]:
    if fn is None:
        # Called with arguments: @tool(shared=False)
        def wrapper(f: Callable[..., Any]) -> Callable[..., Any]:
            setattr(f, "_env_tool", True)
            setattr(f, "_env_tool_shared", shared)
            return f
        return wrapper
    # Called without arguments: @tool
    setattr(fn, "_env_tool", True)
    setattr(fn, "_env_tool_shared", True)
    return fn


def _introspect_tool(fn: Callable[..., Any]) -> tuple[Any, dict, list]:
    """Return (unwrapped_fn, type_hints, non-self params) for a tool function."""
    real = inspect.unwrap(fn)
    hints = get_type_hints(real, include_extras=True)
    params = [p for p in inspect.signature(real).parameters.values() if p.name != "self"]
    return (real, hints, params)


class Environment(ABC):
    """
    An environment is an interface to computation that may be stateful. Clients interface with the
    environment through a persistent connection to perform actions. Environments have _tasks_, which
    are JSON objects describing a particular setup and goal state.
    """
    toolsets: ClassVar[Sequence[type]] = ()

    def __init__(self, task_spec: JSONObject = {}, secrets: dict[str, str] = {}) -> None:
        self.task_spec = task_spec
        self._toolset_instances: dict[type, Any] = {}

    def setup(self) -> Optional[Awaitable[None]]:
        """
        Setup the environment. This is called upon the first tool call by a connected client.
        """
        pass

    def teardown(self) -> Optional[Awaitable[None]]:
        """
        Teardown the environment. This is called upon client disconnect.
        """
        pass

    @abstractmethod
    def get_prompt(self) -> Union[Blocks, Awaitable[Blocks]]:
        """
        Get a default prompt for the current task.
        """

    @classmethod
    @abstractmethod
    def list_tasks(cls, split: str) -> Union[Sequence[JSONObject], Awaitable[Sequence[JSONObject]]]:
        """
        Get a list of tasks for the given split.
        """

    @classmethod
    @abstractmethod
    @functools.cache
    def list_splits(cls) -> Sequence[Union[Split, str]]:
        """
        Get a list of splits for the environment.
        """

    @classmethod
    async def num_tasks(cls, split: str) -> int:
        """Get the number of tasks for a given split."""
        if not hasattr(cls, '_num_tasks_cache'):
            cls._num_tasks_cache: dict[str, int] = {}
        if split not in cls._num_tasks_cache:
            tasks = await maybe_await(cls.list_tasks(split))
            cls._num_tasks_cache[split] = len(tasks)
        return cls._num_tasks_cache[split]

    @classmethod
    async def get_task(cls, split: str, index: int) -> JSONObject:
        """
        Get task `index` for the given `split`. Defaults to listing all tasks and taking an index.
        """
        tasks = await maybe_await(cls.list_tasks(split))
        return tasks[index]

    @classmethod
    async def get_task_range(cls, split: str, start: Optional[int] = None, stop: Optional[int] = None) -> list[JSONObject]:
        """
        Get tasks for indices in range(start, stop) for the given split.
        Follows Python range/slice conventions: start is inclusive, stop is exclusive.
        Supports negative indices and None defaults.
        """
        total = await cls.num_tasks(split)
        if start is None:
            start = 0
        if stop is None:
            stop = total
        if start < 0:
            start = max(total + start, 0)
        if stop < 0:
            stop = max(total + stop, 0)
        start = min(start, total)
        stop = min(stop, total)
        tasks = []
        for i in range(start, stop):
            tasks.append(await cls.get_task(split, i))
        return tasks

    @staticmethod
    def _is_tool(fn: Callable[..., Any]) -> bool:
        if not callable(fn) or not getattr(fn, "_env_tool", False):
            return False
        _, hints, params = _introspect_tool(fn)
        ret = hints.get("return")
        if len(params) == 0:
            return ret == ToolOutput
        if len(params) == 1:
            pt = hints.get(params[0].name)
            return (
                pt is not None and ret is not None and inspect.isclass(pt)
                and issubclass(pt, BaseModel) and ret == ToolOutput
            )
        return False

    @classmethod
    @functools.cache
    def list_tools(cls) -> ListToolsOutput:
        """
        List all tools available on this environment class. Result is cached per class since
        the tool set is static.
        """
        out: list[ToolSpec] = []
        env_tool_names: set[str] = set()

        for name in dir(cls):
            fn = getattr(cls, name)
            if not cls._is_tool(fn) or not getattr(fn, "_env_tool_shared", True):
                continue
            _, hints, params = _introspect_tool(fn)
            schema = None
            if params:
                mdl: type[BaseModel] = hints[params[0].name]
                schema = mdl.model_json_schema() if hasattr(mdl, "model_json_schema") else mdl.schema()
            out.append(ToolSpec(name=name, description=(fn.__doc__ or "").strip(), input_schema=schema))
            env_tool_names.add(name)

        if cls.toolsets:
            for toolset_cls in cls.toolsets:
                for name in dir(toolset_cls):
                    fn = getattr(toolset_cls, name)
                    if not cls._is_tool(fn) or not getattr(fn, "_env_tool_shared", True):
                        continue

                    if name in env_tool_names:
                        raise ValueError(
                            f"Tool name collision: '{name}' is defined in both the environment "
                            f"and toolset '{toolset_cls.__name__}'. Please rename one of them to avoid conflicts."
                        )

                    _, hints, params = _introspect_tool(fn)
                    schema = None
                    if params:
                        mdl: type[BaseModel] = hints[params[0].name]
                        schema = mdl.model_json_schema() if hasattr(mdl, "model_json_schema") else mdl.schema()
                    out.append(ToolSpec(name=name, description=(fn.__doc__ or "").strip(), input_schema=schema))

        return ListToolsOutput(tools=out)

    def list_task_tools(self) -> ListToolsOutput:
        """Override to return task-specific tools based on self.task_spec.
        Default returns an empty list."""
        return ListToolsOutput(tools=[])

    async def _call_tool(self, name: str, input: JSONObject) -> RunToolOutput:
        start = time.monotonic()

        env_fn = getattr(self, name, None)
        has_env_tool = env_fn is not None and self._is_tool(env_fn)

        toolset_fn = None
        toolset_source = None

        if self.__class__.toolsets:
            for toolset_cls in self.__class__.toolsets:
                if toolset_cls not in self._toolset_instances:
                    try:
                        self._toolset_instances[toolset_cls] = toolset_cls(self)
                    except TypeError:
                        try:
                            self._toolset_instances[toolset_cls] = toolset_cls()
                        except Exception as e:
                            logger.error("toolset_instantiation_failed", toolset=toolset_cls.__name__, error=str(e))
                            continue

                toolset_instance = self._toolset_instances[toolset_cls]
                candidate = getattr(toolset_instance, name, None)

                if candidate is not None and self._is_tool(candidate):
                    toolset_fn = candidate
                    toolset_source = toolset_cls.__name__
                    break

        if has_env_tool and toolset_fn is not None:
            logger.error("tool_name_collision", tool=name, toolset=toolset_source)
            return RunToolOutput(RunToolError(
                error=f"Tool name collision: '{name}' is defined in both the environment "
                      f"and toolset '{toolset_source}'. Please rename one of them to avoid conflicts."
            ))

        fn: Callable[..., Any]
        if has_env_tool:
            assert env_fn is not None
            fn = env_fn
        elif toolset_fn is not None:
            fn = toolset_fn
        else:
            logger.warning("tool_not_found", tool=name)
            return RunToolOutput(RunToolError(error=f"{name!r} is not a valid tool"))

        _, hints, params = _introspect_tool(fn)
        try:
            if not params:
                res = await maybe_await(fn())
            else:
                mdl: type[BaseModel] = hints[params[0].name]
                try:
                    inp = mdl(**input)
                except ValidationError as e:
                    logger.warning("tool_input_validation_error", tool=name, error=str(e.errors()))
                    return RunToolOutput(RunToolError(error=f"Tool input validation error: {str(e.errors())}"))
                res = await maybe_await(fn(inp))
        except Exception:
            duration_ms = (time.monotonic() - start) * 1000
            logger.exception("tool_call_failed", tool=name, duration_ms=duration_ms)
            raise

        if not isinstance(res, ToolOutput):
            raise TypeError(f"{name!r} returned {type(res).__name__}; expected ToolOutput")

        duration_ms = (time.monotonic() - start) * 1000
        logger.info("tool_call_completed", tool=name, duration_ms=duration_ms)
        return RunToolOutput(RunToolSuccess(output=res))

    @classmethod
    def name(cls) -> str:
        return cls.__name__
