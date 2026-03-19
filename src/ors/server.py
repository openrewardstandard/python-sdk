import asyncio
import base64
import inspect
import json
import time
import uuid
from contextlib import asynccontextmanager, suppress
from typing import Literal, Optional, Union

import structlog
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.routing import APIRoute
from sse_starlette import EventSourceResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from ors._version import __version__
from ors._log import get_logger, setup_logging, ORS_USE_STRUCTURED_LOGS
from .environment import Environment
from ._reconnect import sse_task_stream
from .types import Blocks, CreateSession, GetTask, GetTaskRange, ListTasks, ListToolsOutput, NumTasks, Split, ToolCall
from ._utils import maybe_await

logger = get_logger("ors.server")


def _get_env_map(environments: list[type[Environment]]) -> dict[str, type[Environment]]:
    env_map = {}
    for env in environments:
        cls = env if inspect.isclass(env) else type(env)
        if not (isinstance(cls, type) and issubclass(cls, Environment)):
            raise TypeError(f"{cls!r} is not Environment")
        key: str = cls.name().lower()
        if key in env_map:
            raise ValueError(f"duplicate env {key}")
        env_map[key] = cls
    return env_map

def _get_env_cls(env_map: dict[str, type[Environment]], env_name: str) -> type[Environment]:
    env_cls = env_map.get(env_name.lower())
    if env_cls is None:
        raise HTTPException(404, f"unknown environment {env_name!r}")
    return env_cls


def _convert_to_split(split: Union[str, Split]) -> Split:
    if isinstance(split, Split):
        return split
    if split in ["train", "validation", "test"]:
        return Split(name=split, type=split)  # type: ignore[arg-type]
    else:
        return Split(name=split, type="validation")


def _parse_secrets(create_session: CreateSession) -> dict[str, str]:
    """Extract secrets from the CreateSession model."""
    if create_session.secrets:
        return dict(create_session.secrets)
    return {}


async def extract_sid(request: Request) -> str:
    x_session_id = request.headers.get("X-Session-ID")
    if not x_session_id:
        raise HTTPException(400, "X-Session-ID header is required")
    return x_session_id.strip()


class _LoggingRoute(APIRoute):
    def get_route_handler(self):
        original = super().get_route_handler()
        async def handler(request: Request):
            start = time.monotonic()
            try:
                response = await original(request)
                logger.info(
                    "request_handled",
                    httpRequest={"latency": f"{time.monotonic() - start:.6f}s", "status": response.status_code},
                )
                return response
            except HTTPException as exc:
                logger.warning(
                    "request_rejected",
                    detail=exc.detail,
                    httpRequest={"latency": f"{time.monotonic() - start:.6f}s", "status": exc.status_code},
                )
                raise
            except Exception as exc:
                logger.exception(
                    "request_error",
                    error_type=type(exc).__name__,
                    error=str(exc),
                    httpRequest={"latency": f"{time.monotonic() - start:.6f}s"},
                )
                raise
        return handler


class RequestContextMiddleware:
    """Binds request context (session_id, method, path) to structlog contextvars."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] == "http":
            structlog.contextvars.clear_contextvars()
            headers = dict(scope.get("headers", []))
            sid = headers.get(b"x-session-id", b"").decode()
            structlog.contextvars.bind_contextvars(
                session_id=sid,
                method=scope.get("method", ""),
                path=scope.get("path", ""),
            )
        await self.app(scope, receive, send)


class ErrorHandlingMiddleware:
    """Catches unhandled exceptions and returns error details based on configured verbosity."""

    def __init__(self, app: ASGIApp, return_errors: Literal["none", "exception", "stacktrace"] = "stacktrace"):
        self.app = app
        self.return_errors = return_errors

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        try:
            await self.app(scope, receive, send)
        except HTTPException:
            raise
        except Exception as exc:
            import traceback
            logger.exception("unhandled_exception", error_type=type(exc).__name__, error=str(exc))

            if self.return_errors == "none":
                detail = "Internal Server Error"
            elif self.return_errors == "exception":
                detail = f"{type(exc).__name__}: {exc}"
            else:
                detail = traceback.format_exc()

            response = JSONResponse(status_code=500, content={"detail": detail})
            await response(scope, receive, send)


async def _delete_session(
    sid: str,
    active_envs: dict[str, Optional[Environment]],
    last_ping: dict[str, float],
    setup_tasks: dict[str, asyncio.Task],
    ready: dict[str, asyncio.Event],
    setup_errors: dict[str, Exception],
):
    last_ping.pop(sid, None)

    task = setup_tasks.pop(sid, None)
    if task and not task.done():
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    evt = ready.get(sid)
    if evt and not evt.is_set():
        setup_errors[sid] = HTTPException(410, "Session deleted")
        evt.set()
    ready.pop(sid, None)

    env = active_envs.pop(sid, None)
    if env:
        try:
            await maybe_await(env.teardown())
        except Exception as exc:
            logger.exception("Environment teardown failed", session_id=sid, environment_name=env.name(), error_type=type(exc).__name__, error=str(exc))

    setup_errors.pop(sid, None)


class RedirectToDefaultEnvMiddleware:
    def __init__(self, app: ASGIApp, env_classes: dict, root_paths: set):
        self.app = app
        self.env_classes = env_classes
        self.root_paths = root_paths

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] == "http":
            path = scope.get("path", "")
            first_segment = path.strip("/").split("/", 1)[0].lower()

            if (
                first_segment not in self.root_paths
                and first_segment not in self.env_classes
                and self.env_classes
            ):
                first_env = next(iter(self.env_classes.keys()))
                query = scope.get("query_string", b"").decode()
                new_path = f"/{first_env}{path}"
                if query:
                    new_path += f"?{query}"
                response = RedirectResponse(url=new_path, status_code=308)
                await response(scope, receive, send)
                return

        await self.app(scope, receive, send)


class Server:
    def __init__(
        self,
        environments: list[type[Environment]],
        return_errors: Literal["none", "exception", "stacktrace"] = "exception",
    ) -> None:
        """ORS environment hosting server.

        The first environment in the provided list is the default and is served at /.
        ``return_errors`` controls how unhandled exceptions are surfaced in 500 responses:
        "none" returns an opaque message, "exception" includes the exception string,
        and "stacktrace" includes the full traceback.
        """
        if not environments:
            raise ValueError("Server requires at least one environment to be provided.")

        self._env_classes: dict[str, type[Environment]] = _get_env_map(environments)

        for env_name, env_cls in self._env_classes.items():
            tools = env_cls.list_tools().tools
            if not tools:
                raise ValueError(f"Environment '{env_name}' has no tools defined. Add at least one @tool method.")

            splits = env_cls.list_splits()
            if not splits:
                raise ValueError(f"Environment '{env_name}' has no splits defined. Implement list_splits() to return at least one split.")

        self._active_envs: dict[str, Optional[Environment]] = {}
        self._last_ping: dict[str, float] = {}

        self._setup_tasks: dict[str, asyncio.Task] = {}
        self._ready: dict[str, asyncio.Event] = {}
        self._setup_errors: dict[str, Exception] = {}

        self._reaper_task: Optional[asyncio.Task] = None

        async def await_environment_ready(sid: str) -> Environment:
            evt = self._ready.get(sid)
            if evt is None:
                raise HTTPException(404, "Active environment not found")

            if evt.is_set():
                err = self._setup_errors.get(sid)
                if err:
                    raise err
                env = self._active_envs.get(sid)
                if env is None:
                    raise HTTPException(410, "Session deleted")
                return env

            await evt.wait()
            err = self._setup_errors.get(sid, None)
            if err:
                raise err
            env = self._active_envs.get(sid)
            if env is None:
                raise HTTPException(410, "Session deleted")
            return env

        async def reaper_coro(reaper_interval: float=5, stale_threshold: float=900):
            while True:
                await asyncio.sleep(reaper_interval)
                now = time.monotonic()
                stale = [sid for sid, ts in list(self._last_ping.items()) if now - ts > stale_threshold]
                for sid in stale:
                    await _delete_session(
                        sid,
                        self._active_envs,
                        self._last_ping,
                        self._setup_tasks,
                        self._ready,
                        self._setup_errors,
                    )

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            self._reaper_task = asyncio.create_task(reaper_coro())
            yield
            if self._reaper_task:
                self._reaper_task.cancel()
                try:
                    await self._reaper_task
                except asyncio.CancelledError:
                    pass
            await asyncio.gather(*[
                _delete_session(
                    sid,
                    self._active_envs,
                    self._last_ping,
                    self._setup_tasks,
                    self._ready,
                    self._setup_errors,
                )
                for sid in list(self._setup_tasks.keys())
            ])

        app = FastAPI(lifespan=lifespan)
        app.router.route_class = _LoggingRoute

        @app.get("/health")
        async def health():
            return {"status": "ok"}

        @app.post("/ping")
        async def ping(sid: str = Depends(extract_sid)):
            if sid not in self._active_envs:
                raise HTTPException(404, "Active environment not found")
            self._last_ping[sid] = time.monotonic()
            return {"status": "ok"}

        @app.get("/list_environments")
        async def get_envs():
            return list(self._env_classes.keys())

        @app.get("/{env_name}/tools")
        async def list_tools(env_name: str):
            env_cls = _get_env_cls(self._env_classes, env_name)
            return env_cls.list_tools().model_dump()

        @app.get("/{env_name}/splits")
        async def list_splits(env_name: str):
            env_cls = _get_env_cls(self._env_classes, env_name)
            return [_convert_to_split(split).model_dump() for split in env_cls.list_splits()]

        @app.post("/{env_name}/tasks")
        async def list_tasks(env_name: str, list_tasks: ListTasks):
            env_cls = _get_env_cls(self._env_classes, env_name)
            split_names = [split.name if isinstance(split, Split) else split for split in env_cls.list_splits()]
            if list_tasks.split not in split_names:
                raise HTTPException(status_code=400, detail="Invalid split")
            try:
                tasks = await maybe_await(env_cls.list_tasks(list_tasks.split))
            except NotImplementedError as e:
                raise HTTPException(status_code=400, detail="list_tasks is not supported for this environment. Use the index-based API (num_tasks + get_task) instead.") from e
            return {"tasks": tasks, "env_name": env_name}

        @app.post("/{env_name}/num_tasks")
        async def num_tasks(env_name: str, num_tasks: NumTasks):
            env_cls = _get_env_cls(self._env_classes, env_name)
            split_names = [split.name if isinstance(split, Split) else split for split in env_cls.list_splits()]
            if num_tasks.split not in split_names:
                raise HTTPException(status_code=400, detail="Invalid split")
            return {"num_tasks": await env_cls.num_tasks(num_tasks.split)}

        @app.post("/{env_name}/task")
        async def get_task(env_name: str, get_task: GetTask):
            env_cls = _get_env_cls(self._env_classes, env_name)
            split_names = [split.name if isinstance(split, Split) else split for split in env_cls.list_splits()]
            if get_task.split not in split_names:
                raise HTTPException(status_code=400, detail="Invalid split")
            try:
                task = await env_cls.get_task(get_task.split, get_task.index)
            except IndexError as e:
                raise HTTPException(status_code=400, detail="Invalid index") from e
            return {"task": task}

        @app.post("/{env_name}/task_range")
        async def get_task_range(env_name: str, get_task_range: GetTaskRange):
            env_cls = _get_env_cls(self._env_classes, env_name)
            split_names = [split.name if isinstance(split, Split) else split for split in env_cls.list_splits()]
            if get_task_range.split not in split_names:
                raise HTTPException(status_code=400, detail="Invalid split")
            try:
                tasks = await env_cls.get_task_range(get_task_range.split, get_task_range.start, get_task_range.stop)
            except IndexError as e:
                raise HTTPException(status_code=400, detail="Invalid index range") from e
            return {"tasks": tasks}

        @app.post("/create")
        async def create_environment(request: Request, create_session: CreateSession, sid: str = Depends(extract_sid)):
            if sid in self._active_envs:
                raise HTTPException(status_code=400, detail="Session already exists")

            env_name = create_session.env_name
            if env_name is None:
                env_name = next(iter(self._env_classes.keys()))
            env_cls = _get_env_cls(self._env_classes, env_name)

            if create_session.task_spec is not None:
                task_spec = create_session.task_spec
            else:
                split_names = [_convert_to_split(s).name for s in env_cls.list_splits()]
                assert create_session.split is not None
                assert create_session.index is not None
                if create_session.split not in split_names:
                    raise HTTPException(status_code=400, detail="Invalid split")
                try:
                    task_spec = await env_cls.get_task(create_session.split, create_session.index)
                except IndexError as e:
                    raise HTTPException(status_code=400, detail="Invalid index") from e

            self._active_envs[sid] = None
            self._last_ping[sid] = time.monotonic()
            self._ready[sid] = asyncio.Event()

            secrets = _parse_secrets(create_session)

            async def perform_setup():
                setup_start = time.monotonic()
                try:
                    env = env_cls(task_spec=task_spec, secrets=secrets)
                    await maybe_await(env.setup())
                    self._active_envs[sid] = env
                    duration_ms = (time.monotonic() - setup_start) * 1000
                    logger.info("setup_completed", session_id=sid, env_name=create_session.env_name, duration_ms=duration_ms)
                except Exception as e:
                    duration_ms = (time.monotonic() - setup_start) * 1000
                    logger.exception("setup_failed", session_id=sid, env_name=create_session.env_name, duration_ms=duration_ms, error_type=type(e).__name__, error=str(e))
                    self._setup_errors[sid] = e
                finally:
                    evt = self._ready.get(sid)
                    if evt:
                        evt.set()

            if sid not in self._setup_tasks:
                self._setup_tasks[sid] = asyncio.create_task(perform_setup())

            logger.info("session_created", env_name=create_session.env_name)
            return {"sid": sid}

        async def require_existing_session(
            sid: str = Depends(extract_sid),
        ):
            self._last_ping[sid] = time.monotonic()
            return await await_environment_ready(sid)

        @app.post("/delete")
        async def delete_environment(sid: str = Depends(extract_sid)):
            if sid in self._setup_tasks:
                await _delete_session(
                    sid,
                    self._active_envs,
                    self._last_ping,
                    self._setup_tasks,
                    self._ready,
                    self._setup_errors,
                )
                logger.info("session_deleted")
            return {"sid": sid}

        @app.post("/create_session")
        async def create_session():
            sid = str(uuid.uuid4())
            async def _stream():
                yield {"event": "task_id", "data": sid}
                yield {"event": "end", "data": ""}
            return EventSourceResponse(
                _stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        @app.post("/delete_session")
        async def delete_session(sid: str = Depends(extract_sid)):
            return {"sid": sid}

        @app.get("/{env_name}/task_tools")
        async def task_tools(env_name: str, env: Environment = Depends(require_existing_session)):
            shared = env.list_tools()
            task_specific = env.list_task_tools()
            combined = ListToolsOutput(tools=shared.tools + task_specific.tools)
            return combined.model_dump()

        @app.post("/{env_name}/call")
        async def call_tool(request: Request, env_name: str, tool_call: ToolCall, env: Environment = Depends(require_existing_session)):
            logger.info("tool_call_started", tool=tool_call.name, env_name=env_name)

            async def call_tool_coro():
                res = await env._call_tool(tool_call.name, tool_call.input)
                return res.model_dump_json(indent=None)

            return EventSourceResponse(
                sse_task_stream(
                    lambda: call_tool_coro(),
                    request,
                    task_id=tool_call.task_id,
                ),
                ping=10,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                }
            )

        @app.get("/{env_name}/prompt")
        async def get_prompt(request: Request, env: Environment = Depends(require_existing_session)) -> Blocks:
            return (await maybe_await(env.get_prompt()))

        # After all routes are defined, before adding the middleware:
        root_paths = set()
        for route in app.routes:
            path = getattr(route, 'path', '')
            segments = path.strip('/').split('/', 1)
            first_segment = segments[0] if segments else ''
            if not first_segment.startswith('{'):
                root_paths.add(first_segment.lower())

        self.app = app
        self.app.add_middleware(
            RedirectToDefaultEnvMiddleware,
            env_classes=self._env_classes,
            root_paths=root_paths
        )
        self.app.add_middleware(RequestContextMiddleware)
        self.app.add_middleware(ErrorHandlingMiddleware, return_errors=return_errors)

    def run(self, host: str = "0.0.0.0", port: int = 8080) -> None:
        """Start the server."""
        setup_logging()
        logger.info("server_starting", version=__version__, host=host, port=port)
        if ORS_USE_STRUCTURED_LOGS:
            uvicorn.run(self.app, host=host, port=port, timeout_keep_alive=60,
                        log_config=None, access_log=False)
        else:
            uvicorn.run(self.app, host=host, port=port, timeout_keep_alive=60,
                        access_log=False)
