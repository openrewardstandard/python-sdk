import asyncio
from typing import Any, Coroutine, Optional

import aiohttp
from ors.client._http import (
    HeartbeatTimeoutError,
    MaxRetriesError,
    _finalize_session,
    request_retryable,
    resumable_sse,
)
from ors.client._ping import ErrorResponse, ping


class SessionTerminatedError(RuntimeError):
    def __init__(self, reason: str, *, sid: Optional[str]):
        super().__init__(f"Session terminated (sid={sid!r}): {reason}")
        self.reason = reason
        self.sid = sid


class BaseAsyncSession:

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str],
        creation_endpoint: str,
        creation_payload: dict[str, Any],
        creation_timeout: int = 60 * 30,
        client: Optional[aiohttp.ClientSession] = None,
        default_headers: Optional[dict[str, str]] = None,
        creation_headers: Optional[dict[str, str]] = None,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self._creation_endpoint = creation_endpoint
        self._creation_payload = creation_payload
        self.creation_timeout = creation_timeout
        self._default_headers = default_headers
        self._creation_headers = creation_headers

        self._external_client = client
        self._own_connector: Optional[aiohttp.TCPConnector] = None
        self._own_client: Optional[aiohttp.ClientSession] = None

        self.sid: Optional[str] = None
        self._pending_task_id: Optional[str] = None

        self._ping_task: asyncio.Task[None] | None = None
        self._dead = asyncio.Event()
        self._dead_exception: Optional[SessionTerminatedError] = None

    @property
    def client(self) -> aiohttp.ClientSession:
        c = self._external_client or self._own_client
        if c is None:
            raise RuntimeError("Session not started. Use as async context manager.")
        return c

    def _mark_dead(self, exc: ErrorResponse):
        if self._dead_exception is None:
            self._dead_exception = SessionTerminatedError(exc.message, sid=self.sid)
            self._dead.set()

    def _ensure_alive(self):
        if self._dead_exception is not None:
            raise self._dead_exception

    async def _run_or_die(self, coro: Coroutine[Any, Any, Any]) -> Any:
        """Run a coroutine until completion or until the session dies."""
        if self._dead_exception is not None:
            raise self._dead_exception

        task = asyncio.create_task(coro)
        stopper = asyncio.create_task(self._dead.wait())
        try:
            done, pending = await asyncio.wait(
                {task, stopper},
                return_when=asyncio.FIRST_COMPLETED,
            )

            if self._dead.is_set() and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                if self._dead_exception:
                    raise self._dead_exception

            return await task

        except SessionTerminatedError:
            raise
        except Exception as e:
            raise self._annotate_error(e) from e.__cause__

        finally:
            stopper.cancel()
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    def _annotate_error(self, e: Exception) -> Exception:
        """Return a copy of *e* whose message is prefixed with ``(sid=...)``."""
        prefix = f"(sid={self.sid})"
        if isinstance(e, MaxRetriesError):
            return MaxRetriesError(f"{prefix} {e}", errors=e.errors).with_traceback(e.__traceback__)
        try:
            new = type(e)(f"{prefix} {e}")
        except Exception:
            new = RuntimeError(f"{prefix} {e}")
        new.__traceback__ = e.__traceback__
        new.__cause__ = e.__cause__
        return new

    async def _post_create(self) -> None:
        """Subclass hook called after SID is obtained. No-op in base."""
        pass

    async def _pre_delete(self) -> None:
        """Subclass hook called before session deletion. No-op in base."""
        pass

    async def _run_ping(self, url: str, sid: str, api_key: Optional[str], sleep_time: float) -> None:
        try:
            await ping(url, sid, api_key, sleep_time, self.client)
        except Exception as e:
            self._mark_dead(ErrorResponse(type="error", message=str(e)))

    async def __aenter__(self) -> "BaseAsyncSession":
        if self._external_client is None:
            self._own_connector = aiohttp.TCPConnector(limit=1_000_000)
            self._own_client = aiohttp.ClientSession(
                base_url=self.base_url,
                connector=self._own_connector,
                headers=self._default_headers,
                trust_env=True,
            )

        def on_event(event: str, data: str) -> None:
            if event == "task_id":
                self._pending_task_id = data.strip()

        try:
            res = await resumable_sse(
                self.client,
                self._creation_endpoint,
                token=self.api_key,
                json=self._creation_payload,
                max_retries=3,
                timeout=self.creation_timeout,
                on_event=on_event,
                extra_headers=self._creation_headers,
            )
        except Exception as e:
            raise self._annotate_error(e) from e.__cause__

        if res and isinstance(res, dict):
            self.sid = res.get("sid")
        if self.sid is None:
            self.sid = self._pending_task_id
        self._pending_task_id = None

        assert self.sid is not None, "No SID returned from creation endpoint"

        try:
            await self._post_create()
        except Exception as e:
            raise self._annotate_error(e) from e.__cause__

        self._ping_task = asyncio.create_task(self._run_ping(
            url="/ping",
            sid=self.sid,
            api_key=self.api_key,
            sleep_time=10,
        ))

        return self

    async def __aexit__(self, *exc):
        if self._ping_task is not None:
            self._ping_task.cancel()
            try:
                await self._ping_task
            except asyncio.CancelledError:
                pass
            self._ping_task = None

        try:
            await self._pre_delete()
        except Exception:
            pass

        if self._pending_task_id is not None:
            try:
                await request_retryable(
                    self.client, "POST", "/delete_session",
                    expect_json=False, token=self.api_key,
                    sid=self._pending_task_id,
                )
            except Exception:
                pass
            self._pending_task_id = None

        if self.sid is not None:
            try:
                await request_retryable(
                    self.client, "POST", "/delete_session",
                    expect_json=False, token=self.api_key,
                    sid=self.sid,
                )
            except Exception:
                pass

        if self._own_client is not None and not self._own_client.closed:
            await self._own_client.close()

    def __del__(self):
        own_client = getattr(self, "_own_client", None)
        if own_client is not None:
            _finalize_session(own_client)
