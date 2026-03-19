import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable, Coroutine, Optional

from fastapi import Request

LINGER_SECONDS = 60
CHUNK_SIZE = 4096


@dataclass
class TaskInfo:
    task: asyncio.Task
    done: asyncio.Event = field(default_factory=asyncio.Event)
    result: Optional[Any] = None
    error: Optional[BaseException] = None
    completed_at: Optional[float] = None
    consumed: bool = False


tasks: dict[str, TaskInfo] = {}


async def _evict_after(task_id: str, seconds: float):
    await asyncio.sleep(seconds)
    ti = tasks.get(task_id)
    if ti and ti.done.is_set():
        tasks.pop(task_id, None)


def _on_task_done(task_id: str, task: asyncio.Task):
    ti = tasks.get(task_id)
    if ti is None:
        return
    ti.error = task.exception()
    if ti.error is None:
        ti.result = task.result()
    ti.completed_at = time.time()
    ti.done.set()
    asyncio.create_task(_evict_after(task_id, LINGER_SECONDS))


def start_task(coro: Coroutine[Any, Any, Any]) -> str:
    task_id = uuid.uuid4().hex
    t = asyncio.create_task(coro)
    tasks[task_id] = TaskInfo(task=t)
    t.add_done_callback(lambda t: _on_task_done(task_id, t))
    return task_id


async def sse_task_stream(
    task_factory: Callable[[], Coroutine[Any, Any, str]],
    request: Request,
    task_id: Optional[str] = None,
) -> AsyncGenerator[dict[str, Any], None]:

    if task_id is None:
        coro = task_factory()
        task_id = start_task(coro)

    ti = tasks.get(task_id)
    if not ti:
        yield {"event": "error", "data": "unknown task_id"}
        return

    yield {"event": "task_id", "data": task_id}
    while not ti.done.is_set():
        try:
            await asyncio.wait_for(asyncio.shield(ti.done.wait()), timeout=10)
        except asyncio.TimeoutError:
            if await request.is_disconnected():
                return

    if ti.error is not None:
        yield {"event": "error", "data": str(ti.error)}
        return

    assert ti.result is not None
    for i in range(0, len(ti.result), CHUNK_SIZE):
        chunk = ti.result[i:i+CHUNK_SIZE]
        yield {
            "event": "end" if i + CHUNK_SIZE >= len(ti.result) else "chunk",
            "data": chunk,
        }
