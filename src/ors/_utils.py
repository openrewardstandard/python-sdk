import inspect
from typing import Awaitable, TypeVar, Union

T = TypeVar("T")


async def maybe_await(x: Union[T, Awaitable[T]]) -> T:
    if inspect.isawaitable(x):
        return await x
    else:
        return x
