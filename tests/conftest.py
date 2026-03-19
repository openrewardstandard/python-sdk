import pytest
from sse_starlette.sse import AppStatus


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def _reset_sse_app_status():
    """Reset sse_starlette's class-level Event between tests to avoid
    'bound to a different event loop' errors in async test suites."""
    AppStatus.should_exit_event = None
    yield
    AppStatus.should_exit_event = None
