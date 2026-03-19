import os
from typing import Optional

from ors.client.environment import (
    AsyncEnvironment,
    AsyncEnvironmentsAPI,
    Environment,
    EnvironmentsAPI,
)

ORS_API_KEY_ENV_VAR = "ORS_API_KEY"


class AsyncORS:
    """Async client for ORS servers."""

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        api_key: Optional[str] = None,
    ):
        self.base_url = base_url
        self.api_key = api_key or os.getenv(ORS_API_KEY_ENV_VAR)
        self._environments_api: Optional[AsyncEnvironmentsAPI] = None

    @property
    def environments(self) -> AsyncEnvironmentsAPI:
        if self._environments_api is None:
            self._environments_api = AsyncEnvironmentsAPI(
                base_url=self.base_url,
                api_key=self.api_key,
            )
        return self._environments_api

    def environment(self, name: str) -> AsyncEnvironment:
        """Get an environment by name."""
        return self.environments.get(name)


class ORS:
    """Sync client for ORS servers."""

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        api_key: Optional[str] = None,
    ):
        self.base_url = base_url
        self.api_key = api_key or os.getenv(ORS_API_KEY_ENV_VAR)
        self._environments_api: Optional[EnvironmentsAPI] = None

    @property
    def environments(self) -> EnvironmentsAPI:
        if self._environments_api is None:
            self._environments_api = EnvironmentsAPI(
                base_url=self.base_url,
                api_key=self.api_key,
            )
        return self._environments_api

    def environment(self, name: str) -> Environment:
        """Get an environment by name."""
        return self.environments.get(name)

    def close(self):
        """Clean up resources."""
        if self._environments_api is not None:
            self._environments_api.close()

    def __enter__(self) -> "ORS":
        return self

    def __exit__(self, *exc):
        self.close()
