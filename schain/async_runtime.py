"""One long-lived asyncio loop for sync pipeline calls that reuse async clients.

The public pipeline remains synchronous.  Running each document with ``asyncio.run``
closed the loop that OpenAI/httpx async resources had become bound to.  A dedicated
background loop gives every extraction call in the process the same lifecycle while
leaving retrieval and provider abstractions unchanged.
"""

from __future__ import annotations

import asyncio
import atexit
import threading
from collections.abc import Coroutine
from typing import Any, TypeVar


T = TypeVar("T")


class _PersistentLoop:
    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._serve,
            name="schain-async-runtime",
            daemon=True,
        )
        self._thread.start()

    def _serve(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def run(self, coroutine: Coroutine[Any, Any, T]) -> T:
        if not self._loop.is_running():
            coroutine.close()
            raise RuntimeError("SChain async runtime is not running")
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop).result()

    def close(self) -> None:
        if not self._loop.is_running():
            return

        async def shutdown() -> None:
            current = asyncio.current_task()
            pending = [task for task in asyncio.all_tasks() if task is not current]
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

        asyncio.run_coroutine_threadsafe(shutdown(), self._loop).result(timeout=5)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
        self._loop.close()


_runtime: _PersistentLoop | None = None
_lock = threading.Lock()


def run_async(coroutine: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine on the process-wide extraction loop."""
    global _runtime
    with _lock:
        if _runtime is None:
            _runtime = _PersistentLoop()
    return _runtime.run(coroutine)


def close_async_runtime() -> None:
    global _runtime
    with _lock:
        runtime, _runtime = _runtime, None
    if runtime is not None:
        runtime.close()


atexit.register(close_async_runtime)
