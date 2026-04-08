"""Conftest for test_extractors."""

from __future__ import annotations

import signal
import sys
from typing import Any


def pytest_configure(config: Any) -> None:  # type: ignore[explicit-any]
    """Patch testing.postgresql for Windows compatibility.

    testing.postgresql sends SIGINT to stop postgres processes, but Windows
    does not support SIGINT in subprocess.Popen.send_signal(). Remap the
    Postgresql.terminate() method to use SIGTERM (TerminateProcess) instead.
    This runs before any test module is imported, so the patch is in place
    before the module-level PostgresqlFactory(...) call executes.
    """
    if sys.platform != "win32":
        return

    try:
        import testing.common.database as _tcd  # type: ignore[import-untyped]
        import testing.postgresql as _tpg  # type: ignore[import-untyped]

        def _win32_pg_terminate(self: Any, _sig: Any = None) -> None:  # type: ignore[explicit-any]
            _tcd.Database.terminate(self, signal.SIGTERM)

        _tpg.Postgresql.terminate = _win32_pg_terminate  # type: ignore[method-assign]
    except ImportError:
        pass
