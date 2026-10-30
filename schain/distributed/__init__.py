"""PostgreSQL-backed multi-worker Expand coordination.

The package deliberately does not replace :mod:`schain.state`.  SQLite remains the
local/offline backend; these classes are loaded only for an explicitly configured
PostgreSQL distributed run.
"""

from .config import DistributedSettings, distributed_settings
from .models import (
    IdentityConflictError,
    OutboxLease,
    StaleLeaseError,
    TaskLease,
)

__all__ = [
    "DistributedSettings",
    "IdentityConflictError",
    "OutboxLease",
    "StaleLeaseError",
    "TaskLease",
    "distributed_settings",
]
