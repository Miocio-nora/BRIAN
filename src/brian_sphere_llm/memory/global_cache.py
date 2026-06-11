from __future__ import annotations


class GlobalCacheNotEnabled(RuntimeError):
    pass


def require_global_kv_enabled() -> None:
    raise GlobalCacheNotEnabled("Global KV is intentionally deferred until route-core gates pass.")
