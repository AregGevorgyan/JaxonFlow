"""SQLite-backed kernel cache for JaxonFlow."""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config import CacheConfig
from .exceptions import CacheError

if TYPE_CHECKING:
    from .spec import CompiledKernel

logger = logging.getLogger(__name__)


@dataclass
class CacheStats:
    """Statistics for cache operations."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_entries: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


@dataclass
class CacheEntry:
    """A single cache entry with metadata."""

    cache_key: str
    operation: str
    code: str
    metadata: dict[str, Any]
    created_at: datetime
    last_used: datetime
    use_count: int


class KernelCache:
    """Persistent cache for generated kernels.

    Uses SQLite for persistence with thread-safe connections.
    Implements LRU eviction when the cache is full.

    Example:
        cache = KernelCache(config)

        # Check cache
        kernel = cache.get("abc123")
        if kernel is None:
            # Generate kernel...
            cache.put("abc123", kernel)
    """

    def __init__(self, config: CacheConfig | None = None) -> None:
        """Initialize the kernel cache.

        Args:
            config: Cache configuration. Uses defaults if not provided.
        """
        self.config = config or CacheConfig()
        self._local = threading.local()
        self._stats = CacheStats()
        self._lock = threading.Lock()

        if self.config.enabled:
            self._init_db()

    @property
    def _conn(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "conn"):
            assert self.config.cache_dir is not None
            self._local.conn = sqlite3.connect(
                str(self.config.cache_dir / "kernels.db"),
                check_same_thread=False,
            )
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_db(self) -> None:
        """Initialize database schema."""
        assert self.config.cache_dir is not None
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)

        with self._conn:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS kernels (
                    cache_key TEXT PRIMARY KEY,
                    operation TEXT NOT NULL,
                    code TEXT NOT NULL,
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    last_used TEXT NOT NULL,
                    use_count INTEGER DEFAULT 1
                )
            """)
            self._conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_last_used ON kernels(last_used)
            """)
            self._conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_operation ON kernels(operation)
            """)

    def get(self, cache_key: str) -> CacheEntry | None:
        """Retrieve kernel from cache.

        Args:
            cache_key: The cache key to look up.

        Returns:
            CacheEntry if found, None otherwise.
        """
        if not self.config.enabled:
            return None

        try:
            cursor = self._conn.execute(
                "SELECT * FROM kernels WHERE cache_key = ?",
                (cache_key,),
            )
            row = cursor.fetchone()

            if row is None:
                with self._lock:
                    self._stats.misses += 1
                return None

            # Update usage statistics
            now = datetime.now().isoformat()
            self._conn.execute(
                "UPDATE kernels SET last_used = ?, use_count = use_count + 1 WHERE cache_key = ?",
                (now, cache_key),
            )
            self._conn.commit()

            with self._lock:
                self._stats.hits += 1

            return CacheEntry(
                cache_key=row["cache_key"],
                operation=row["operation"],
                code=row["code"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                created_at=datetime.fromisoformat(row["created_at"]),
                last_used=datetime.fromisoformat(row["last_used"]),
                use_count=row["use_count"],
            )

        except sqlite3.Error as e:
            logger.error(f"Cache get failed: {e}")
            with self._lock:
                self._stats.misses += 1
            return None

    def put(
        self,
        cache_key: str,
        operation: str,
        code: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store kernel in cache.

        Args:
            cache_key: Unique cache key.
            operation: Operation type (e.g., "matmul").
            code: Generated kernel code.
            metadata: Optional metadata dictionary.
        """
        if not self.config.enabled:
            return

        try:
            self._evict_if_needed()

            now = datetime.now().isoformat()
            metadata_json = json.dumps(metadata) if metadata else None

            self._conn.execute(
                """
                INSERT OR REPLACE INTO kernels
                (cache_key, operation, code, metadata, created_at, last_used, use_count)
                VALUES (?, ?, ?, ?, ?, ?,
                    COALESCE((SELECT use_count FROM kernels WHERE cache_key = ?), 0) + 1
                )
                """,
                (cache_key, operation, code, metadata_json, now, now, cache_key),
            )
            self._conn.commit()

            logger.debug(f"Cached kernel: {cache_key} ({operation})")

        except sqlite3.Error as e:
            logger.error(f"Cache put failed: {e}")
            raise CacheError(f"Failed to cache kernel: {e}") from e

    def put_kernel(self, kernel: CompiledKernel) -> None:
        """Store a CompiledKernel in cache.

        Args:
            kernel: The compiled kernel to cache.
        """
        self.put(
            cache_key=kernel.spec.cache_key(),
            operation=kernel.spec.operation,
            code=kernel.code,
            metadata=kernel.to_cache_dict(),
        )

    def _evict_if_needed(self) -> None:
        """Perform LRU eviction if cache is full."""
        cursor = self._conn.execute("SELECT COUNT(*) as count FROM kernels")
        count = cursor.fetchone()["count"]

        if count >= self.config.max_entries:
            evict_count = int(self.config.max_entries * self.config.eviction_ratio)
            evict_count = max(1, evict_count)

            self._conn.execute(
                """
                DELETE FROM kernels WHERE cache_key IN (
                    SELECT cache_key FROM kernels ORDER BY last_used ASC LIMIT ?
                )
                """,
                (evict_count,),
            )
            self._conn.commit()

            with self._lock:
                self._stats.evictions += evict_count

            logger.info(f"Evicted {evict_count} entries from cache")

    def delete(self, cache_key: str) -> bool:
        """Delete a specific entry from cache.

        Args:
            cache_key: The cache key to delete.

        Returns:
            True if entry was deleted, False if not found.
        """
        if not self.config.enabled:
            return False

        try:
            cursor = self._conn.execute(
                "DELETE FROM kernels WHERE cache_key = ?",
                (cache_key,),
            )
            self._conn.commit()
            return cursor.rowcount > 0

        except sqlite3.Error as e:
            logger.error(f"Cache delete failed: {e}")
            return False

    def clear(self) -> int:
        """Clear all entries from cache.

        Returns:
            Number of entries deleted.
        """
        if not self.config.enabled:
            return 0

        try:
            cursor = self._conn.execute("SELECT COUNT(*) as count FROM kernels")
            count = cursor.fetchone()["count"]

            self._conn.execute("DELETE FROM kernels")
            self._conn.commit()

            logger.info(f"Cleared {count} entries from cache")
            return count

        except sqlite3.Error as e:
            logger.error(f"Cache clear failed: {e}")
            raise CacheError(f"Failed to clear cache: {e}") from e

    def list_entries(
        self,
        operation: str | None = None,
        limit: int = 100,
    ) -> list[CacheEntry]:
        """List cache entries.

        Args:
            operation: Filter by operation type (optional).
            limit: Maximum number of entries to return.

        Returns:
            List of CacheEntry objects.
        """
        if not self.config.enabled:
            return []

        try:
            if operation:
                cursor = self._conn.execute(
                    "SELECT * FROM kernels WHERE operation = ? ORDER BY last_used DESC LIMIT ?",
                    (operation, limit),
                )
            else:
                cursor = self._conn.execute(
                    "SELECT * FROM kernels ORDER BY last_used DESC LIMIT ?",
                    (limit,),
                )

            entries = []
            for row in cursor:
                entries.append(
                    CacheEntry(
                        cache_key=row["cache_key"],
                        operation=row["operation"],
                        code=row["code"],
                        metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                        created_at=datetime.fromisoformat(row["created_at"]),
                        last_used=datetime.fromisoformat(row["last_used"]),
                        use_count=row["use_count"],
                    )
                )
            return entries

        except sqlite3.Error as e:
            logger.error(f"Cache list failed: {e}")
            return []

    @property
    def stats(self) -> CacheStats:
        """Get cache statistics."""
        if self.config.enabled:
            try:
                cursor = self._conn.execute("SELECT COUNT(*) as count FROM kernels")
                with self._lock:
                    self._stats.total_entries = cursor.fetchone()["count"]
            except sqlite3.Error:
                pass
        return self._stats

    def close(self) -> None:
        """Close database connections."""
        if hasattr(self._local, "conn"):
            self._local.conn.close()
            del self._local.conn
