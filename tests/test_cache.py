"""Tests for JaxonFlow kernel cache."""

from __future__ import annotations

import pytest

from jaxonflow.cache import CacheStats, KernelCache
from jaxonflow.config import CacheConfig
from jaxonflow.spec import CompiledKernel, KernelSpec


class TestKernelCache:
    def test_put_and_get(self, kernel_cache):
        kernel_cache.put("key1", "matmul", "# code", {"meta": "data"})
        entry = kernel_cache.get("key1")

        assert entry is not None
        assert entry.cache_key == "key1"
        assert entry.operation == "matmul"
        assert entry.code == "# code"
        assert entry.metadata == {"meta": "data"}
        assert entry.use_count >= 1

    def test_get_missing_key(self, kernel_cache):
        entry = kernel_cache.get("nonexistent")
        assert entry is None

    def test_put_replaces_existing(self, kernel_cache):
        kernel_cache.put("key1", "matmul", "# code v1")
        kernel_cache.put("key1", "matmul", "# code v2")

        entry = kernel_cache.get("key1")
        assert entry is not None
        assert entry.code == "# code v2"

    def test_delete(self, kernel_cache):
        kernel_cache.put("key1", "matmul", "# code")
        assert kernel_cache.delete("key1") is True
        assert kernel_cache.get("key1") is None

    def test_delete_missing(self, kernel_cache):
        assert kernel_cache.delete("nonexistent") is False

    def test_clear(self, kernel_cache):
        kernel_cache.put("key1", "matmul", "# code1")
        kernel_cache.put("key2", "softmax", "# code2")

        count = kernel_cache.clear()
        assert count == 2
        assert kernel_cache.get("key1") is None
        assert kernel_cache.get("key2") is None

    def test_list_entries(self, kernel_cache):
        kernel_cache.put("key1", "matmul", "# code1")
        kernel_cache.put("key2", "softmax", "# code2")
        kernel_cache.put("key3", "matmul", "# code3")

        all_entries = kernel_cache.list_entries()
        assert len(all_entries) == 3

        matmul_entries = kernel_cache.list_entries(operation="matmul")
        assert len(matmul_entries) == 2

    def test_list_entries_limit(self, kernel_cache):
        for i in range(10):
            kernel_cache.put(f"key{i}", "op", f"# code{i}")

        entries = kernel_cache.list_entries(limit=5)
        assert len(entries) == 5

    def test_stats(self, kernel_cache):
        kernel_cache.put("key1", "matmul", "# code")

        # Hit
        kernel_cache.get("key1")
        # Miss
        kernel_cache.get("key2")

        stats = kernel_cache.stats
        assert stats.hits == 1
        assert stats.misses == 1
        assert stats.hit_rate == pytest.approx(0.5)
        assert stats.total_entries == 1

    def test_lru_eviction(self, tmp_path):
        config = CacheConfig(
            enabled=True,
            cache_dir=tmp_path / "eviction_cache",
            max_entries=5,
            eviction_ratio=0.4,  # Evict 2 of 5
        )
        cache = KernelCache(config)

        try:
            # Fill cache
            for i in range(5):
                cache.put(f"key{i}", "op", f"# code{i}")

            # Access some entries to update last_used
            cache.get("key3")
            cache.get("key4")

            # Adding a 6th should trigger eviction of oldest 2
            cache.put("key5", "op", "# code5")

            # key0 and key1 should be evicted (oldest, unused)
            assert cache.get("key0") is None
            assert cache.get("key1") is None

            # key3, key4, key5 should still be there
            assert cache.get("key3") is not None
            assert cache.get("key4") is not None
            assert cache.get("key5") is not None
        finally:
            cache.close()

    def test_put_kernel(self, kernel_cache, matmul_spec):
        kernel = CompiledKernel(
            spec=matmul_spec,
            code="# generated kernel code",
            callable=lambda: None,
            speedup_vs_reference=1.5,
            iterations_to_generate=3,
        )
        kernel_cache.put_kernel(kernel)

        entry = kernel_cache.get(matmul_spec.cache_key())
        assert entry is not None
        assert entry.code == "# generated kernel code"
        assert entry.metadata["speedup_vs_reference"] == 1.5

    def test_disabled_cache(self, disabled_cache):
        disabled_cache.put("key1", "op", "code")
        assert disabled_cache.get("key1") is None
        assert disabled_cache.list_entries() == []
        assert disabled_cache.delete("key1") is False
        assert disabled_cache.clear() == 0


class TestCacheStats:
    def test_hit_rate_empty(self):
        stats = CacheStats()
        assert stats.hit_rate == 0.0

    def test_hit_rate_calculation(self):
        stats = CacheStats(hits=3, misses=7)
        assert stats.hit_rate == pytest.approx(0.3)
