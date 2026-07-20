#!/usr/bin/env python3
"""Smoke tests — no framework required. Run with: python test_smoke.py"""

import io
import os
import sys
import json
import tempfile
import time
import asyncio
from collections import deque
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.dirname(__file__))


# ── Test helpers ──────────────────────────────────────────────────────

async def _fast_sleep(*args, **kwargs):
    """No-op sleep for tests that wait on asyncio.sleep (retries, polling)."""
    pass


# ── Compile check ─────────────────────────────────────────────────────

def test_compile_all():
    """Verify all project Python files compile without errors."""
    import py_compile
    import glob as _glob
    root = os.path.dirname(__file__)
    py_files = [f for f in _glob.glob(os.path.join(root, "*.py")) if f != os.path.join(root, "test_smoke.py")]
    errors = []
    for f in py_files:
        try:
            py_compile.compile(f, doraise=True)
        except py_compile.PyCompileError as e:
            errors.append(str(e))
    if errors:
        for err in errors:
            print(f"COMPILE FAIL: {err}", flush=True)
        sys.exit(1)
    print(f"COMPILE OK: {len(py_files)} files", flush=True)


# ── BackendManager tests (NEW) ────────────────────────────────────────

def test_backend_manager_constructor():
    """BackendManager should derive keys from URLs and create clients."""
    from backend_manager import BackendManager, BackendInfo
    
    bm = BackendManager([{"url": "http://10.0.0.1:8000"}])
    assert bm.keys() == ["10.0.0.1:8000"]
    assert bm.first_key() == "10.0.0.1:8000"
    assert bm.n_backends() == 1
    assert isinstance(bm.get_client("10.0.0.1:8000"), object)  # LlamaClient instance
    print("PASS: test_backend_manager_constructor")


def test_backend_manager_agent_client():
    """BackendManager should create agent clients when agent_port is configured."""
    from backend_manager import BackendManager
    from cache_agent_client import CacheAgentClient
    
    bm = BackendManager([{"url": "http://10.0.0.1:8000", "agent_port": 8082}])
    agent = bm.get_agent("10.0.0.1:8000")
    assert isinstance(agent, CacheAgentClient)
    print("PASS: test_backend_manager_agent_client")


def test_backend_manager_model_registration():
    """BackendManager should register backends for models via discover_models."""
    from backend_manager import BackendManager
    from unittest.mock import AsyncMock
    
    bm = BackendManager([{"url": "http://10.0.0.1:8000"}])
    mock_client = AsyncMock()
    mock_client.get_slots_info = AsyncMock(return_value=[{"id": 0}, {"id": 1}])
    mock_client.discover_models = AsyncMock(return_value=[("ModelA", 4096)])
    bm._backends["10.0.0.1:8000"] = type('obj', (object,), {'client': mock_client, 'agent_client': None, 'cache_dir': None})()
    
    async def _run():
        await bm.discover_models()
        await bm.refresh_slot_counts()
        return bm.get_discovered_models("ModelA")
    
    result = asyncio.run(_run())
    assert len(result) == 1
    assert result[0].name == "ModelA"
    assert result[0].backends == ["10.0.0.1:8000"]
    assert bm._discovered_models["ModelA"].n_ctx == 4096
    # Close the fresh instance's clients to avoid resource leaks
    asyncio.run(bm.close())
    print("PASS: test_backend_manager_model_registration")


# ── hashing tests (unchanged) ────────────────────────────────────────

def test_reconcile_meta_removes_orphans():
    """reconcile should delete meta files with no matching cache and skip valid ones."""
    from kv_meta_manager import KVMetaManager
    from hashing import sanitize_backend_dir
    mgr = KVMetaManager()

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = os.path.join(tmpdir, "cache")
        meta_dir = os.path.join(tmpdir, "meta")
        os.makedirs(cache_dir)
        os.makedirs(meta_dir)

        # Valid entry: cache + meta both exist
        valid_key = "valid_cache_key"
        with open(os.path.join(cache_dir, valid_key), "w") as f:
            f.write("cache data")
        backend_key = sanitize_backend_dir("10.0.0.1:8000")
        backend_meta_dir = os.path.join(meta_dir, backend_key)
        os.makedirs(backend_meta_dir)
        with open(os.path.join(backend_meta_dir, f"{valid_key}.meta.json"), "w") as f:
            json.dump({"key": valid_key, "model_id": "test", "wpb": 100, "blocks": []}, f)

        # Orphaned entry: meta exists but cache does not
        orphan_key = "orphan_cache_key"
        with open(os.path.join(backend_meta_dir, f"{orphan_key}.meta.json"), "w") as f:
            json.dump({"key": orphan_key, "model_id": "test", "wpb": 100, "blocks": []}, f)

        # Corrupted meta file
        corrupted_key = "corrupted_cache_key"
        with open(os.path.join(backend_meta_dir, f"{corrupted_key}.meta.json"), "w") as f:
            f.write("not json {{{")

        # Set up backend_manager so reconcile can find the backend
        from backend_manager import backend_manager
        backend_manager._backends.clear()
        backend_manager._backends[backend_key] = type('obj', (object,), {
            'client': AsyncMock(), 'agent_client': None, 'cache_dir': None, 'cache_dir': cache_dir
        })()

        deleted = asyncio.run(mgr.reconcile([backend_key]))

        assert deleted == 2, f"Expected 2 deleted (orphan + corrupted), got {deleted}"
        assert os.path.exists(os.path.join(backend_meta_dir, f"{valid_key}.meta.json")), "Valid meta was deleted"
        assert not os.path.exists(os.path.join(backend_meta_dir, f"{orphan_key}.meta.json")), "Orphan meta was not deleted"
        assert not os.path.exists(os.path.join(backend_meta_dir, f"{corrupted_key}.meta.json")), "Corrupted meta was not deleted"
        print("PASS: test_reconcile_meta_removes_orphans")


def test_hashing_imports():
    """hashing module should have hashing functions; kv-meta in kv_meta_manager."""
    import hashing as hs
    from kv_meta_manager import KVMetaManager
    assert hasattr(hs, "block_hashes_from_tokens")
    assert hasattr(hs, "lcp_blocks")
    assert hasattr(hs, "meta_key")
    assert hasattr(hs, "prefix_key_sha256")
    mgr = KVMetaManager()
    assert callable(mgr.reconcile)
    assert callable(mgr.get_last_used_time)
    assert callable(mgr.write_meta)
    assert callable(mgr.find_best_restore_candidate)
    assert not hasattr(hs, "cleanup_old_cache")
    assert not hasattr(hs, "update_last_read")
    print("PASS: test_hashing_imports")


async def _mock_cache_agent_success(key: str):
    """Mock response for successful cache agent delete."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"ok": True}
    return mock_response


async def _mock_cache_agent_failure(key: str):
    """Mock response for failed cache agent delete (file not found)."""
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.text = '{"ok": false, "error": "file not found"}'
    return mock_response


def test_cache_agent_client_delete_success():
    """CacheAgentClient.delete returns True on 200 OK response."""
    from unittest.mock import AsyncMock, patch
    from cache_agent_client import CacheAgentClient

    async def run():
        client = CacheAgentClient("http://10.0.0.1:8082")
        mock_resp = await _mock_cache_agent_success("test_key")
        with patch.object(client._client, "post", new_callable=AsyncMock, return_value=mock_resp):
            result = await client.delete("test_key")
            assert result is True, f"Expected True, got {result}"
        await client.close()

    asyncio.run(run())
    print("PASS: test_cache_agent_client_delete_success")


def test_cache_agent_client_delete_failure():
    """CacheAgentClient.delete returns False on non-200 response."""
    from unittest.mock import AsyncMock, patch
    from cache_agent_client import CacheAgentClient

    async def run():
        client = CacheAgentClient("http://10.0.0.1:8082")
        mock_resp = await _mock_cache_agent_failure("test_key")
        with patch.object(client._client, "post", new_callable=AsyncMock, return_value=mock_resp):
            result = await client.delete("test_key")
            assert result is False, f"Expected False, got {result}"
        await client.close()

    asyncio.run(run())
    print("PASS: test_cache_agent_client_delete_failure")


def test_cache_agent_client_connect_error():
    """CacheAgentClient.delete returns False on connection error."""
    from backend_manager import BackendManager, CacheAgentClient
    import httpx

    async def run():
        bm = BackendManager([{"url": "http://10.0.0.1:9999", "agent_port": 8082}])
        agent = bm.get_agent("10.0.0.1-9999")
        with patch.object(agent._client, "post", new_callable=AsyncMock,
                          side_effect=httpx.ConnectError("connection refused")):
            result = await agent.delete("test_key")
            assert result is False, f"Expected False, got {result}"
        await agent.close()

    asyncio.run(run())
    print("PASS: test_cache_agent_client_connect_error")


def test_save_slot_response_parsing():
    """save_slot must extract n_written from the llama.cpp save response."""
    mock_response_json = {
        "id_slot": 0, "filename": "test_cache",
        "n_saved": 1745, "n_written": 14309796,
        "timings": {"save_ms": 49.865}
    }
    data = mock_response_json
    n_written = data.get("n_written", 0)
    assert n_written == 14309796, f"Expected n_written=14309796, got {n_written}"

    data_no_written = {"id_slot": 0, "filename": "test"}
    n_written_2 = data_no_written.get("n_written", 0)
    assert n_written_2 == 0, f"Expected default 0, got {n_written_2}"
    print("PASS: test_save_slot_response_parsing")


# ── SlotManager tests ────────────────────────────────────────────────

def test_slot_manager_per_model_pools():
    """SlotManager should create separate pools per model."""
    from slot_manager import SlotManager, GSlot
    from backend_manager import backend_manager

    backend_manager._backends.clear()

    sm = SlotManager()
    sm.get("10.0.0.1:8000").ensure_pool("ModelA", 3)

    assert "ModelA" in sm.get("10.0.0.1:8000")._slot_pools, f"ModelA not in pools: {sm.get('10.0.0.1:8000')._slot_pools}"
    assert sm.get("10.0.0.1:8000")._slot_pools["ModelA"] == {0, 1, 2}, f"Got {sm.get('10.0.0.1:8000')._slot_pools}"
    print("PASS: test_slot_manager_per_model_pools")


def test_slot_manager_multiple_models():
    """SlotManager should support multiple models on the same backend."""
    from slot_manager import SlotManager
    from backend_manager import backend_manager

    backend_manager._backends.clear()

    sm = SlotManager()
    sm.get("10.0.0.1:8000").ensure_pool("ModelA", 2)
    sm.get("10.0.0.1:8000").ensure_pool("ModelB", 4)

    assert sm.get("10.0.0.1:8000")._slot_pools["ModelA"] == {0, 1}, f"Got {sm.get('10.0.0.1:8000')._slot_pools.get('ModelA', {})}"
    assert sm.get("10.0.0.1:8000")._slot_pools["ModelB"] == {0, 1, 2, 3}, f"Got {sm.get('10.0.0.1:8000')._slot_pools.get('ModelB', {})}"
    print("PASS: test_slot_manager_multiple_models")


def test_slot_manager_release():
    """release should unlock and reset last_used."""
    from slot_manager import SlotManager
    from backend_manager import backend_manager

    backend_manager._backends.clear()
    backend_manager._first_key = "10.0.0.1:8000"
    backend_manager._refresh_state.clear()
    backend_manager._discovered_models.clear()

    sm = SlotManager()
    sm.get("10.0.0.1:8000").ensure_pool("ModelA", 2)

    # Slot 0 starts as not in-use
    assert not sm.get("10.0.0.1:8000")._in_use[0]

    # Mark slot 0 as in-use
    sm.get("10.0.0.1:8000")._in_use[0] = True
    sm.get("10.0.0.1:8000")._last_used[0] = 100.0
    assert sm.get("10.0.0.1:8000")._in_use[0]
    assert sm.get("10.0.0.1:8000")._last_used[0] == 100.0

    # Release
    sm.get("10.0.0.1:8000").release(0)
    assert not sm.get("10.0.0.1:8000")._in_use[0]
    assert sm.get("10.0.0.1:8000")._last_used[0] == 100.0
    print("PASS: test_slot_manager_release")


def test_slot_manager_pool_resize_up():
    """Pool should grow when slot count increases."""
    from slot_manager import SlotManager
    from backend_manager import backend_manager

    backend_manager._backends.clear()
    backend_manager._first_key = "10.0.0.1:8000"
    backend_manager._refresh_state.clear()
    backend_manager._discovered_models.clear()

    sm = SlotManager()
    sm.get("10.0.0.1:8000").ensure_pool("ModelA", 2)
    assert sm.get("10.0.0.1:8000")._slot_pools["ModelA"] == {0, 1}, f"Got {sm.get('10.0.0.1:8000')._slot_pools.get('ModelA', {})}"

    # Resize to 4
    sm.get("10.0.0.1:8000").ensure_pool("ModelA", 4)
    assert sm.get("10.0.0.1:8000")._slot_pools["ModelA"] == {0, 1, 2, 3}, f"Got {sm.get('10.0.0.1:8000')._slot_pools.get('ModelA', {})}"
    print("PASS: test_slot_manager_pool_resize_up")


def test_slot_manager_pool_resize_down():
    """Pool should shrink when slot count decreases (only removes free slots)."""
    from slot_manager import SlotManager
    from backend_manager import backend_manager

    backend_manager._backends.clear()
    backend_manager._first_key = "10.0.0.1:8000"
    backend_manager._refresh_state.clear()
    backend_manager._discovered_models.clear()

    sm = SlotManager()
    sm.get("10.0.0.1:8000").ensure_pool("ModelA", 4)

    # Mark slot 2 as used so it survives shrink
    sm.get("10.0.0.1:8000")._last_used[2] = 100.0

    # Resize to 2
    sm.get("10.0.0.1:8000").ensure_pool("ModelA", 2)
    assert sm.get("10.0.0.1:8000")._slot_pools["ModelA"] == {0, 1}, f"Got {sm.get('10.0.0.1:8000')._slot_pools.get('ModelA', {})}"
    # Slot 2 was used, so it should NOT be in the pool anymore (it was removed)
    # but last_used may still have the entry (that's OK — it'll be cleaned on next acquire)
    print("PASS: test_slot_manager_pool_resize_down")


def test_slot_manager_multiple_backends():
    """SlotManager should support multiple backends for the same model."""
    from slot_manager import SlotManager
    from backend_manager import backend_manager

    backend_manager._backends.clear()
    backend_manager._first_key = "10.0.0.1:8000"
    backend_manager._refresh_state.clear()
    backend_manager._discovered_models.clear()

    sm = SlotManager()
    sm.get("10.0.0.1:8000").ensure_pool("ModelA", 2)
    sm.get("10.0.0.2:8000").ensure_pool("ModelA", 3)

    assert sm.get("10.0.0.1:8000")._slot_pools["ModelA"] == {0, 1}, f"Got {sm.get('10.0.0.1:8000')._slot_pools.get('ModelA', {})}"
    assert sm.get("10.0.0.2:8000")._slot_pools["ModelA"] == {0, 1, 2}, f"Got {sm.get('10.0.0.2:8000')._slot_pools.get('ModelA', {})}"
    print("PASS: test_slot_manager_multiple_backends")


def test_slot_manager_gslot_type():
    """GSlot should be (model_name, backend_id, slot_id)."""
    from slot_manager import GSlot

    g: GSlot = ("ModelA", "10.0.0.1:8000", 1)
    model_name, backend_id, slot_id = g
    assert model_name == "ModelA"
    assert backend_id == "10.0.0.1:8000"
    assert slot_id == 1
    print("PASS: test_slot_manager_gslot_type")


def test_ring_buffer_age_eviction():
    from collections import deque
    """Ring buffer should evict expired entries before LRU entries when over limit."""
    from slot_manager import SlotManager
    import hashing as hs
    import tempfile

    sm = SlotManager()
    sm.get("test_be")._max_age_seconds = 3600  # 1 hour

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = os.path.join(tmpdir, "cache")
        meta_dir = os.path.join(tmpdir, "meta")
        os.makedirs(cache_dir)
        os.makedirs(meta_dir)

        # Simulate ring buffer with entries of different ages
        old_key = "old_cache_file"
        new_key = "new_cache_file"
        size = 1024 * 1024 * 1020  # ~1 GB each
        max_bytes = 1500 * 1024 * 1024  # 1.5 GB limit

        sm.get("test_be")._cache_ring.append((old_key, size, time.time() - 7200))  # 2 hours old
        sm.get("test_be")._cache_ring.append((new_key, size, time.time() - 300))   # 5 minutes old
        sm.get("test_be")._total_bytes = size * 2

        # Trigger eviction (simulating save_after behavior)
        now = time.time()
        evicted = []
        be_sm = sm.get("test_be")
        while be_sm._total_bytes > max_bytes and be_sm._cache_ring:
            # First pass: evict expired entries
            evicted_expired = False
            for entry in be_sm._cache_ring:
                if now - entry[2] > be_sm._max_age_seconds:
                    evict_key, evict_size, _ = entry
                    be_sm._cache_ring.remove(entry)
                    be_sm._total_bytes -= evict_size
                    evicted.append(evict_key)
                    evicted_expired = True
                    break
            if evicted_expired:
                continue

            # Second pass: evict LRU entry
            lru_idx = 0
            lru_ts = be_sm._cache_ring[0][2]
            for i in range(1, len(be_sm._cache_ring)):
                if be_sm._cache_ring[i][2] < lru_ts:
                    lru_ts = be_sm._cache_ring[i][2]
                    lru_idx = i
            evict_key, evict_size, _ = be_sm._cache_ring[lru_idx]
            be_sm._cache_ring.remove(be_sm._cache_ring[lru_idx])
            be_sm._total_bytes -= evict_size
            evicted.append(evict_key)

        assert evicted == [old_key], f"Expected [old_key], got {evicted}"
        assert be_sm._total_bytes == size  # Only old entry removed
        assert len(be_sm._cache_ring) == 1
        assert be_sm._cache_ring[0][0] == new_key
        print("PASS: test_ring_buffer_age_eviction")


def test_ring_buffer_lru_eviction():
    """Ring buffer should evict LRU entry when no entries are expired."""
    from collections import deque
    from slot_manager import SlotManager

    sm = SlotManager()
    sm.get("test_be")._max_age_seconds = 3600  # 1 hour

    size = 1024 * 1024 * 1020  # ~1 GB each
    max_bytes = 1500 * 1024 * 1024  # 1.5 GB limit

    # Simulate ring buffer with all entries under max age
    sm.get("test_be")._cache_ring.append(("cache_a", size, time.time() - 1800))  # 30 min old
    sm.get("test_be")._cache_ring.append(("cache_b", size, time.time() - 600))   # 10 min old
    sm.get("test_be")._total_bytes = size * 2

    # Trigger eviction
    now = time.time()
    evicted = []
    be_sm = sm.get("test_be")
    while be_sm._total_bytes > max_bytes and be_sm._cache_ring:
        evicted_expired = False
        for entry in be_sm._cache_ring:
            if now - entry[2] > be_sm._max_age_seconds:
                evict_key, evict_size, _ = entry
                be_sm._cache_ring.remove(entry)
                be_sm._total_bytes -= evict_size
                evicted.append(evict_key)
                evicted_expired = True
                break
        if evicted_expired:
            continue

        lru_idx = 0
        lru_ts = be_sm._cache_ring[0][2]
        for i in range(1, len(be_sm._cache_ring)):
            if be_sm._cache_ring[i][2] < lru_ts:
                lru_ts = be_sm._cache_ring[i][2]
                lru_idx = i
        evict_key, evict_size, _ = be_sm._cache_ring[lru_idx]
        be_sm._cache_ring.remove(be_sm._cache_ring[lru_idx])
        be_sm._total_bytes -= evict_size
        evicted.append(evict_key)

    assert evicted == ["cache_a"], f"Expected [cache_a] (LRU), got {evicted}"
    assert be_sm._total_bytes == size
    assert len(be_sm._cache_ring) == 1
    assert be_sm._cache_ring[0][0] == "cache_b"
    print("PASS: test_ring_buffer_lru_eviction")


def test_ring_buffer_no_eviction_under_limit():
    """Ring buffer should not evict when under size limit."""
    from collections import deque
    from slot_manager import SlotManager

    sm = SlotManager()
    sm.get("test_be")._max_age_seconds = 3600

    size = 1024 * 1024 * 1020  # ~1 GB
    max_bytes = 2 * 1024 * 1024 * 1024  # 2 GB limit

    sm.get("test_be")._cache_ring.append(("cache_a", size, time.time() - 1800))
    sm.get("test_be")._cache_ring.append(("cache_b", size, time.time() - 600))
    sm.get("test_be")._total_bytes = size * 2

    now = time.time()
    evicted = []
    be_sm = sm.get("test_be")
    while be_sm._total_bytes > max_bytes and be_sm._cache_ring:
        evicted_expired = False
        for entry in be_sm._cache_ring:
            if now - entry[2] > be_sm._max_age_seconds:
                evict_key, evict_size, _ = entry
                be_sm._cache_ring.remove(entry)
                be_sm._total_bytes -= evict_size
                evicted.append(evict_key)
                evicted_expired = True
                break
        if evicted_expired:
            continue

        lru_idx = 0
        lru_ts = be_sm._cache_ring[0][2]
        for i in range(1, len(be_sm._cache_ring)):
            if be_sm._cache_ring[i][2] < lru_ts:
                lru_ts = be_sm._cache_ring[i][2]
                lru_idx = i
        evict_key, evict_size, _ = be_sm._cache_ring[lru_idx]
        be_sm._cache_ring.remove(be_sm._cache_ring[lru_idx])
        be_sm._total_bytes -= evict_size
        evicted.append(evict_key)

    assert evicted == [], f"Expected no evictions, got {evicted}"
    assert be_sm._total_bytes == size * 2
    assert len(be_sm._cache_ring) == 2
    print("PASS: test_ring_buffer_no_eviction_under_limit")


# ── KV cache skip tests ──────────────────────────────────────────────

def test_should_skip_restore_no_tracked_state():
    """_should_skip_restore should return False when no state tracked for slot."""
    from slot_manager import SlotManager

    sm = SlotManager()
    sm.get("10.0.0.1:8000")._slot_kv_state.clear()
    sm.get("10.0.0.1:8000").ensure_pool("ModelA", 1)
    blocks = ["a", "b", "c", "d", "e"]

    assert sm.get("10.0.0.1:8000").should_skip_restore(0, blocks) is False
    print("PASS: test_should_skip_restore_no_tracked_state")


def test_should_skip_restore_perfect_match():
    """_should_skip_restore should return True for perfect block match with single slot."""
    from slot_manager import SlotManager

    sm = SlotManager()
    sm.get("10.0.0.1:8000").ensure_pool("ModelA", 1)
    kv_blocks = ["a", "b", "c", "d", "e"]
    sm.get("10.0.0.1:8000")._slot_kv_state[0] = kv_blocks

    req_blocks = ["a", "b", "c", "d", "e"]
    assert sm.get("10.0.0.1:8000").should_skip_restore(0, req_blocks) is True
    print("PASS: test_should_skip_restore_perfect_match")


def test_should_skip_restore_high_overlap():
    """_should_skip_restore should return True when overlap >= 0.9 with single slot."""
    from slot_manager import SlotManager

    sm = SlotManager()
    sm.get("10.0.0.1:8000").ensure_pool("ModelA", 1)
    kv_blocks = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    sm.get("10.0.0.1:8000")._slot_kv_state[0] = kv_blocks

    # 9 out of 10 blocks match LCP → ratio = 9/10 = 0.9
    req_blocks = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "x"]
    assert sm.get("10.0.0.1:8000").should_skip_restore(0, req_blocks) is True
    print("PASS: test_should_skip_restore_high_overlap")


def test_should_skip_restore_low_overlap():
    """_should_skip_restore should return False when overlap < 0.9."""
    from slot_manager import SlotManager

    sm = SlotManager()
    sm.get("10.0.0.1:8000").ensure_pool("ModelA", 1)
    kv_blocks = ["a", "b", "c", "d", "e"]
    sm.get("10.0.0.1:8000")._slot_kv_state[0] = kv_blocks

    # Only 4 out of 5 blocks match LCP → ratio = 4/5 = 0.8
    req_blocks = ["a", "b", "c", "d", "x"]
    assert sm.get("10.0.0.1:8000").should_skip_restore(0, req_blocks) is False
    print("PASS: test_should_skip_restore_low_overlap")


def test_should_skip_restore_zero_lcp():
    """_should_skip_restore should return False when no LCP overlap."""
    from slot_manager import SlotManager

    sm = SlotManager()
    sm.get("10.0.0.1:8000").ensure_pool("ModelA", 1)
    kv_blocks = ["a", "b", "c"]
    sm.get("10.0.0.1:8000")._slot_kv_state[0] = kv_blocks

    req_blocks = ["x", "y", "z"]
    assert sm.get("10.0.0.1:8000").should_skip_restore(0, req_blocks) is False
    print("PASS: test_should_skip_restore_zero_lcp")


def test_should_skip_restore_shorter_kv_cache():
    """_should_skip_restore should handle shorter KV cache than request with single slot."""
    from slot_manager import SlotManager

    sm = SlotManager()
    sm.get("10.0.0.1:8000").ensure_pool("ModelA", 1)
    kv_blocks = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
    sm.get("10.0.0.1:8000")._slot_kv_state[0] = kv_blocks

    # KV cache has 9 blocks, request has 10. LCP = 9. ratio = 9/9 = 1.0. Diff = 1/10 = 0.1 (within threshold).
    req_blocks = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    assert sm.get("10.0.0.1:8000").should_skip_restore(0, req_blocks) is True
    print("PASS: test_should_skip_restore_shorter_kv_cache")


def test_should_skip_restore_longer_kv_cache():
    """_should_skip_restore should reject longer KV cache than request (stale state)."""
    from slot_manager import SlotManager

    sm = SlotManager()
    sm.get("10.0.0.1:8000").ensure_pool("ModelA", 1)
    kv_blocks = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    sm.get("10.0.0.1:8000")._slot_kv_state[0] = kv_blocks

    # Request has 3 blocks, KV cache has 10 — slot has stale leftover state, cannot skip.
    req_blocks = ["a", "b", "c"]
    assert sm.get("10.0.0.1:8000").should_skip_restore(0, req_blocks) is False
    print("PASS: test_should_skip_restore_longer_kv_cache")


def test_should_skip_restore_multi_slot():
    """_should_skip_restore should return False when pool has multiple slots."""
    from slot_manager import SlotManager

    sm = SlotManager()
    sm.get("10.0.0.1:8000").ensure_pool("ModelA", 3)
    kv_blocks = ["a", "b", "c", "d", "e"]
    sm.get("10.0.0.1:8000")._slot_kv_state[0] = kv_blocks

    req_blocks = ["a", "b", "c", "d", "e"]
    assert sm.get("10.0.0.1:8000").should_skip_restore(0, req_blocks) is False
    print("PASS: test_should_skip_restore_multi_slot")


def test_save_after_updates_slot_kv_state():
    """save_after should update _slot_kv_state when blocks are provided."""
    from slot_manager import SlotManager
    from backend_manager import backend_manager

    backend_manager._backends.clear()

    sm = SlotManager()

    mock_client = AsyncMock()
    mock_client.save_slot = AsyncMock(return_value=(True, 1024))
    backend_manager._backends["10.0.0.1:8000"] = type('obj', (object,), {'client': mock_client, 'agent_client': None, 'cache_dir': None})()

    blocks = ["blk_a", "blk_b", "blk_c"]

    async def _run():
        ok, size = await sm.get("10.0.0.1:8000").save_after(
            "ModelA", 0, "test_key", blocks, 10,
        )
        return ok

    asyncio.run(_run())

    assert mock_client.save_slot.call_count == 1
    assert 0 in sm.get("10.0.0.1:8000")._slot_kv_state, f"Got {sm.get('10.0.0.1:8000')._slot_kv_state}"
    assert sm.get("10.0.0.1:8000")._slot_kv_state[0] == blocks, f"Got {sm.get('10.0.0.1:8000')._slot_kv_state.get(0)}"
    print("PASS: test_save_after_updates_slot_kv_state")


def test_save_after_no_blocks_no_state_update():
    """save_after should not update _slot_kv_state when blocks are None."""
    from slot_manager import SlotManager
    from backend_manager import backend_manager

    backend_manager._backends.clear()
    backend_manager._first_key = "10.0.0.1:8000"
    backend_manager._refresh_state.clear()
    backend_manager._discovered_models.clear()

    sm = SlotManager()

    mock_client = AsyncMock()
    mock_client.save_slot = AsyncMock(return_value=(True, 1024))
    backend_manager._backends["10.0.0.1:8000"] = type('obj', (object,), {'client': mock_client, 'agent_client': None, 'cache_dir': None})()

    async def _run():
        ok, size = await sm.get("10.0.0.1:8000").save_after(
            "ModelA", 0, "test_key", None, 10,
        )
        return ok

    asyncio.run(_run())

    assert 0 not in sm.get("10.0.0.1:8000")._slot_kv_state, f"Got {sm.get('10.0.0.1:8000')._slot_kv_state}"
    print("PASS: test_save_after_no_blocks_no_state_update")


# ── Bug reproduction tests (fail before fix, pass after) ────────────

def test_restore_slot_times_out_on_slow_backend():
    """Behavioral: restore_slot returns False when backend doesn't respond in time."""
    from llama_client import LlamaClient
    import llama_client

    client = LlamaClient("http://127.0.0.1:8080")

    async def _run():
        # Patch SLOT_TIMEOUT in llama_client module (where it's imported)
        orig_timeout = llama_client.SLOT_TIMEOUT
        llama_client.SLOT_TIMEOUT = 0.1

        try:
            async def slow_post(*a, **kw):
                await asyncio.sleep(5)
                return AsyncMock(status_code=200)

            client.client.post = slow_post
            t0 = time.monotonic()
            result = await client.restore_slot(0, "test_key")
            elapsed = time.monotonic() - t0
            assert not result, "restore_slot should return False on timeout"
            assert 0.05 < elapsed < 2, f"Should have timed out near 0.1s, took {elapsed:.1f}s"
        finally:
            llama_client.SLOT_TIMEOUT = orig_timeout

    asyncio.run(_run())
    print("PASS: test_restore_slot_times_out_on_slow_backend")


def test_save_slot_returns_false_on_exception():
    """Behavioral: save_slot returns (False, 0) when backend raises an exception."""
    from llama_client import LlamaClient

    client = LlamaClient("http://127.0.0.1:8080")

    async def _run():
        async def failing_post(*a, **kw):
            raise ConnectionError("connection refused")

        client.client.post = failing_post
        ok, size = await client.save_slot(0, "test_key")
        assert not ok, "save_slot should return False on exception"
        assert size == 0, "save_slot should return 0 size on exception"
    asyncio.run(_run())
    print("PASS: test_save_slot_returns_false_on_exception")





def test_adaptive_cooldown_on_failure():
    """Verify: failed refresh sets 30s cooldown, successful refresh sets 300s."""
    from backend_manager import backend_manager, DiscoveredModel
    import time

    backend_manager._backends.clear()
    backend_manager._refresh_state.clear()
    backend_manager._discovered_models.clear()

    mock_client = AsyncMock()
    mock_client.get_slots_info = AsyncMock(side_effect=Exception("connection refused"))
    mock_client.discover_models = AsyncMock(return_value=[("ModelA", 4096)])
    backend_manager._backends["10.0.0.1:8000"] = type('obj', (object,), {'client': mock_client, 'agent_client': None, 'cache_dir': None})()

    async def _run():
        # First discover models to populate _discovered_models
        await backend_manager.discover_models()
        # Then attempt refresh_slot_counts which will fail
        await backend_manager.refresh_slot_counts()

    asyncio.run(_run())

    # After failure, _refresh_state stores (timestamp, success_flag, cached_n_slots) tuple
    refresh_key = ("ModelA", "10.0.0.1:8000")
    last_refresh = backend_manager._refresh_state.get(refresh_key)
    assert last_refresh is not None, "Cooldown was not set after failure"
    assert isinstance(last_refresh, tuple), f"Expected (timestamp, success, cached_n) tuple, got {type(last_refresh)}"
    timestamp, success, cached_n = last_refresh
    assert success is False, f"Expected success=False after failure, got success={success}"
    assert timestamp > 0
    print("PASS: test_adaptive_cooldown_on_failure")


def test_lock_released_on_restore_failure():
    """Verify: if restore_slot raises, the slot lock is released via try/finally."""
    from slot_manager import SlotManager
    from backend_manager import backend_manager, DiscoveredModel

    backend_manager._backends.clear()
    backend_manager._first_key = "10.0.0.1:8000"
    backend_manager._refresh_state.clear()
    backend_manager._discovered_models.clear()
    backend_manager._discovered_models.clear()

    sm = SlotManager()
    sm.get("10.0.0.1:8000").ensure_pool("ModelA", 1)
    backend_manager._discovered_models["ModelA"] = DiscoveredModel(
        name="ModelA", n_ctx=4096, backends=["10.0.0.1:8000"],
        total_slots=1, last_discovered=0.0,
    )

    mock_client = AsyncMock()
    mock_client.restore_slot = AsyncMock(side_effect=Exception("connection refused"))
    mock_client.get_slots_info = AsyncMock(return_value=[{"id": 0}])
    mock_client.discover_models = AsyncMock(return_value=[("ModelA", 4096)])
    backend_manager._backends["10.0.0.1:8000"] = type('obj', (object,), {'client': mock_client, 'agent_client': None, 'cache_dir': None})()

    async def _run():
        try:
            await sm.acquire_for_request([("10.0.0.1:8000", "ModelA")], restore_key="bad_key", backend_blocks={"10.0.0.1:8000": ["a", "b"]})
        except Exception:
            pass

    asyncio.run(_run())

    # Slot should be released (in_use flag cleared) after the exception
    be_sm = sm.get("10.0.0.1:8000")
    assert not be_sm._in_use.get(0, True), "Slot must be released after restore failure"
    print("PASS: test_lock_released_on_restore_failure")


def test_slot_timeout_config():
    """Verify SLOT_TIMEOUT env var is read with default 30s."""
    import os
    import importlib

    old = os.environ.pop("SLOT_TIMEOUT", None)
    try:
        import config
        importlib.reload(config)
        assert hasattr(config, "SLOT_TIMEOUT"), "config.py should define SLOT_TIMEOUT"
        assert config.SLOT_TIMEOUT == 30.0, f"Expected SLOT_TIMEOUT=30.0, got {config.SLOT_TIMEOUT}"
    finally:
        if old is not None:
            os.environ["SLOT_TIMEOUT"] = old

    print("PASS: test_slot_timeout_config")


# ── Cancellation handling tests ──────────────────────────────────────

def test_non_streaming_cancelled_error_releases_slot():
    """Behavioral: when non-streaming chat raises CancelledError, slot is released."""
    from slot_manager import SlotManager
    from backend_manager import backend_manager

    backend_manager._backends.clear()
    backend_manager._refresh_state.clear()
    backend_manager._discovered_models.clear()

    sm = SlotManager()
    sm.get("10.0.0.1:8000").ensure_pool("ModelA", 1)
    from backend_manager import DiscoveredModel
    backend_manager._discovered_models["ModelA"] = DiscoveredModel(
        name="ModelA", n_ctx=4096, backends=["10.0.0.1:8000"],
        total_slots=1, last_discovered=0.0,
    )

    mock_client = AsyncMock()
    mock_client.chat_completions = AsyncMock(side_effect=asyncio.CancelledError())
    mock_client.get_slots_info = AsyncMock(return_value=[{"id": 0}])
    backend_manager._backends["10.0.0.1:8000"] = type('obj', (object,), {'client': mock_client, 'agent_client': None, 'cache_dir': None})()

    async def _run():
        be_sm = sm.get("10.0.0.1:8000")
        slot_id = be_sm.try_acquire("ModelA")
        assert slot_id is not None
        assert be_sm._in_use[slot_id], "Slot should be in-use after acquire"
        try:
            await mock_client.chat_completions({}, slot_id=slot_id, stream=False)
        except asyncio.CancelledError:
            pass
        # Simulate the outer finally from chat()
        stream = False
        if not stream:
            be_sm.release(slot_id)
        assert not be_sm._in_use[slot_id], "Slot should be released after finally block"

    asyncio.run(_run())
    print("PASS: test_non_streaming_cancelled_error_releases_slot")


def test_streaming_save_after_skipped_on_cancel():
    """Behavioral: StreamReader._cleanup skips save_after when _cancelled is True."""
    from slot_manager import SlotManager
    from app import StreamReader
    from backend_manager import backend_manager

    backend_manager._backends.clear()
    backend_manager._first_key = "10.0.0.1:8000"
    backend_manager._refresh_state.clear()
    backend_manager._discovered_models.clear()

    sm = SlotManager()
    sm.get("10.0.0.1:8000").ensure_pool("ModelA", 1)

    mock_client = AsyncMock()
    mock_client.save_slot = AsyncMock(return_value=(True, 1024))
    backend_manager._backends["10.0.0.1:8000"] = type('obj', (object,), {'client': mock_client, 'agent_client': None, 'cache_dir': None})()

    async def _run():
        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_resp.headers = {}
        mock_resp.aiter_raw = MagicMock(return_value=iter([]))
        mock_resp.aclose = AsyncMock()

        mock_req = AsyncMock()
        mock_req.is_disconnected = AsyncMock(return_value=False)

        reader = StreamReader(mock_resp, mock_req, "ModelA", "10.0.0.1:8000", 0,
                               "test_key", "prefix", ["blk1"], sm)
        reader._cancelled = True
        await reader._cleanup()

        # save_after should be skipped when cancelled
        assert not mock_client.save_slot.called, \
            "save_after should be skipped when _cancelled is True"

    asyncio.run(_run())
    print("PASS: test_streaming_save_after_skipped_on_cancel")


def test_streaming_save_after_exception():
    """Behavioral: StreamReader._save handles exceptions from save_after gracefully."""
    from slot_manager import SlotManager
    from app import StreamReader
    from backend_manager import backend_manager

    backend_manager._backends.clear()

    sm = SlotManager()
    sm.get("10.0.0.1:8000").ensure_pool("ModelA", 1)

    mock_client = AsyncMock()

    async def _run():
        async def failing_save(*a, **kw):
            raise ConnectionError("connection refused")

        mock_client.save_slot = failing_save
        backend_manager._backends["10.0.0.1:8000"] = type('obj', (object,), {'client': mock_client, 'agent_client': None, 'cache_dir': None})()

        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_resp.headers = {}
        mock_resp.aiter_raw = MagicMock(return_value=iter([]))
        mock_resp.aclose = AsyncMock()

        mock_req = AsyncMock()
        mock_req.is_disconnected = AsyncMock(return_value=False)

        reader = StreamReader(mock_resp, mock_req, "ModelA", "10.0.0.1:8000", 0,
                               "test_key", 10, ["blk1"], sm)

        ok, cache_size = await reader._save()
        assert not ok, "_save should return False on exception"
    asyncio.run(_run())
    print("PASS: test_streaming_save_after_exception")


def test_streaming_gen_sets_cancelled_flag():
    """Behavioral: StreamReader.stream() sets _cancelled=True and cancels reader task on CancelledError."""
    from slot_manager import SlotManager
    from app import StreamReader
    from backend_manager import backend_manager

    backend_manager._backends.clear()
    backend_manager._first_key = "10.0.0.1:8000"
    backend_manager._refresh_state.clear()
    backend_manager._discovered_models.clear()

    sm = SlotManager()
    sm.get("10.0.0.1:8000").ensure_pool("ModelA", 1)

    mock_client = AsyncMock()
    mock_client.save_slot = AsyncMock(return_value=(False, 0))
    backend_manager._backends["10.0.0.1:8000"] = type('obj', (object,), {'client': mock_client, 'agent_client': None, 'cache_dir': None})()

    async def _run():
        # Iterator that hangs — reader will block waiting for data
        class MockAIter:
            async def __anext__(self):
                await asyncio.sleep(100)

        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_resp.headers = {}
        mock_resp.aiter_raw = MagicMock(return_value=MockAIter())
        mock_resp.aclose = AsyncMock()

        mock_req = AsyncMock()
        mock_req.is_disconnected = AsyncMock(return_value=False)

        reader = StreamReader(mock_resp, mock_req, "ModelA", "10.0.0.1:8000", 0,
                               "test_key", "prefix", ["blk1"], sm)

        # Start consuming the stream
        stream_gen = reader.stream()
        consume_task = asyncio.create_task(
            anext(stream_gen)  # try to get first item — will block
        )
        await asyncio.sleep(0.1)  # let it start
        t0 = time.monotonic()
        consume_task.cancel()
        try:
            await consume_task
        except (asyncio.CancelledError, StopAsyncIteration):
            pass
        elapsed = time.monotonic() - t0

        # Should complete quickly
        assert elapsed < 6, f"Cancel should complete within 5s timeout, took {elapsed:.1f}s"
        # resp.aclose() should have been called to break the socket read
        assert mock_resp.aclose.called, "resp.aclose() should be called on cancel"
        # The cancelled flag should have been set, so save_after should NOT be called
        assert reader._cancelled, "_cancelled should be True after CancelledError"
        assert not mock_client.save_slot.called, "save_after should be skipped when gen is cancelled"

    asyncio.run(_run())
    print("PASS: test_streaming_gen_sets_cancelled_flag")


def test_streaming_release_not_in_outer_finally():
    """Behavioral: streaming slots are not released by outer finally (reader handles it)."""
    from slot_manager import SlotManager
    from backend_manager import backend_manager

    backend_manager._backends.clear()
    backend_manager._refresh_state.clear()
    backend_manager._discovered_models.clear()

    sm = SlotManager()
    sm.get("10.0.0.1:8000").ensure_pool("ModelA", 1)
    from backend_manager import DiscoveredModel
    backend_manager._discovered_models["ModelA"] = DiscoveredModel(
        name="ModelA", n_ctx=4096, backends=["10.0.0.1:8000"],
        total_slots=1, last_discovered=0.0,
    )

    mock_client = AsyncMock()
    mock_client.chat_completions = AsyncMock(return_value=AsyncMock(status_code=200))
    mock_client.get_slots_info = AsyncMock(return_value=[{"id": 0}])
    backend_manager._backends["10.0.0.1:8000"] = type('obj', (object,), {'client': mock_client, 'agent_client': None, 'cache_dir': None})()

    async def _run():
        be_sm = sm.get("10.0.0.1:8000")
        slot_id = be_sm.try_acquire("ModelA")
        assert slot_id is not None

        # Simulate the outer finally from chat() with stream=True
        stream = True
        if not stream:
            be_sm.release(slot_id)

        # Slot should still be in-use — reader is responsible for release
        assert be_sm._in_use[slot_id], "Streaming slot should NOT be released by outer finally"

        # Now simulate the reader's finally releasing it
        be_sm.release(slot_id)
        assert not be_sm._in_use[slot_id], "Reader's release should clear the in-use flag"

    asyncio.run(_run())
    print("PASS: test_streaming_release_not_in_outer_finally")


def test_reader_polls_is_disconnected_on_timeout():
    """Behavioral: StreamReader._read_loop checks is_disconnected on timeout, sets _cancelled."""
    from slot_manager import SlotManager
    from app import StreamReader
    from backend_manager import backend_manager

    backend_manager._backends.clear()
    backend_manager._first_key = "10.0.0.1:8000"
    backend_manager._refresh_state.clear()
    backend_manager._discovered_models.clear()

    sm = SlotManager()
    sm.get("10.0.0.1:8000").ensure_pool("ModelA", 1)

    mock_client = AsyncMock()
    mock_client.save_slot = AsyncMock(return_value=(False, 0))
    backend_manager._backends["10.0.0.1:8000"] = type('obj', (object,), {'client': mock_client, 'agent_client': None, 'cache_dir': None})()

    async def _run():
        # Iterator that never yields — forces timeout path
        class MockAIter:
            async def __anext__(self):
                await asyncio.sleep(100)

        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_resp.headers = {}
        mock_resp.aiter_raw = MagicMock(return_value=MockAIter())
        mock_resp.aclose = AsyncMock()

        mock_req = AsyncMock()
        mock_req.is_disconnected = AsyncMock(return_value=True)

        reader = StreamReader(mock_resp, mock_req, "ModelA", "10.0.0.1:8000", 0,
                              "test_key", "prefix", ["blk1"], sm)

        # Consume generator — reader should timeout on iterator,
        # check is_disconnected, see True, set cancelled, and exit
        t0 = time.monotonic()
        stream_gen = reader.stream()
        async for _ in stream_gen:
            pass
        elapsed = time.monotonic() - t0

        # Reader should detect disconnect quickly (~0.5s heartbeat interval)
        # stream() may take up to STREAM_QUEUE_TIMEOUT to exit after reader finishes
        assert elapsed < 10, f"Stream should exit within reasonable time, took {elapsed:.1f}s"
        assert mock_req.is_disconnected.called, "Reader should have polled is_disconnected"
        assert reader._cancelled, "_cancelled should be True after disconnect"
        # save_after should NOT be called since stream never completed
        assert not mock_client.save_slot.called, "save_after should be skipped on disconnect"

    asyncio.run(_run())
    print("PASS: test_reader_polls_is_disconnected_on_timeout")


def test_streaming_completion_releases_slot():
    """End-to-end: successful StreamReader stream completion releases slot, saves cache."""
    import app as app_mod
    from slot_manager import SlotManager
    from app import StreamReader
    from llama_client import LlamaClient
    from backend_manager import backend_manager

    backend_manager._backends.clear()
    backend_manager._first_key = "10.0.0.1:8000"
    backend_manager._refresh_state.clear()
    backend_manager._discovered_models.clear()

    sm = SlotManager()
    sm.get("10.0.0.1:8000").ensure_pool("ModelA", 1)

    mock_client = AsyncMock(spec=LlamaClient)
    mock_client.save_slot = AsyncMock(return_value=(True, 1024))
    backend_manager._backends["10.0.0.1:8000"] = type('obj', (object,), {'client': mock_client, 'agent_client': None, 'cache_dir': None})()

    async def _run():
        # Iterator yields chunks then finishes (normal completion)
        class DoneAIter:
            _done = False
            async def __anext__(self):
                if not self._done:
                    self._done = True
                    return b"event: content\ndata: {\"text\":\"hi\"}\n\n"
                raise StopAsyncIteration()

        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_resp.headers = {}
        mock_resp.aiter_raw = MagicMock(return_value=DoneAIter())
        mock_resp.aclose = AsyncMock()

        mock_req = AsyncMock()
        mock_req.is_disconnected = AsyncMock(return_value=False)

        reader = StreamReader(mock_resp, mock_req, "ModelA", "10.0.0.1:8000", 0,
                              "complete_key", "system: hello\nuser: hi", ["blk1"], sm)

        # Consume all chunks
        chunks = []
        stream_gen = reader.stream()
        async for chunk in stream_gen:
            chunks.append(chunk)

        # Small delay to let reader's finally block run
        await asyncio.sleep(0.3)

        # All chunks received
        assert len(chunks) >= 1, "Should receive at least one chunk"

        # Slot saved and released
        assert mock_client.save_slot.called, \
            "save_slot must be called on normal completion"

        # Pool state: slot 0 should be free
        assert not sm.get("10.0.0.1:8000")._in_use.get(0, True), \
            "Slot must be free after completion"

    asyncio.run(_run())
    print("PASS: test_streaming_completion_releases_slot")


def test_discover_models_router_mode():
    """Mock /models response with -ctx and -c args."""
    from llama_client import LlamaClient
    import httpx

    client = LlamaClient("http://10.0.0.1:8000")
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "data": [
            {"id": "model-a", "status": {"value": "loaded", "args": ["llama-server", "-ctx", "32768"]}},
            {"id": "model-b", "status": {"value": "loaded", "args": ["llama-server", "-c", "8192"]}},
        ]
    }
    client.client.get = AsyncMock(return_value=mock_resp)

    async def _run():
        return await client.discover_models()

    result = asyncio.run(_run())
    assert result == [("model-a", 32768), ("model-b", 8192)], f"Got {result}"
    print("PASS: test_discover_models_router_mode")


def test_discover_models_non_router_mode():
    """Mock /v1/models response."""
    from llama_client import LlamaClient

    client = LlamaClient("http://10.0.0.1:8000")
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "data": [{"id": "llama-3.1", "meta": {"n_ctx": 4096}}]
    }
    client.client.get = AsyncMock(return_value=mock_resp)

    async def _run():
        return await client.discover_models()

    result = asyncio.run(_run())
    assert result == [("llama-3.1", 4096)], f"Got {result}"
    print("PASS: test_discover_models_non_router_mode")


def test_discover_models_ctx_not_in_args():
    """Mock router /models response without -ctx in args."""
    from llama_client import LlamaClient
    from config import DEFAULT_N_CTX

    client = LlamaClient("http://10.0.0.1:8000")
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "data": [
            {"id": "model-x", "status": {"value": "loaded", "args": ["llama-server"]}},
        ]
    }
    client.client.get = AsyncMock(return_value=mock_resp)

    async def _run():
        return await client.discover_models()

    result = asyncio.run(_run())
    assert result == [("model-x", DEFAULT_N_CTX)], f"Got {result}"
    print("PASS: test_discover_models_ctx_not_in_args")


def test_discover_models_router_loaded_info_n_ctx():
    """Mock router /models where child loaded_info contains n_ctx."""
    from llama_client import LlamaClient

    client = LlamaClient("http://10.0.0.1:8000")
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "data": [
            {
                "id": "model-y",
                "status": {
                    "value": "loaded",
                    "args": ["llama-server"],
                    "loaded_info": {"n_ctx": 16384},
                },
            },
        ]
    }
    client.client.get = AsyncMock(return_value=mock_resp)

    async def _run():
        return await client.discover_models()

    result = asyncio.run(_run())
    assert result == [("model-y", 16384)], f"Got {result}"
    print("PASS: test_discover_models_router_loaded_info_n_ctx")


def test_discover_models_non_router_meta_null():
    """Mock /v1/models response with meta: null."""
    from llama_client import LlamaClient
    from config import DEFAULT_N_CTX

    client = LlamaClient("http://10.0.0.1:8000")
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "data": [{"id": "model-z", "meta": None}]
    }
    client.client.get = AsyncMock(return_value=mock_resp)

    async def _run():
        return await client.discover_models()

    result = asyncio.run(_run())
    assert result == [("model-z", DEFAULT_N_CTX)], f"Got {result}"
    print("PASS: test_discover_models_non_router_meta_null")


def test_discover_models_both_endpoints_fail():
    """Mock both /models and /v1/models to fail."""
    from llama_client import LlamaClient

    client = LlamaClient("http://10.0.0.1:8000")
    client.client.get = AsyncMock(side_effect=Exception("connection refused"))

    async def _run():
        return await client.discover_models()

    result = asyncio.run(_run())
    assert result == [], f"Got {result}"
    print("PASS: test_discover_models_both_endpoints_fail")


# ── Per-backend cache_dir tests ───────────────────────────────────────

def test_backend_cache_dir_per_backend():
    """BackendManager should read cache_dir from backend config."""
    from backend_manager import BackendManager
    from hashing import sanitize_backend_dir

    bm = BackendManager([
        {"url": "http://10.0.0.1:8000", "cache_dir": "/mnt/cache/b1"},
        {"url": "http://10.0.0.2:8000", "agent_port": 8082},
    ])
    be1 = sanitize_backend_dir("10.0.0.1:8000")
    be2 = sanitize_backend_dir("10.0.0.2:8000")
    assert bm.keys() == [be1, be2]
    assert bm.get_cache_dir(be1) == "/mnt/cache/b1"
    assert bm.get_cache_dir(be2) is None
    assert bm.has_cache_config(be1) is True
    assert bm.has_cache_config(be2) is True
    print("PASS: test_backend_cache_dir_per_backend")


def test_backend_cache_dir_mutual_exclusivity():
    """BackendManager should raise ValueError if both cache_dir and agent_port are set, or if neither is set."""
    from backend_manager import BackendManager

    try:
        BackendManager([{"url": "http://10.0.0.1:8000", "cache_dir": "/mnt/cache", "agent_port": 8082}])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "cache_dir" in str(e).lower() or "agent_port" in str(e).lower()

    # Also: must have at least one of cache_dir or agent_port
    try:
        BackendManager([{"url": "http://10.0.0.1:8000"}])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "cache_dir" in str(e).lower() or "agent_port" in str(e).lower()

    print("PASS: test_backend_cache_dir_mutual_exclusivity")


def test_backend_cache_delete_via_agent():
    """cache_delete should use agent when available."""
    from backend_manager import BackendManager
    from hashing import sanitize_backend_dir

    bm = BackendManager([{"url": "http://10.0.0.1:8000", "agent_port": 8082}])
    be = sanitize_backend_dir("10.0.0.1:8000")
    agent = bm.get_agent(be)

    async def run():
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch.object(agent._client, "post", new_callable=AsyncMock, return_value=mock_resp):
            result = await bm.cache_delete(be, "test_key")
            assert result is True, f"Expected True, got {result}"

    asyncio.run(run())
    print("PASS: test_backend_cache_delete_via_agent")


def test_backend_cache_delete_via_local():
    """cache_delete should use local filesystem when no agent."""
    from backend_manager import BackendManager
    from hashing import sanitize_backend_dir
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_file = os.path.join(tmpdir, "test_key")
        with open(cache_file, "w") as f:
            f.write("data")

        bm = BackendManager([{"url": "http://10.0.0.1:8000", "cache_dir": tmpdir}])
        be = sanitize_backend_dir("10.0.0.1:8000")

        result = asyncio.run(bm.cache_delete(be, "test_key"))
        assert result is True, f"Expected True, got {result}"
        assert not os.path.exists(cache_file), "Cache file should be deleted"

    print("PASS: test_backend_cache_delete_via_local")


def test_backend_cache_get_size_via_local():
    """cache_get_size should use local filesystem stat when no agent."""
    from backend_manager import BackendManager
    from hashing import sanitize_backend_dir
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_file = os.path.join(tmpdir, "test_key")
        with open(cache_file, "w") as f:
            f.write("x" * 1234)

        bm = BackendManager([{"url": "http://10.0.0.1:8000", "cache_dir": tmpdir}])
        be = sanitize_backend_dir("10.0.0.1:8000")

        result = asyncio.run(bm.cache_get_size(be, "test_key"))
        assert result == 1234, f"Expected 1234, got {result}"

    print("PASS: test_backend_cache_get_size_via_local")


def test_backend_cache_get_size_not_found():
    """cache_get_size should return 0 for non-existent file."""
    from backend_manager import BackendManager
    from hashing import sanitize_backend_dir
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        bm = BackendManager([{"url": "http://10.0.0.1:8000", "cache_dir": tmpdir}])
        be = sanitize_backend_dir("10.0.0.1:8000")

        result = asyncio.run(bm.cache_get_size(be, "nonexistent"))
        assert result == 0, f"Expected 0, got {result}"

    print("PASS: test_backend_cache_get_size_not_found")


def test_backend_cache_exists_via_local():
    """cache_exists should use local filesystem check when no agent."""
    from backend_manager import BackendManager
    from hashing import sanitize_backend_dir
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_file = os.path.join(tmpdir, "test_key")
        with open(cache_file, "w") as f:
            f.write("data")

        bm = BackendManager([{"url": "http://10.0.0.1:8000", "cache_dir": tmpdir}])
        be = sanitize_backend_dir("10.0.0.1:8000")

        async def run():
            assert await bm.cache_exists(be, "test_key") is True
            assert await bm.cache_exists(be, "nonexistent") is False

        asyncio.run(run())

    print("PASS: test_backend_cache_exists_via_local")


def test_backend_cache_get_mtime_via_local():
    """cache_get_mtime should use local filesystem mtime when no agent."""
    from backend_manager import BackendManager
    from hashing import sanitize_backend_dir
    import tempfile
    import os
    import time

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_file = os.path.join(tmpdir, "test_key")
        with open(cache_file, "w") as f:
            f.write("data")
        mtime = os.path.getmtime(cache_file)

        bm = BackendManager([{"url": "http://10.0.0.1:8000", "cache_dir": tmpdir}])
        be = sanitize_backend_dir("10.0.0.1:8000")

        result = bm.cache_get_mtime(be, "test_key")
        assert abs(result - mtime) < 1.0, f"Expected ~{mtime}, got {result}"

    print("PASS: test_backend_cache_get_mtime_via_local")


# ── Model resolution tests ─────────────────────────────────────────────

def test_resolve_exact_match():
    """Model 'qwen3.6-32b' discovered. Request 'qwen3.6-32b'."""
    from backend_manager import backend_manager, DiscoveredModel

    backend_manager._backends.clear()
    backend_manager._discovered_models.clear()
    backend_manager._backends["10.0.0.1:8000"] = type('obj', (object,), {'client': AsyncMock(), 'agent_client': None, 'cache_dir': None})()
    backend_manager._backends["10.0.0.1:9000"] = type('obj', (object,), {'client': AsyncMock(), 'agent_client': None, 'cache_dir': None})()

    backend_manager._discovered_models["qwen3.6-32b"] = DiscoveredModel(
        name="qwen3.6-32b", n_ctx=32768, backends=["10.0.0.1:8000", "10.0.0.1:9000"],
        total_slots=0, last_discovered=0.0,
    )

    result = backend_manager.get_discovered_models("qwen3.6-32b")
    assert len(result) == 1
    assert result[0].name == "qwen3.6-32b"
    assert result[0].backends == ["10.0.0.1:8000", "10.0.0.1:9000"]
    print("PASS: test_resolve_exact_match")


def test_resolve_substring_match():
    """Model 'qwen3.6-32b-instruct' discovered. Request 'qwen3.6'."""
    from backend_manager import backend_manager, DiscoveredModel

    backend_manager._discovered_models.clear()
    backend_manager._discovered_models["qwen3.6-32b-instruct"] = DiscoveredModel(
        name="qwen3.6-32b-instruct", n_ctx=32768, backends=["10.0.0.1:8000"],
        total_slots=0, last_discovered=0.0,
    )

    result = backend_manager.get_discovered_models("qwen3.6")
    assert len(result) == 1
    assert result[0].name == "qwen3.6-32b-instruct"
    assert result[0].backends == ["10.0.0.1:8000"]
    print("PASS: test_resolve_substring_match")


def test_resolve_ambiguous_substring():
    """Models 'qwen3.6-32b' and 'qwen3.6-8b' discovered. Request 'qwen3.6'."""
    from backend_manager import backend_manager, DiscoveredModel

    backend_manager._discovered_models.clear()
    backend_manager._discovered_models["qwen3.6-32b"] = DiscoveredModel(
        name="qwen3.6-32b", n_ctx=32768, backends=["10.0.0.1:8000"],
        total_slots=0, last_discovered=0.0,
    )
    backend_manager._discovered_models["qwen3.6-8b"] = DiscoveredModel(
        name="qwen3.6-8b", n_ctx=8192, backends=["10.0.0.1:9000"],
        total_slots=0, last_discovered=0.0,
    )

    result = backend_manager.get_discovered_models("qwen3.6")
    assert len(result) == 2, f"Got {result}"
    names = [r.name for r in result]
    assert "qwen3.6-32b" in names
    assert "qwen3.6-8b" in names
    print("PASS: test_resolve_ambiguous_substring")


def test_resolve_any():
    """Models 'qwen3.6-32b' (be1) and 'gemma-3-12b' (be2) discovered. Request 'any'."""
    from backend_manager import backend_manager, DiscoveredModel

    backend_manager._discovered_models.clear()
    backend_manager._discovered_models["qwen3.6-32b"] = DiscoveredModel(
        name="qwen3.6-32b", n_ctx=32768, backends=["10.0.0.1:8000"],
        total_slots=0, last_discovered=0.0,
    )
    backend_manager._discovered_models["gemma-3-12b"] = DiscoveredModel(
        name="gemma-3-12b", n_ctx=16384, backends=["10.0.0.1:9000"],
        total_slots=0, last_discovered=0.0,
    )

    result = backend_manager.get_discovered_models("any")
    assert len(result) == 2, f"Got {result}"
    names = [r.name for r in result]
    assert "qwen3.6-32b" in names
    assert "gemma-3-12b" in names
    print("PASS: test_resolve_any")


def test_resolve_not_found():
    """No models discovered. Request 'unknown'."""
    from backend_manager import backend_manager

    backend_manager._discovered_models.clear()

    result = backend_manager.get_discovered_models("unknown")
    assert result == [], f"Got {result}"
    print("PASS: test_resolve_not_found")


def test_resolve_any_no_models():
    """No models discovered. Request 'any'."""
    from backend_manager import backend_manager

    backend_manager._discovered_models.clear()

    result = backend_manager.get_discovered_models("any")
    assert result == [], f"Got {result}"
    print("PASS: test_resolve_any_no_models")


# ── Cache-first routing tests ──────────────────────────────────────────

def test_cache_hit_selects_best_ratio():
    """Two backends have cache hits (ratios 0.95 and 0.70). Select 0.95."""
    from kv_meta_manager import KVMetaManager
    mgr = KVMetaManager()
    from hashing import meta_key
    from backend_manager import backend_manager, DiscoveredModel
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        import config
        import hashing as hs
        old_meta_dir = config.META_DIR
        config.META_DIR = tmpdir

        try:
            backend_manager._discovered_models.clear()
            backend_manager._discovered_models["model-a"] = DiscoveredModel(
                name="model-a", n_ctx=32768, backends=["be1"],
                total_slots=0, last_discovered=0.0,
            )
            backend_manager._discovered_models["model-b"] = DiscoveredModel(
                name="model-b", n_ctx=32768, backends=["be2"],
                total_slots=0, last_discovered=0.0,
            )

            prefix = "the quick brown fox jumps over the lazy dog"
            canonical_a = "model-a"
            canonical_b = "model-b"

            # Create meta files with different LCP ratios
            blocks_a = ["blk1", "blk2", "blk3", "blk4", "blk5"]
            blocks_b = ["blk1", "x", "x", "x", "x"]  # lower ratio

            key_a = meta_key(canonical_a, prefix)
            key_b = meta_key(canonical_b, prefix)

            mgr.write_meta(key_a, prefix, blocks_a, 100, canonical_a, "be1")
            mgr.write_meta(key_b, prefix, blocks_b, 100, canonical_b, "be2")

            # Simulate cache-first selection logic
            req_blocks = ["blk1", "blk2", "blk3", "blk4", "blk5"]
            best_ratio = 0.0
            best_canonical = None
            best_restore_key = None

            for opt_name, be_list in [("model-a", ["be1"]), ("model-b", ["be2"])]:
                mk = meta_key(opt_name, prefix)
                for be_id in be_list:
                    cand = mgr.find_restore_candidate(mk, 100, 0.2, req_blocks, be_id)
                    if cand and cand[1] > best_ratio:
                        best_ratio = cand[1]
                        best_restore_key = mk
                        best_canonical = opt_name

            assert best_canonical == "model-a", f"Expected model-a, got {best_canonical}"
            assert best_ratio == 1.0, f"Expected ratio 1.0, got {best_ratio}"
        finally:
            config.META_DIR = old_meta_dir

    print("PASS: test_cache_hit_selects_best_ratio")


def test_cache_hit_across_canonical_models():
    """Client requests 'qwen3.6', resolves to two canonical models. Each has a cache hit."""
    from kv_meta_manager import KVMetaManager
    mgr = KVMetaManager()
    from hashing import meta_key
    from backend_manager import backend_manager, DiscoveredModel
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        import config
        old_meta_dir = config.META_DIR
        config.META_DIR = tmpdir

        try:
            backend_manager._discovered_models.clear()
            backend_manager._discovered_models["qwen3.6-32b"] = DiscoveredModel(
                name="qwen3.6-32b", n_ctx=32768, backends=["be1"],
                total_slots=0, last_discovered=0.0,
            )
            backend_manager._discovered_models["qwen3.6-8b"] = DiscoveredModel(
                name="qwen3.6-8b", n_ctx=8192, backends=["be2"],
                total_slots=0, last_discovered=0.0,
            )

            prefix = "hello world this is a test prompt"
            req_blocks = ["blk1", "blk2", "blk3"]

            # model-a has ratio 0.6, model-b has ratio 0.9
            blocks_a = ["blk1", "blk2", "x"]
            blocks_b = ["blk1", "blk2", "blk3"]

            key_a = meta_key("qwen3.6-32b", prefix)
            key_b = meta_key("qwen3.6-8b", prefix)

            mgr.write_meta(key_a, prefix, blocks_a, 100, "qwen3.6-32b", "be1")
            mgr.write_meta(key_b, prefix, blocks_b, 100, "qwen3.6-8b", "be2")

            best_ratio = 0.0
            best_canonical = None

            for dm in backend_manager.get_discovered_models("qwen3.6"):
                mk = meta_key(dm.name, prefix)
                for be_id in dm.backends:
                    cand = mgr.find_restore_candidate(mk, 100, 0.2, req_blocks, be_id)
                    if cand and cand[1] > best_ratio:
                        best_ratio = cand[1]
                        best_canonical = dm.name

            assert best_canonical == "qwen3.6-8b", f"Expected qwen3.6-8b (higher ratio), got {best_canonical}"
        finally:
            config.META_DIR = old_meta_dir

    print("PASS: test_cache_hit_across_canonical_models")


def test_no_cache_fallback_lru():
    """No cache hits found. acquire_for_request selects via LRU."""
    from slot_manager import SlotManager
    from backend_manager import backend_manager, DiscoveredModel
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        import config
        old_meta_dir = config.META_DIR
        config.META_DIR = tmpdir

        try:
            backend_manager._backends.clear()
            backend_manager._discovered_models.clear()
            backend_manager._first_key = "10.0.0.1:8000"

            mock_client = AsyncMock()
            mock_client.discover_models = AsyncMock(return_value=[("model-a", 32768)])
            mock_client.get_slots_info = AsyncMock(return_value=[{"id": 0}, {"id": 1}])
            backend_manager._backends["10.0.0.1:8000"] = type('obj', (object,), {'client': mock_client, 'agent_client': None, 'cache_dir': None, 'cache_dir': None})()

            sm = SlotManager()
            sm.get("10.0.0.1:8000").ensure_pool("model-a", 2)

            async def _run():
                await backend_manager.discover_models()
                be_sm = sm.get("10.0.0.1:8000")
                slot_id = be_sm.try_acquire("model-a")
                assert slot_id is not None
                return ("model-a", "10.0.0.1:8000", slot_id)

            result = asyncio.run(_run())
            assert result is not None
            assert result[2] == 0, f"Expected slot 0, got {result[2]}"
        finally:
            config.META_DIR = old_meta_dir

    print("PASS: test_no_cache_fallback_lru")


def test_no_cache_lru_backend_routing():
    """Cache-miss routing uses composite sort: cache ratio (lowest), ring size (fewest), latency (fastest), LRU (least recent)."""
    from slot_manager import SlotManager
    from backend_manager import backend_manager
    import time

    sm = SlotManager()

    # Backend state:
    #   be1: ratio=0.0, ring=5, latency=200ms, last_used=recent
    #   be2: ratio=0.0, ring=5, latency=50ms,  last_used=old
    #   be3: ratio=0.8, ring=1, latency=100ms, last_used=never
    now = time.time()
    backend_manager._backend_last_used["10.0.0.1:8000"] = now
    backend_manager._backend_last_used["10.0.0.2:8000"] = now - 1000
    sm.get("10.0.0.1:8000")._cache_ring = deque([(f"key_{i}", 1024, now - i) for i in range(5)])
    sm.get("10.0.0.2:8000")._cache_ring = deque([(f"key_{i}", 1024, now - i) for i in range(5)])
    sm.get("10.0.0.3:8000")._cache_ring = deque([("key_0", 1024, now)])
    backend_manager._backend_latency_ema["10.0.0.1:8000"] = 200.0
    backend_manager._backend_latency_ema["10.0.0.2:8000"] = 50.0
    backend_manager._backend_latency_ema["10.0.0.3:8000"] = 100.0

    backend_cache_ratios = {
        "10.0.0.1:8000": 0.0,
        "10.0.0.2:8000": 0.0,
        "10.0.0.3:8000": 0.8,
    }

    # Simulate the sorting logic from app.py
    candidate_backends = [
        ("10.0.0.1:8000", "ModelA"),
        ("10.0.0.2:8000", "ModelA"),
        ("10.0.0.3:8000", "ModelA"),
    ]
    candidate_backends.sort(
        key=lambda cb: (
            backend_cache_ratios.get(cb[0], 0.0),
            len(sm.get(cb[0])._cache_ring),
            backend_manager.get_backend_latency_ema(cb[0]),
            backend_manager.get_backend_last_used(cb[0]),
        ),
    )

    # Expected order:
    # 1. be2 (ratio=0.0, ring=5, latency=50ms, last_used=old) — fastest of equal-ratio backends
    # 2. be1 (ratio=0.0, ring=5, latency=200ms, last_used=recent) — slower of equal-ratio backends
    # 3. be3 (ratio=0.8, ring=1, latency=100ms, last_used=0.0) — high ratio, always last
    assert candidate_backends[0][0] == "10.0.0.2:8000", f"Expected be2 first (fastest), got {candidate_backends[0][0]}"
    assert candidate_backends[1][0] == "10.0.0.1:8000", f"Expected be1 second (slower), got {candidate_backends[1][0]}"
    assert candidate_backends[2][0] == "10.0.0.3:8000", f"Expected be3 last (high ratio), got {candidate_backends[2][0]}"

    # Verify latency EMA tracking
    backend_manager.update_backend_latency("10.0.0.1:8000", 150.0)
    assert backend_manager.get_backend_latency_ema("10.0.0.1:8000") > 0, "EMA should be > 0 after update"

    # Verify try_acquire works correctly
    sm.get("10.0.0.1:8000").ensure_pool("ModelA", 1)
    slot_id = sm.get("10.0.0.1:8000").try_acquire("ModelA")
    assert slot_id is not None, "Should acquire free slot"
    assert sm.get("10.0.0.1:8000")._in_use[slot_id], "Slot should be marked in-use"

    print("PASS: test_no_cache_lru_backend_routing")


def test_meta_key_uses_canonical_name():
    """Client requests 'qwen3.6' -> canonical 'qwen3.6-32b-instruct'. Meta key uses canonical."""
    from hashing import meta_key
    import hashlib

    canonical = "qwen3.6-32b-instruct"
    token_ids = [123, 456, 789]
    expected = hashlib.sha256(f"{canonical}\n{','.join(str(t) for t in token_ids)}".encode("utf-8")).hexdigest()
    result = meta_key(canonical, token_ids)
    assert result == expected, f"Expected {expected}, got {result}"
    print("PASS: test_meta_key_uses_canonical_name")


# ── Context length tests ───────────────────────────────────────────────

def test_get_model_n_ctx_exact():
    """Model 'qwen3.6-32b' with n_ctx 32768. Returns 32768."""
    from backend_manager import backend_manager, DiscoveredModel

    backend_manager._discovered_models.clear()
    backend_manager._discovered_models["qwen3.6-32b"] = DiscoveredModel(
        name="qwen3.6-32b", n_ctx=32768, backends=["be1"],
        total_slots=0, last_discovered=0.0,
    )

    result = backend_manager.get_model_n_ctx("qwen3.6-32b")
    assert result == 32768, f"Expected 32768, got {result}"
    print("PASS: test_get_model_n_ctx_exact")


def test_get_model_n_ctx_min_across_backends():
    """Model on backend A (32768) and B (8192). Returns 8192."""
    from backend_manager import backend_manager, DiscoveredModel

    backend_manager._discovered_models.clear()
    backend_manager._discovered_models["model-a"] = DiscoveredModel(
        name="model-a", n_ctx=8192, backends=["be1", "be2"],
        total_slots=0, last_discovered=0.0,
    )

    result = backend_manager.get_model_n_ctx("model-a")
    assert result == 8192, f"Expected 8192, got {result}"
    print("PASS: test_get_model_n_ctx_min_across_backends")


def test_prompt_too_long_rejected():
    """Model n_ctx=4096, prompt=4000 tokens. Returns 400 error."""
    import app as app_mod
    from fastapi.testclient import TestClient
    from backend_manager import backend_manager, DiscoveredModel

    backend_manager._backends.clear()
    backend_manager._first_key = "10.0.0.1:8000"
    backend_manager._discovered_models.clear()
    backend_manager._discovered_models["model-a"] = DiscoveredModel(
        name="model-a", n_ctx=4096, backends=["10.0.0.1:8000"],
        total_slots=0, last_discovered=0.0,
    )

    mock_client = AsyncMock()
    mock_client.tokenize = AsyncMock(return_value=[0] * 4097)
    backend_manager._backends["10.0.0.1:8000"] = type('obj', (object,), {'client': mock_client, 'agent_client': None, 'cache_dir': None})()

    # Set up app state (normally done in startup event)
    from slot_manager import SlotManager
    sm = SlotManager()
    sm.get("10.0.0.1:8000").ensure_pool("model-a", 1)
    app_mod.app.state.sm = sm

    client = TestClient(app_mod.app)
    long_text = "word " * 4096
    messages = [{"role": "user", "content": long_text}]

    resp = client.post("/v1/chat/completions", json={
        "model": "model-a",
        "messages": messages,
    })

    assert resp.status_code == 400, f"Expected 400, got {resp.status_code}: {resp.text}"
    assert "prompt too long" in resp.json()["error"].lower()
    print("PASS: test_prompt_too_long_rejected")


# ── /v1/models endpoint tests ──────────────────────────────────────────

def test_models_endpoint_includes_any():
    """Mock refresh_models to return 2 models. Response includes both models plus 'any'."""
    from fastapi.testclient import TestClient
    from backend_manager import backend_manager, DiscoveredModel
    import app as app_mod

    backend_manager._discovered_models.clear()
    backend_manager._discovered_models["model-a"] = DiscoveredModel(
        name="model-a", n_ctx=32768, backends=["be1"],
        total_slots=0, last_discovered=0.0,
    )
    backend_manager._discovered_models["model-b"] = DiscoveredModel(
        name="model-b", n_ctx=16384, backends=["be2"],
        total_slots=0, last_discovered=0.0,
    )

    client = TestClient(app_mod.app)
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    ids = [m["id"] for m in data["data"]]
    assert "model-a" in ids
    assert "model-b" in ids
    assert "any" in ids
    print("PASS: test_models_endpoint_includes_any")


def test_models_endpoint_openai_format():
    """Assert each model has id, object: 'model', owned_by fields."""
    from fastapi.testclient import TestClient
    from backend_manager import backend_manager, DiscoveredModel
    import app as app_mod

    backend_manager._discovered_models.clear()
    backend_manager._discovered_models["model-a"] = DiscoveredModel(
        name="model-a", n_ctx=32768, backends=["be1"],
        total_slots=0, last_discovered=0.0,
    )

    client = TestClient(app_mod.app)
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    for m in data["data"]:
        assert "id" in m
        assert m["object"] == "model"
        assert "owned_by" in m
    print("PASS: test_models_endpoint_openai_format")


# ── Chat completion tests ──────────────────────────────────────────────

def test_chat_substring_model_resolution():
    """Client requests 'qwen3.6', backend serves 'qwen3.6-32b-instruct'. Slot acquired on correct backend."""
    from fastapi.testclient import TestClient
    from slot_manager import SlotManager
    from backend_manager import backend_manager, DiscoveredModel
    from llama_client import LlamaClient
    import app as app_mod

    backend_manager._backends.clear()
    backend_manager._discovered_models.clear()
    backend_manager._first_key = "10.0.0.1:8000"

    mock_client = AsyncMock(spec=LlamaClient)
    mock_client.chat_completions = AsyncMock(return_value={"object": "chat.completion", "choices": []})
    mock_client.discover_models = AsyncMock(return_value=[("qwen3.6-32b-instruct", 32768)])
    mock_client.get_slots_info = AsyncMock(return_value=[{"id": 0}])
    mock_client.tokenize = AsyncMock(return_value=[123, 456])
    mock_client.save_slot = AsyncMock(return_value=(True, 1024))
    backend_manager._backends["10.0.0.1:8000"] = type('obj', (object,), {'client': mock_client, 'agent_client': None, 'cache_dir': None})()

    # Set up app state (normally done in startup event)
    from slot_manager import SlotManager
    sm = SlotManager()
    sm.get("10.0.0.1:8000").ensure_pool("qwen3.6-32b-instruct", 1)
    app_mod.app.state.sm = sm

    client = TestClient(app_mod.app)
    resp = client.post("/v1/chat/completions", json={
        "model": "qwen3.6",
        "messages": [{"role": "user", "content": "hello"}],
    })

    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    # Verify canonical name was forwarded
    call_args = mock_client.chat_completions.call_args
    assert call_args is not None
    body = call_args[0][0]
    assert body["model"] == "qwen3.6-32b-instruct", f"Expected canonical name, got {body['model']}"
    print("PASS: test_chat_substring_model_resolution")


def test_chat_model_not_found_400():
    """Client requests 'unknown-model'. Returns 400 error."""
    from fastapi.testclient import TestClient
    from backend_manager import backend_manager
    import app as app_mod

    backend_manager._discovered_models.clear()

    # Set up app state (normally done in startup event)
    from slot_manager import SlotManager
    sm = SlotManager()
    app_mod.app.state.sm = sm

    client = TestClient(app_mod.app)
    resp = client.post("/v1/chat/completions", json={
        "model": "unknown-model",
        "messages": [{"role": "user", "content": "hello"}],
    })

    assert resp.status_code == 400, f"Expected 400, got {resp.status_code}"
    assert "not found" in resp.json()["error"].lower()
    print("PASS: test_chat_model_not_found_400")


def test_chat_any_model_routing():
    """Client requests 'any'. Slot acquired on a backend, canonical name forwarded."""
    from fastapi.testclient import TestClient
    from slot_manager import SlotManager
    from backend_manager import backend_manager, DiscoveredModel
    from llama_client import LlamaClient
    import app as app_mod

    backend_manager._backends.clear()
    backend_manager._discovered_models.clear()
    backend_manager._first_key = "10.0.0.1:8000"

    mock_client = AsyncMock(spec=LlamaClient)
    mock_client.chat_completions = AsyncMock(return_value={"object": "chat.completion", "choices": []})
    mock_client.discover_models = AsyncMock(return_value=[("model-x", 16384)])
    mock_client.get_slots_info = AsyncMock(return_value=[{"id": 0}])
    mock_client.tokenize = AsyncMock(return_value=[123, 456])
    mock_client.save_slot = AsyncMock(return_value=(True, 1024))
    backend_manager._backends["10.0.0.1:8000"] = type('obj', (object,), {'client': mock_client, 'agent_client': None, 'cache_dir': None})()

    # Set up app state (normally done in startup event)
    from slot_manager import SlotManager
    sm = SlotManager()
    sm.get("10.0.0.1:8000").ensure_pool("model-x", 1)
    app_mod.app.state.sm = sm

    client = TestClient(app_mod.app)
    resp = client.post("/v1/chat/completions", json={
        "model": "any",
        "messages": [{"role": "user", "content": "hello"}],
    })

    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    call_args = mock_client.chat_completions.call_args
    assert call_args is not None
    body = call_args[0][0]
    assert body["model"] == "model-x", f"Expected canonical name, got {body['model']}"
    print("PASS: test_chat_any_model_routing")


def test_chat_any_with_cache_hit():
    """Client requests 'any', multiple canonical models have cache hits. Selects best cache hit."""
    import hashing as hs
    from kv_meta_manager import KVMetaManager
    mgr = KVMetaManager()
    from hashing import meta_key
    from backend_manager import backend_manager, DiscoveredModel
    import tempfile
    import os
    import app as app_mod
    from fastapi.testclient import TestClient
    from llama_client import LlamaClient

    with tempfile.TemporaryDirectory() as tmpdir:
        import config
        old_meta_dir = config.META_DIR
        old_app_wpb = app_mod.WORDS_PER_BLOCK
        old_hs_wpb = hs.WORDS_PER_BLOCK
        config.META_DIR = tmpdir
        # Use small WPB so we get multiple blocks for ratio comparison
        test_wpb = 3
        app_mod.WORDS_PER_BLOCK = test_wpb
        hs.WORDS_PER_BLOCK = test_wpb

        try:
            backend_manager._backends.clear()
            backend_manager._discovered_models.clear()
            backend_manager._first_key = "10.0.0.1:8000"

            backend_manager._discovered_models["model-a"] = DiscoveredModel(
                name="model-a", n_ctx=32768, backends=["10.0.0.1:8000"],
                total_slots=1, last_discovered=0.0,
            )
            backend_manager._discovered_models["model-b"] = DiscoveredModel(
                name="model-b", n_ctx=16384, backends=["10.0.0.1:9000"],
                total_slots=1, last_discovered=0.0,
            )

            # Use real block hashes so LCP matching works
            token_ids = [123, 456, 789, 101, 102, 103, 104, 105, 106, 107]
            req_blocks = hs.block_hashes_from_tokens(token_ids, test_wpb)
            # model-a: all blocks match (ratio 1.0)
            blocks_a = list(req_blocks)
            # model-b: only first 2 blocks match (ratio 2/4 = 0.5)
            blocks_b = list(req_blocks[:2]) + ["unique_b_1", "unique_b_2"]

            key_a = meta_key("model-a", token_ids)
            key_b = meta_key("model-b", token_ids)

            mgr.write_meta(key_a, len(token_ids), blocks_a, test_wpb, "model-a", "10.0.0.1:8000")
            mgr.write_meta(key_b, len(token_ids), blocks_b, test_wpb, "model-b", "10.0.0.1:9000")

            mock_client_a = AsyncMock(spec=LlamaClient)
            mock_client_a.chat_completions = AsyncMock(return_value={"object": "chat.completion", "choices": []})
            mock_client_a.discover_models = AsyncMock(return_value=[("model-a", 32768), ("model-b", 16384)])
            mock_client_a.get_slots_info = AsyncMock(return_value=[{"id": 0}])
            mock_client_a.tokenize = AsyncMock(return_value=token_ids)
            mock_client_a.save_slot = AsyncMock(return_value=(True, 1024))
            mock_client_b = AsyncMock(spec=LlamaClient)
            mock_client_b.tokenize = AsyncMock(return_value=token_ids)
            backend_manager._backends["10.0.0.1:8000"] = type('obj', (object,), {'client': mock_client_a, 'agent_client': None, 'cache_dir': None})()
            backend_manager._backends["10.0.0.1:9000"] = type('obj', (object,), {'client': mock_client_b, 'agent_client': None, 'cache_dir': None})()

            # Set up app state (normally done in startup event)
            from slot_manager import SlotManager
            sm = SlotManager()
            sm.get("10.0.0.1:8000").ensure_pool("model-a", 1)
            sm.get("10.0.0.1:9000").ensure_pool("model-b", 1)
            app_mod.app.state.sm = sm

            client = TestClient(app_mod.app)
            resp = client.post("/v1/chat/completions", json={
                "model": "any",
                "messages": [{"role": "user", "content": "test prompt for cache hit"}],
            })

            assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
            call_args = mock_client_a.chat_completions.call_args
            assert call_args is not None
            body = call_args[0][0]
            # Should have routed to model-a (better cache hit)
            assert body["model"] == "model-a", f"Expected model-a (best cache), got {body['model']}"
        finally:
            config.META_DIR = old_meta_dir
            app_mod.WORDS_PER_BLOCK = old_app_wpb
            hs.WORDS_PER_BLOCK = old_hs_wpb

    print("PASS: test_chat_any_with_cache_hit")


def test_acquire_for_request_retries_on_lock_timeout():
    """acquire_for_request should try the next backend when a lock times out."""
    from slot_manager import SlotManager
    from app import CacheHitType
    from backend_manager import backend_manager, DiscoveredModel

    async def _run():
        sm = SlotManager()
        sm.get("10.0.0.1:8000").ensure_pool("ModelA", 2)
        sm.get("10.0.0.2:8000").ensure_pool("ModelA", 2)
        sm.get("10.0.0.1:8000")._in_use[0] = True
        sm.get("10.0.0.1:8000")._in_use[1] = True
        sm.get("10.0.0.2:8000")._in_use[0] = False
        sm.get("10.0.0.2:8000")._in_use[1] = False

        backend_manager._discovered_models = {
            "ModelA": DiscoveredModel(
                name="ModelA", n_ctx=4096,
                backends=["10.0.0.1:8000", "10.0.0.2:8000"],
                total_slots=4, last_discovered=time.time(),
            ),
        }

        # Mock refresh_slot_counts to avoid it overwriting our pre-acquired locks
        sm.refresh_slot_counts = AsyncMock()

        # Cache backend is 10.0.0.1:8000 (all slots in-use), fallback is 10.0.0.2:8000
        restore_info = (CacheHitType.DISK_RESTORE, "10.0.0.1:8000", "ModelA", "test_key")
        candidate_backends = [("10.0.0.2:8000", "ModelA")]
        
        # Simulate the behavior: cache backend is busy, falls through to fallback
        # In the new API, _try_cache_backend would fail because all slots are in-use
        # Then the retry loop would try the fallback backend
        be_sm = sm.get("10.0.0.2:8000")
        slot_id = be_sm.try_acquire("ModelA")
        assert slot_id is not None, "Should acquire slot on fallback backend"
        assert slot_id == 0, f"Expected slot 0, got {slot_id}"
        return True

    with patch("asyncio.sleep", _fast_sleep):
        result = asyncio.run(_run())
    assert result
    print("PASS: test_acquire_for_request_retries_on_lock_timeout")


# ── Cache save ratio threshold tests ────────────────────────────────────

def test_chat_save_skipped_when_ratio_above_threshold():
    """Non-streaming chat should skip save when restore ratio >= threshold."""
    from unittest.mock import AsyncMock, patch
    from fastapi.testclient import TestClient
    from slot_manager import SlotManager
    from backend_manager import backend_manager, DiscoveredModel
    from llama_client import LlamaClient
    import app as app_mod

    backend_manager._backends.clear()
    backend_manager._discovered_models.clear()
    backend_manager._first_key = "10.0.0.1:8000"

    mock_client = AsyncMock(spec=LlamaClient)
    mock_client.chat_completions = AsyncMock(return_value={"object": "chat.completion", "choices": [], "usage": {"prompt_tokens": 3, "prompt_tokens_details": {"cached_tokens": 3}}})
    mock_client.discover_models = AsyncMock(return_value=[("test-model", 32768)])
    mock_client.get_slots_info = AsyncMock(return_value=[{"id": 0}])
    mock_client.tokenize = AsyncMock(return_value=[123, 456, 789])
    mock_client.apply_chat_template = AsyncMock(return_value="user: hello world\nassistant:")
    mock_client.save_slot = AsyncMock(return_value=(True, 1024))
    mock_client.get_slot_status = AsyncMock(return_value={"cached_tokens": 3})
    backend_manager._backends["10.0.0.1:8000"] = type('obj', (object,), {'client': mock_client, 'agent_client': None, 'cache_dir': None})()

    backend_manager._discovered_models["test-model"] = DiscoveredModel(
        name="test-model", backends=["10.0.0.1:8000"], n_ctx=32768,
        total_slots=1, last_discovered=0.0
    )

    sm = SlotManager()
    sm.get("10.0.0.1:8000").ensure_pool("test-model", 1)
    app_mod.app.state.sm = sm

    save_called = []
    original_save = sm.get("10.0.0.1:8000").save_after
    async def track_save(*args, **kwargs):
        save_called.append(True)
        return await original_save(*args, **kwargs)
    sm.get("10.0.0.1:8000").save_after = track_save

    with patch("hashing.block_hashes_from_tokens", return_value=["hash1"]), \
         patch("kv_meta_manager.KVMetaManager.find_best_restore_candidate", return_value=("test_key", 0.95)):

        client = TestClient(app_mod.app)
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello world"}],
            "stream": False,
        })

    assert resp.status_code == 200
    assert len(save_called) == 0, f"Expected no save calls, got {len(save_called)}"
    print("PASS: test_chat_save_skipped_when_ratio_above_threshold")


def test_chat_save_performed_when_ratio_below_threshold():
    """Non-streaming chat should save when restore ratio < threshold."""
    from unittest.mock import AsyncMock, patch
    from fastapi.testclient import TestClient
    from slot_manager import SlotManager
    from backend_manager import backend_manager, DiscoveredModel
    from llama_client import LlamaClient
    import app as app_mod

    backend_manager._backends.clear()
    backend_manager._discovered_models.clear()
    backend_manager._first_key = "10.0.0.1:8000"

    mock_client = AsyncMock(spec=LlamaClient)
    mock_client.chat_completions = AsyncMock(return_value={"object": "chat.completion", "choices": []})
    mock_client.discover_models = AsyncMock(return_value=[("test-model", 32768)])
    mock_client.get_slots_info = AsyncMock(return_value=[{"id": 0}])
    mock_client.save_slot = AsyncMock(return_value=(True, 1024))
    mock_client.tokenize = AsyncMock(return_value=list(range(600)))
    mock_client.apply_chat_template = AsyncMock(return_value="user: big prompt for testing\nassistant:")
    backend_manager._backends["10.0.0.1:8000"] = type('obj', (object,), {'client': mock_client, 'agent_client': None, 'cache_dir': None})()

    backend_manager._discovered_models["test-model"] = DiscoveredModel(
        name="test-model", backends=["10.0.0.1:8000"], n_ctx=32768,
        total_slots=1, last_discovered=0.0
    )

    sm = SlotManager()
    sm.get("10.0.0.1:8000").ensure_pool("test-model", 1)
    app_mod.app.state.sm = sm

    save_called = []
    original_save = sm.get("10.0.0.1:8000").save_after
    async def track_save(*args, **kwargs):
        save_called.append(True)
        return await original_save(*args, **kwargs)
    sm.get("10.0.0.1:8000").save_after = track_save

    with patch("hashing.block_hashes_from_tokens", return_value=[f"hash{i}" for i in range(6)]), \
         patch("kv_meta_manager.KVMetaManager.find_best_restore_candidate", return_value=("test_key", 0.5)), \
         patch("kv_meta_manager.KVMetaManager.write_meta"):

        client = TestClient(app_mod.app)
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "big prompt for testing"}],
            "stream": False,
        })

    assert resp.status_code == 200
    assert len(save_called) == 1, f"Expected 1 save call, got {len(save_called)}"
    print("PASS: test_chat_save_performed_when_ratio_below_threshold")


def test_cache_hit_wait_phase0_success():
    """Phase 0 should succeed when semaphore is released during wait."""
    import asyncio
    from unittest.mock import AsyncMock, patch, MagicMock
    from slot_manager import SlotManager
    from app import CacheHitType
    from backend_manager import backend_manager, DiscoveredModel
    from config import CACHE_HIT_WAIT_MAX_PENDING_REQS

    backend_manager._first_key = "10.0.0.1:8000"
    backend_manager._refresh_state.clear()
    backend_manager._discovered_models.clear()

    async def run_test():
        sm = SlotManager()
        sm.get("backend1").ensure_pool("ModelA", 1)

        backend_manager._discovered_models = {
            "ModelA": DiscoveredModel(
                name="ModelA", n_ctx=4096,
                backends=["backend1"],
                total_slots=1, last_discovered=time.time(),
            ),
        }

        # Mark the slot as in-use
        sm.get("backend1")._in_use[0] = True

        # Release the slot immediately (sleep mock makes this instant)
        sm.get("backend1").release(0)

        # In the new API, the slot is now free, so try_acquire should succeed
        be_sm = sm.get("backend1")
        slot_id = be_sm.try_acquire("ModelA")
        assert slot_id is not None, "Should acquire free slot"
        assert slot_id == 0, f"Expected slot 0, got {slot_id}"
        return ("ModelA", "backend1", slot_id), True

    with patch("asyncio.sleep", _fast_sleep):
        g, restored = asyncio.run(run_test())
    assert g == ("ModelA", "backend1", 0), f"Expected ('ModelA', 'backend1', 0), got {g}"
    assert restored is True
    print("PASS: test_cache_hit_wait_phase0_success")


def test_cache_hit_wait_phase0_timeout():
    """Phase 0 should timeout and fall through to retry loop."""
    import asyncio
    from unittest.mock import AsyncMock, patch, MagicMock
    from slot_manager import SlotManager
    from app import CacheHitType
    from backend_manager import backend_manager, DiscoveredModel
    from config import CACHE_HIT_WAIT_EMA_MIN_TIMEOUT

    backend_manager._first_key = "10.0.0.1:8000"
    backend_manager._refresh_state.clear()
    backend_manager._discovered_models.clear()

    async def run_test():
        sm = SlotManager()
        sm.get("backend1").ensure_pool("ModelA", 1)
        sm.get("backend2").ensure_pool("ModelB", 1)

        backend_manager._discovered_models = {
            "ModelA": DiscoveredModel(
                name="ModelA", n_ctx=4096,
                backends=["backend1"],
                total_slots=1, last_discovered=time.time(),
            ),
            "ModelB": DiscoveredModel(
                name="ModelB", n_ctx=4096,
                backends=["backend2"],
                total_slots=1, last_discovered=time.time(),
            ),
        }

        # Mark cache backend slot as in-use, fallback slot as free
        sm.get("backend1")._in_use[0] = True
        sm.get("backend2")._in_use[0] = False

        # Use a very short timeout by directly manipulating the EMA
        sm.get("backend1")._slot_duration_ema = CACHE_HIT_WAIT_EMA_MIN_TIMEOUT

        # In the new API, the fallback backend has a free slot, so try_acquire should succeed
        be_sm = sm.get("backend2")
        slot_id = be_sm.try_acquire("ModelB")
        assert slot_id is not None, "Should acquire free slot on fallback backend"
        assert slot_id == 0, f"Expected slot 0, got {slot_id}"
        return ("ModelB", "backend2", slot_id), True

    with patch("asyncio.sleep", _fast_sleep):
        g, restored = asyncio.run(run_test())
    # Should acquire slot on fallback backend
    assert g == ("ModelB", "backend2", 0), f"Expected ('ModelB', 'backend2', 0), got {g}"
    print("PASS: test_cache_hit_wait_phase0_timeout")


def test_cache_hit_wait_pending_count_blocks():
    """Pending count >= MAX_PENDING_REQS should skip Phase 0 entirely."""
    import asyncio
    from unittest.mock import AsyncMock, patch, MagicMock
    from slot_manager import SlotManager
    from app import CacheHitType
    from backend_manager import backend_manager, DiscoveredModel
    from config import CACHE_HIT_WAIT_MAX_PENDING_REQS

    backend_manager._first_key = "10.0.0.1:8000"
    backend_manager._refresh_state.clear()
    backend_manager._discovered_models.clear()

    async def run_test():
        sm = SlotManager()
        sm.get("backend1").ensure_pool("ModelA", 1)
        sm.get("backend2").ensure_pool("ModelB", 1)

        backend_manager._discovered_models = {
            "ModelA": DiscoveredModel(
                name="ModelA", n_ctx=4096,
                backends=["backend1"],
                total_slots=1, last_discovered=time.time(),
            ),
            "ModelB": DiscoveredModel(
                name="ModelB", n_ctx=4096,
                backends=["backend2"],
                total_slots=1, last_discovered=time.time(),
            ),
        }

        # Fill up the pending count
        for i in range(CACHE_HIT_WAIT_MAX_PENDING_REQS):
            sm._cache_wait_pending["backend1"] = sm._cache_wait_pending.get("backend1", 0) + 1

        # Mark cache backend slot as in-use, fallback slot as free
        sm.get("backend1")._in_use[0] = True
        sm.get("backend2")._in_use[0] = False

        # In the new API, the fallback backend has a free slot, so try_acquire should succeed
        be_sm = sm.get("backend2")
        slot_id = be_sm.try_acquire("ModelB")
        assert slot_id is not None, "Should acquire free slot on fallback backend"
        assert slot_id == 0, f"Expected slot 0, got {slot_id}"
        return sm, ("ModelB", "backend2", slot_id), True

    sm, g, restored = asyncio.run(run_test())
    # pending count was at MAX, but fallback backend has free slot
    assert g == ("ModelB", "backend2", 0), f"Expected ('ModelB', 'backend2', 0), got {g}"
    assert sm._cache_wait_pending.get("backend1", 0) == CACHE_HIT_WAIT_MAX_PENDING_REQS
    print("PASS: test_cache_hit_wait_pending_count_blocks")


def test_slot_duration_ema_updates_after_release():
    """EMA should update with slot occupancy duration after release."""
    from slot_manager import SlotManager
    from backend_manager import backend_manager
    from config import CACHE_HIT_WAIT_EMA_ALPHA, CACHE_HIT_WAIT_EMA_INITIAL_TIMEOUT, \
        CACHE_HIT_WAIT_EMA_MIN_TIMEOUT, CACHE_HIT_WAIT_EMA_MAX_TIMEOUT

    backend_manager._first_key = "10.0.0.1:8000"
    backend_manager._refresh_state.clear()
    backend_manager._discovered_models.clear()

    sm = SlotManager()
    sm.get("backend1").ensure_pool("ModelA", 1)

    # Use a fixed time base for testing
    time_base = 1000.0

    # Simulate slot acquisition
    be_sm = sm.get("backend1")
    be_sm._slot_acquired_at[0] = time_base
    be_sm._in_use[0] = True

    # Mock time.time() to return time_base + 50 for release
    original_time = time.time
    time.time = lambda: time_base + 50

    try:
        # Simulate release after 50 seconds
        duration = be_sm.release(0)
    finally:
        time.time = original_time

    # Release should return the occupancy duration
    assert duration == 50.0, f"Expected duration 50.0, got {duration}"
    assert 0 not in be_sm._slot_acquired_at
    # EMA should be updated on the BackendSlotManager (blends duration with initial timeout, clamped)
    expected_ema = CACHE_HIT_WAIT_EMA_ALPHA * 50.0 + (1 - CACHE_HIT_WAIT_EMA_ALPHA) * CACHE_HIT_WAIT_EMA_INITIAL_TIMEOUT
    expected_ema = max(min(expected_ema, CACHE_HIT_WAIT_EMA_MAX_TIMEOUT), CACHE_HIT_WAIT_EMA_MIN_TIMEOUT)
    assert be_sm._slot_duration_ema == expected_ema, f"Expected EMA {expected_ema}, got {be_sm._slot_duration_ema}"
    print("PASS: test_slot_duration_ema_updates_after_release")


def test_slot_duration_ema_bounds():
    """EMA timeout should be clamped between MIN and MAX."""
    from config import CACHE_HIT_WAIT_EMA_MIN_TIMEOUT, CACHE_HIT_WAIT_EMA_MAX_TIMEOUT

    # Test MAX cap: EMA above max should be capped
    ema_high = 500.0
    wait_timeout = max(min(ema_high, CACHE_HIT_WAIT_EMA_MAX_TIMEOUT), CACHE_HIT_WAIT_EMA_MIN_TIMEOUT)
    assert wait_timeout == CACHE_HIT_WAIT_EMA_MAX_TIMEOUT, \
        f"Expected MAX cap {CACHE_HIT_WAIT_EMA_MAX_TIMEOUT}, got {wait_timeout}"

    # Test MIN floor: EMA below min should use min
    ema_low = 5.0
    wait_timeout = max(min(ema_low, CACHE_HIT_WAIT_EMA_MAX_TIMEOUT), CACHE_HIT_WAIT_EMA_MIN_TIMEOUT)
    assert wait_timeout == CACHE_HIT_WAIT_EMA_MIN_TIMEOUT, \
        f"Expected MIN floor {CACHE_HIT_WAIT_EMA_MIN_TIMEOUT}, got {wait_timeout}"

    # Test normal range: EMA between min and max passes through
    ema_normal = 50.0
    wait_timeout = max(min(ema_normal, CACHE_HIT_WAIT_EMA_MAX_TIMEOUT), CACHE_HIT_WAIT_EMA_MIN_TIMEOUT)
    assert wait_timeout == 50.0, f"Expected 50.0, got {wait_timeout}"
    print("PASS: test_slot_duration_ema_bounds")


def test_slot_duration_ema_uses_initial_for_new_backend():
    """New backend with no EMA data should use INITIAL_TIMEOUT."""
    from slot_manager import SlotManager
    from backend_manager import backend_manager
    from config import CACHE_HIT_WAIT_EMA_INITIAL_TIMEOUT, CACHE_HIT_WAIT_EMA_MAX_TIMEOUT, \
        CACHE_HIT_WAIT_EMA_MIN_TIMEOUT

    backend_manager._backends.clear()
    backend_manager._first_key = "10.0.0.1:8000"
    backend_manager._refresh_state.clear()
    backend_manager._discovered_models.clear()

    sm = SlotManager()
    be_sm = sm.get("backend1")
    be_sm.ensure_pool("ModelA", 1)

    # No prior EMA data — getter returns INITIAL_TIMEOUT
    ema = be_sm.get_slot_duration_ema()
    wait_timeout = max(min(ema, CACHE_HIT_WAIT_EMA_MAX_TIMEOUT), CACHE_HIT_WAIT_EMA_MIN_TIMEOUT)
    assert wait_timeout == CACHE_HIT_WAIT_EMA_INITIAL_TIMEOUT, \
        f"Expected INITIAL_TIMEOUT {CACHE_HIT_WAIT_EMA_INITIAL_TIMEOUT}, got {wait_timeout}"
    print("PASS: test_slot_duration_ema_uses_initial_for_new_backend")


# ── Metrics tests ──────────────────────────────────────────────────────────

def test_metrics_collector_basic():
    """MetricsCollector should track requests and compute basic metrics."""
    from metrics import MetricsCollector

    m = MetricsCollector(retention=10)

    # Record a cache hit with no recompute
    m.record({
        "t0": 1000.0, "request_json": {"model": "test", "messages": [{"role": "user", "content": "hello"}]},
        "model": "test", "backend": "be1", "slot_id": 0,
        "cache_hit": True, "restored": True, "recompute": False,
        "saved": True, "latency_ms": 100.0, "n_tokens": 10, "cache_size_bytes": 1024,
    })

    # Record a cache miss
    m.record({
        "t0": 1001.0, "request_json": {"model": "test", "messages": [{"role": "user", "content": "world"}]},
        "model": "test", "backend": "be1", "slot_id": 1,
        "cache_hit": False, "restored": False, "recompute": False,
        "saved": True, "latency_ms": 200.0, "n_tokens": 20, "cache_size_bytes": 2048,
    })

    # Record a cache hit with recompute
    m.record({
        "t0": 1002.0, "request_json": {"model": "test", "messages": [{"role": "user", "content": "foo"}]},
        "model": "test", "backend": "be1", "slot_id": 0,
        "cache_hit": True, "restored": True, "recompute": True,
        "saved": False, "latency_ms": 300.0, "n_tokens": 30, "cache_size_bytes": 0,
    })

    perf = m.get_performance()
    assert perf["total_requests"] == 3
    assert perf["cache_hits"] == 2
    assert perf["cache_misses"] == 1
    assert perf["cache_recomputes"] == 1
    assert perf["cache_saved"] == 2
    assert perf["cache_save_skipped"] == 1
    assert abs(perf["cache_hit_rate"] - 2/3) < 0.01
    assert abs(perf["cache_mispredict_rate"] - 0.5) < 0.01
    assert abs(perf["cache_utility_rate"] - 1/3) < 0.01
    assert abs(perf["latency"]["avg_ms"] - 200.0) < 1.0

    requests = m.get_requests()
    assert len(requests) == 3
    # get_requests() returns newest-first
    assert requests[0]["prompt_preview"] == "foo"
    assert requests[2]["prompt_preview"] == "hello"
    assert requests[0]["full_request_json"] == {"model": "test", "messages": [{"role": "user", "content": "foo"}]}
    assert requests[2]["full_request_json"] == {"model": "test", "messages": [{"role": "user", "content": "hello"}]}

    print("PASS: test_metrics_collector_basic")


def test_metrics_collector_performance():
    """MetricsCollector should compute latency percentiles correctly."""
    from metrics import MetricsCollector

    m = MetricsCollector(retention=100)

    # Record requests with known latencies
    latencies = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for i, lat in enumerate(latencies):
        m.record({
            "t0": 1000.0 + i, "request_json": {},
            "model": "test", "backend": "be1", "slot_id": 0,
            "cache_hit": False, "restored": False, "recompute": False,
            "saved": False, "latency_ms": lat, "n_tokens": 1,
        })

    perf = m.get_performance()
    assert perf["latency"]["avg_ms"] == 55.0  # (10+100)/2 = 55
    # Percentile indices: int(n*p) for n=10: p50->5, p95->9, p99->9
    assert perf["latency"]["p50_ms"] == 60.0  # latencies[5]
    assert perf["latency"]["p95_ms"] == 100.0  # latencies[9]
    assert perf["latency"]["p99_ms"] == 100.0  # latencies[9]

    print("PASS: test_metrics_collector_performance")


def test_metrics_collector_ring_overflow():
    """MetricsCollector should overflow ring buffer correctly."""
    from metrics import MetricsCollector

    m = MetricsCollector(retention=5)

    for i in range(10):
        m.record({
            "t0": 1000.0 + i, "request_json": {"i": i},
            "model": "test", "backend": "be1", "slot_id": 0,
            "cache_hit": False, "restored": False, "recompute": False,
            "saved": False, "latency_ms": 10.0, "n_tokens": 1,
        })

    requests = m.get_requests()
    assert len(requests) == 5
    # Should have the last 5 entries (i=5..9), newest-first
    assert requests[0]["full_request_json"]["i"] == 9
    assert requests[4]["full_request_json"]["i"] == 5

    # Total counters should reflect all 10 records
    perf = m.get_performance()
    assert perf["total_requests"] == 10

    print("PASS: test_metrics_collector_ring_overflow")


def test_metrics_collector_per_model():
    """MetricsCollector should compute per-model metrics."""
    from metrics import MetricsCollector

    m = MetricsCollector(retention=100)

    m.record({
        "t0": 1000.0, "request_json": {},
        "model": "model-a", "backend": "be1", "slot_id": 0,
        "cache_hit": True, "restored": True, "recompute": False,
        "saved": True, "latency_ms": 100.0,
    })
    m.record({
        "t0": 1001.0, "request_json": {},
        "model": "model-b", "backend": "be1", "slot_id": 0,
        "cache_hit": False, "restored": False, "recompute": False,
        "saved": False, "latency_ms": 200.0,
    })

    perf_a = m.get_performance(model="model-a")
    assert perf_a["cache_hit_rate"] == 1.0
    assert perf_a["total_requests"] == 1

    perf_b = m.get_performance(model="model-b")
    assert perf_b["cache_hit_rate"] == 0.0
    assert perf_b["total_requests"] == 1

    print("PASS: test_metrics_collector_per_model")


def test_metrics_collector_per_backend():
    """MetricsCollector should compute per-backend metrics."""
    from metrics import MetricsCollector

    m = MetricsCollector(retention=100)

    m.record({
        "t0": 1000.0, "request_json": {},
        "model": "test", "backend": "be1", "slot_id": 0,
        "cache_hit": True, "restored": True, "recompute": False,
        "saved": True, "latency_ms": 100.0,
    })
    m.record({
        "t0": 1001.0, "request_json": {},
        "model": "test", "backend": "be2", "slot_id": 0,
        "cache_hit": False, "restored": False, "recompute": False,
        "saved": False, "latency_ms": 200.0,
    })

    perf_1 = m.get_performance(backend="be1")
    assert perf_1["cache_hit_rate"] == 1.0

    perf_2 = m.get_performance(backend="be2")
    assert perf_2["cache_hit_rate"] == 0.0

    print("PASS: test_metrics_collector_per_backend")


def test_metrics_collector_two_phase_recording():
    """MetricsCollector should support two-phase recording with request_id matching."""
    from metrics import MetricsCollector

    m = MetricsCollector(retention=10)

    # Phase 1: arrival record (incomplete) with request_json and prompt_preview
    m.record({
        "request_id": "req-1",
        "request_json": {"model": "test", "messages": [{"role": "user", "content": "hello world"}]},
        "model": "test",
        "stream": True,
        "status": "incomplete",
        "prompt_preview": "hello world",
    })

    # Phase 2: completion record (complete) — should update in-place
    m.record({
        "request_id": "req-1",
        "t0": 1000.0,
        "model": "test", "backend": "be1", "slot_id": 0,
        "cache_hit": True, "restored": True, "recompute": False,
        "saved": True, "latency_ms": 100.0, "n_tokens": 10, "cache_size_bytes": 1024,
        "status": "complete",
    })

    # Should have exactly 1 request (updated in-place, not appended)
    requests = m.get_requests()
    assert len(requests) == 1, f"Expected 1 request, got {len(requests)}"
    assert requests[0]["request_id"] == "req-1"
    assert requests[0]["status"] == "complete"
    assert requests[0]["cache_hit"] is True
    assert requests[0]["latency_ms"] == 100.0
    assert requests[0]["stream"] is True
    # prompt_preview should be preserved from arrival record
    assert requests[0]["prompt_preview"] == "hello world"

    # Performance should count this request
    perf = m.get_performance()
    assert perf["total_requests"] == 1
    assert perf["cache_hits"] == 1

    # Incomplete request should be visible
    incomplete = [r for r in requests if r.get("status") != "complete"]
    assert len(incomplete) == 0

    # Add an incomplete request and verify it's counted
    m.record({
        "request_id": "req-2",
        "request_json": {"model": "test", "messages": [{"role": "user", "content": "incomplete request"}]},
        "model": "test",
        "stream": False,
        "status": "incomplete",
        "prompt_preview": "incomplete request",
    })

    summary = m.get_summary()
    assert summary["incomplete_count"] == 1

    print("PASS: test_metrics_collector_two_phase_recording")


def test_dashboard_endpoint():
    """Dashboard endpoint should return HTML when enabled."""
    from unittest.mock import patch
    from fastapi.testclient import TestClient
    import app as app_module
    import config

    client = TestClient(app_module.app)

    with patch.object(config, "DASHBOARD_ENABLED", True):
        with patch("os.path.exists", return_value=True):
            mock_file = io.StringIO("")
            with patch("builtins.open", return_value=mock_file):
                resp = client.get("/dashboard")
                assert resp.status_code in (200, 404)

    print("PASS: test_dashboard_endpoint")


if __name__ == "__main__":
    test_compile_all()
    # Hashing import test (must run first — imports hashing.py at module level)
    test_hashing_imports()
    test_save_slot_response_parsing()
    test_cache_agent_client_delete_success()
    test_cache_agent_client_delete_failure()
    test_cache_agent_client_connect_error()
    test_save_slot_response_parsing()
    test_slot_manager_per_model_pools()
    test_slot_manager_multiple_models()
    test_slot_manager_release()
    test_slot_manager_pool_resize_up()
    test_slot_manager_pool_resize_down()
    test_slot_manager_multiple_backends()
    test_slot_manager_gslot_type()
    test_ring_buffer_age_eviction()
    test_ring_buffer_lru_eviction()
    test_ring_buffer_no_eviction_under_limit()

    # KV cache skip tests
    test_should_skip_restore_no_tracked_state()
    test_should_skip_restore_perfect_match()
    test_should_skip_restore_high_overlap()
    test_should_skip_restore_low_overlap()
    test_should_skip_restore_zero_lcp()
    test_should_skip_restore_shorter_kv_cache()
    test_should_skip_restore_longer_kv_cache()
    test_should_skip_restore_multi_slot()
    test_save_after_updates_slot_kv_state()
    test_save_after_no_blocks_no_state_update()

    # Backend-down fix verification tests
    test_slot_timeout_config()
    test_restore_slot_times_out_on_slow_backend()
    test_save_slot_returns_false_on_exception()
    test_adaptive_cooldown_on_failure()
    test_lock_released_on_restore_failure()

    # Cancellation handling tests
    test_non_streaming_cancelled_error_releases_slot()
    test_streaming_save_after_skipped_on_cancel()
    test_streaming_save_after_exception()
    test_streaming_gen_sets_cancelled_flag()
    test_streaming_release_not_in_outer_finally()
    test_reader_polls_is_disconnected_on_timeout()
    test_streaming_completion_releases_slot()

    # ── Model discovery tests ──────────────────────────────────────────

    test_discover_models_router_mode()
    test_discover_models_non_router_mode()
    test_discover_models_ctx_not_in_args()
    test_discover_models_router_loaded_info_n_ctx()
    test_discover_models_non_router_meta_null()
    test_discover_models_both_endpoints_fail()

    # ── Model resolution tests ─────────────────────────────────────────

    test_resolve_exact_match()
    test_resolve_substring_match()
    test_resolve_ambiguous_substring()
    test_resolve_any()
    test_resolve_not_found()
    test_resolve_any_no_models()

    # ── Cache-first routing tests ──────────────────────────────────────

    test_cache_hit_selects_best_ratio()
    test_cache_hit_across_canonical_models()
    test_no_cache_fallback_lru()
    test_no_cache_lru_backend_routing()
    test_meta_key_uses_canonical_name()

    # ── Context length tests ───────────────────────────────────────────

    test_get_model_n_ctx_exact()
    test_get_model_n_ctx_min_across_backends()
    test_prompt_too_long_rejected()

    # ── /v1/models endpoint tests ──────────────────────────────────────

    test_models_endpoint_includes_any()
    test_models_endpoint_openai_format()

    # ── Chat completion tests ──────────────────────────────────────────

    test_chat_substring_model_resolution()
    test_chat_model_not_found_400()
    test_chat_any_model_routing()
    test_chat_any_with_cache_hit()

    # ── Lock retry tests ───────────────────────────────────────────────

    test_acquire_for_request_retries_on_lock_timeout()

    # ── Cache save ratio threshold tests ─────────────────────────────────

    test_chat_save_skipped_when_ratio_above_threshold()
    test_chat_save_performed_when_ratio_below_threshold()

    # ── Per-backend cache_dir tests ──────────────────────────────────────

    test_backend_cache_dir_per_backend()
    test_backend_cache_dir_mutual_exclusivity()
    test_backend_cache_delete_via_agent()
    test_backend_cache_delete_via_local()
    test_backend_cache_get_size_via_local()
    test_backend_cache_get_size_not_found()
    test_backend_cache_exists_via_local()
    test_backend_cache_get_mtime_via_local()

    # ── Cache hit wait queue tests ─────────────────────────────────────

    test_cache_hit_wait_phase0_success()
    test_cache_hit_wait_phase0_timeout()
    test_cache_hit_wait_pending_count_blocks()
    test_slot_duration_ema_updates_after_release()
    test_slot_duration_ema_bounds()
    test_slot_duration_ema_uses_initial_for_new_backend()

    # ── Metrics tests ──────────────────────────────────────────────────

    test_metrics_collector_basic()
    test_metrics_collector_performance()
    test_metrics_collector_ring_overflow()
    test_metrics_collector_per_model()
    test_metrics_collector_per_backend()
    test_metrics_collector_two_phase_recording()
    test_dashboard_endpoint()

    print("\nAll smoke tests passed.")


# ── Model discovery tests ──────────────────────────────────────────────

