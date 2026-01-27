"""Tests for JaxonFlow agents."""

import unittest
import tempfile
import shutil
from pathlib import Path

from jaxonflow.cache import KernelCache
from jaxonflow.spec import KernelSpec
from jaxonflow.agents.verification import KernelVerifier
from jaxonflow.agents.orchestrator import MultiAgentOrchestrator
from jaxonflow.dispatch import AgentBackendConfig
from jaxonflow.config import CacheConfig

class TestKernelCache(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.config = CacheConfig(
            enabled=True,
            cache_dir=Path(self.test_dir)
        )
        self.cache = KernelCache(self.config)

    def tearDown(self):
        self.cache.close()
        shutil.rmtree(self.test_dir)

    def test_put_get(self):
        key = "test_key"
        self.cache.put(key, "op", "code", {"meta": "data"})
        entry = self.cache.get(key)
        self.assertIsNotNone(entry)
        self.assertEqual(entry.code, "code")


class TestVerifier(unittest.TestCase):
    def test_verification_pass(self):
        spec = KernelSpec(
            operation="add",
            input_shapes=[(10,), (10,)],
            input_dtypes=["float32", "float32"],
            output_shape=(10,),
            output_dtype="float32",
            reference_impl=lambda x, y: x + y
        )
        verifier = KernelVerifier(None)
        
        # Correct kernel
        result = verifier.run(lambda x, y: x + y, spec)
        self.assertTrue(result.correct)

    def test_verification_fail(self):
        spec = KernelSpec(
            operation="add",
            input_shapes=[(10,), (10,)],
            input_dtypes=["float32", "float32"],
            output_shape=(10,),
            output_dtype="float32",
            reference_impl=lambda x, y: x + y
        )
        verifier = KernelVerifier(None)
        
        # Incorrect kernel
        result = verifier.run(lambda x, y: x - y, spec)
        self.assertFalse(result.correct)
