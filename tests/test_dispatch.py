"""Tests for JaxonFlow dispatch."""

import unittest
from unittest.mock import MagicMock, patch

from jaxonflow.dispatch import AgentBackend, AgentBackendConfig
from jaxonflow.spec import KernelSpec
from jaxonflow.hardware import HardwareContext

class TestAgentBackend(unittest.TestCase):
    def setUp(self):
        self.config = AgentBackendConfig(enable_caching=False)
        self.backend = AgentBackend(self.config)

    def test_initialization(self):
        self.assertIsInstance(self.backend.hardware_context, HardwareContext)
        self.assertIsNotNone(self.backend.agent_system)

    def test_get_or_generate_kernel(self):
        spec = KernelSpec(
            operation="matmul",
            input_shapes=[(128, 128), (128, 128)],
            input_dtypes=["float32", "float32"],
            output_shape=(128, 128),
            output_dtype="float32",
            hardware=self.backend.hardware_context
        )
        
        # Test generation flow
        kernel = self.backend.get_or_generate_kernel(spec)
        self.assertIsNotNone(kernel)
        self.assertEqual(kernel.spec.operation, "matmul")
        
        # Test callable
        result = kernel.callable("dummy_input")
        self.assertEqual(result, "dummy_input")
