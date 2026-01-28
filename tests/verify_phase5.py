
import sys
import unittest
from unittest.mock import MagicMock

# Ensure we can import jaxonflow
sys.path.append(".")

from jaxonflow.async_backend import AsyncKernelGenerator
from jaxonflow.warmup import WarmupSystem
from jaxonflow.cost import CostTracker, DEFAULT_PRICING
from jaxonflow.telemetry import TelemetryLogger, TelemetryEvent

class TestPhase5(unittest.TestCase):
    def test_async_backend(self):
        print("Testing AsyncBackend...")
        mock_backend = MagicMock()
        generator = AsyncKernelGenerator(mock_backend)
        self.assertIsNotNone(generator)
        generator.shutdown()
        print("AsyncBackend: OK")

    def test_warmup(self):
        print("Testing WarmupSystem...")
        mock_backend = MagicMock()
        warmup = WarmupSystem(mock_backend)
        warmup.warmup(verbose=False)
        # Verify calls were made - checking if at least one call to get_or_generate_kernel happened
        self.assertTrue(mock_backend.get_or_generate_kernel.called)
        print("WarmupSystem: OK")

    def test_cost(self):
        print("Testing CostTracker...")
        tracker1 = CostTracker()
        tracker2 = CostTracker()
        self.assertIs(tracker1, tracker2)
        
        tracker1.reset()
        tracker1.track("gpt-4o", 1000, 500)
        
        cost = tracker1.get_total_cost()
        expected = (1000/1e6 * 5.0) + (500/1e6 * 15.0)
        self.assertAlmostEqual(cost, expected)
        print(f"CostTracker: OK (Cost: ${cost:.6f})")

    def test_telemetry(self):
        print("Testing TelemetryLogger...")
        logger1 = TelemetryLogger()
        logger2 = TelemetryLogger()
        self.assertIs(logger1, logger2)
        
        logger1.reset()
        logger1.log(TelemetryEvent.SYSTEM_INIT, version="0.1.0")
        
        metrics = logger1.get_metrics()
        self.assertEqual(metrics[TelemetryEvent.SYSTEM_INIT.value], 1)
        print("TelemetryLogger: OK")

if __name__ == "__main__":
    unittest.main()
