# Testing Guide for JaxonFlow

This guide explains how to run tests for the JaxonFlow backend.

## Prerequisites

- Python 3.8+
- `jax`, `torch`
- `pytest`

## Running Tests

Since this environment might not have a GPU, the tests are designed to mock the GPU-dependent parts or run using CPU fallbacks where possible.

### Run All Tests

```bash
python3 -m unittest discover tests
```

### Run Specific Tests

```bash
python3 -m unittest tests/test_dispatch.py
python3 -m unittest tests/test_agents.py
```

## GPU Verification

To verify on a machine with a GPU:

1. Ensure CUDA is installed and `nvidia-smi` works.
2. Install JAX with CUDA support.
3. Set `JAXONFLOW_HARDWARE="auto"` (default).
4. Run the verification script (to be added).
