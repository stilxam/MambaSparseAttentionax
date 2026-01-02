"""
Shared pytest fixtures and utilities for MambaSparseAttentionax tests.

This module provides common fixtures that can be used across all test files,
including:
- Random key generators
- Common model configurations
- Helper functions for testing JAX models
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import pytest
from typing import Tuple, Dict, Any


# ============================================================================
# Random Key Fixtures
# ============================================================================

@pytest.fixture
def rng_key():
    """Provides a basic JAX random key."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def rng_keys():
    """Provides a function to generate multiple JAX random keys."""
    def _make_keys(n: int, seed: int = 0):
        return jax.random.split(jax.random.PRNGKey(seed), n)
    return _make_keys


# ============================================================================
# Model Configuration Fixtures
# ============================================================================

@pytest.fixture
def small_model_config() -> Dict[str, Any]:
    """Small model configuration for fast testing."""
    return {
        "d_model": 64,
        "n_layers": 2,
        "vocab_size": 256,
        "seq_len": 32,
        "batch_size": 2,
    }


@pytest.fixture
def medium_model_config() -> Dict[str, Any]:
    """Medium model configuration for more thorough testing."""
    return {
        "d_model": 128,
        "n_layers": 4,
        "vocab_size": 512,
        "seq_len": 64,
        "batch_size": 4,
    }


@pytest.fixture
def mamba_config() -> Dict[str, Any]:
    """Default Mamba2 configuration."""
    return {
        "d_model": 128,
        "n_layers": 2,
        "d_state": 64,
        "d_conv": 4,
        "expand_factor": 2,
        "headdim": 64,
        "chunk_size": 64,
        "vocab_size": 256,
    }


@pytest.fixture
def attention_config() -> Dict[str, Any]:
    """Default Multi-Head Latent Attention configuration."""
    return {
        "d_model": 128,
        "n_heads": 4,
        "d_kv": 64,
        "rope_theta": 10000.0,
    }


@pytest.fixture
def mhc_config() -> Dict[str, Any]:
    """Default mHC (Manifold-Constrained Hyper-Connection) configuration."""
    return {
        "n_streams": 4,
        "dim": 64,
    }


# ============================================================================
# Input Data Fixtures
# ============================================================================

@pytest.fixture
def sample_sequence(rng_key) -> jnp.ndarray:
    """Generate a sample input sequence."""
    return jax.random.randint(rng_key, (8, 32), 0, 256)


@pytest.fixture
def sample_embeddings(rng_key) -> jnp.ndarray:
    """Generate sample embedding vectors."""
    return jax.random.normal(rng_key, (8, 32, 128))


@pytest.fixture
def sample_attention_input(rng_key) -> jnp.ndarray:
    """Generate sample attention input."""
    return jax.random.normal(rng_key, (8, 128))


# ============================================================================
# Helper Functions
# ============================================================================

@pytest.fixture
def assert_shape():
    """Fixture providing a shape assertion helper."""
    def _assert_shape(array: jnp.ndarray, expected_shape: Tuple[int, ...], name: str = "array"):
        """Assert that an array has the expected shape."""
        assert array.shape == expected_shape, (
            f"{name} shape mismatch: expected {expected_shape}, got {array.shape}"
        )
    return _assert_shape


@pytest.fixture
def assert_no_nans():
    """Fixture providing a NaN checking helper."""
    def _assert_no_nans(array: jnp.ndarray, name: str = "array"):
        """Assert that an array contains no NaN values."""
        assert not jnp.any(jnp.isnan(array)), f"{name} contains NaN values"
    return _assert_no_nans


@pytest.fixture
def assert_finite():
    """Fixture providing a finite value checking helper."""
    def _assert_finite(array: jnp.ndarray, name: str = "array"):
        """Assert that an array contains only finite values."""
        assert jnp.all(jnp.isfinite(array)), f"{name} contains non-finite values"
    return _assert_finite


@pytest.fixture
def check_gradients():
    """Fixture providing a gradient checking helper."""
    def _check_gradients(model: eqx.Module, x: jnp.ndarray, loss_fn=None):
        """
        Check that gradients can be computed for a model.

        Args:
            model: An Equinox module
            x: Input data
            loss_fn: Optional custom loss function. If None, uses sum of outputs.

        Returns:
            The computed gradients
        """
        if loss_fn is None:
            def default_loss(m, x):
                output = m(x)
                return jnp.sum(output)
            loss_fn = default_loss

        grads = jax.grad(loss_fn)(model, x)
        return grads

    return _check_gradients


# ============================================================================
# Performance Testing Utilities
# ============================================================================

@pytest.fixture
def benchmark_jit():
    """Fixture for benchmarking JIT compilation time."""
    def _benchmark(func, *args, **kwargs):
        """
        Benchmark a JAX function with JIT compilation.
        Returns (first_call_time, subsequent_call_time).
        """
        import time

        # First call (includes compilation)
        start = time.time()
        result1 = func(*args, **kwargs)
        first_time = time.time() - start

        # Second call (no compilation)
        start = time.time()
        result2 = func(*args, **kwargs)
        second_time = time.time() - start

        return first_time, second_time, result1

    return _benchmark


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "gradient: mark test as testing gradient computation"
    )
