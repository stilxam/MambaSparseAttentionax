"""
Pytest tests for SparseMambaTransformerBlock.
"""

import pytest
import jax
import jax.numpy as jnp
import equinox as eqx

from src.transformer import SparseMambaTransformerBlock, SparseMambaBlockInferenceCache


class TestSparseMambaTransformerBlock:
    """Test suite for SparseMambaTransformerBlock."""

    @pytest.fixture
    def basic_config(self):
        """Basic configuration for tests."""
        return {
            'dim': 64,
            'n_streams': 2,
            'num_heads': 4,
            'top_k': 8,
            'indexer_dim': 32,
            'mlp_ratio': 4,
        }

    @pytest.fixture
    def block(self, basic_config):
        """Create a basic transformer block."""
        key = jax.random.PRNGKey(42)
        return SparseMambaTransformerBlock(**basic_config, key=key)

    def test_initialization(self, basic_config):
        """Test that the block initializes without errors."""
        key = jax.random.PRNGKey(0)
        block = SparseMambaTransformerBlock(**basic_config, key=key)

        assert block.attn_mhc is not None
        assert block.mlp_mhc is not None
        assert hasattr(block.attn_mhc, 'layer_f')
        assert hasattr(block.mlp_mhc, 'layer_f')

    def test_forward_pass_shape(self, block, basic_config):
        """Test that forward pass produces correct output shape."""
        seq_len = 128
        dim = basic_config['dim']
        n_streams = basic_config['n_streams']

        key = jax.random.PRNGKey(1)
        x_stream = jax.random.normal(key, (seq_len, n_streams, dim))

        output, cache = block(x_stream)

        assert output.shape == (seq_len, n_streams, dim)
        assert cache is None  # No cache when not in inference mode

    def test_forward_pass_values(self, block, basic_config):
        """Test that forward pass produces finite values."""
        seq_len = 128
        dim = basic_config['dim']
        n_streams = basic_config['n_streams']

        key = jax.random.PRNGKey(2)
        x_stream = jax.random.normal(key, (seq_len, n_streams, dim))

        output, _ = block(x_stream)

        assert jnp.all(jnp.isfinite(output))
        assert output.dtype == x_stream.dtype

    def test_with_causal_mask(self, block, basic_config):
        """Test forward pass with causal mask (should still work but mask is not used)."""
        seq_len = 128
        dim = basic_config['dim']
        n_streams = basic_config['n_streams']

        key = jax.random.PRNGKey(3)
        x_stream = jax.random.normal(key, (seq_len, n_streams, dim))
        mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))

        # Mask is accepted but not passed to SparseMambaAttax (which has built-in causality)
        output, _ = block(x_stream, mask=mask)

        assert output.shape == (seq_len, n_streams, dim)
        assert jnp.all(jnp.isfinite(output))

    def test_cache_initialization(self, block):
        """Test cache initialization."""
        max_seq_len = 256
        cache = block.init_cache(max_seq_len)

        assert isinstance(cache, SparseMambaBlockInferenceCache)
        assert cache.attn_cache is not None

    def test_inference_with_cache(self, block, basic_config):
        """Test inference mode with caching."""
        seq_len = 128
        dim = basic_config['dim']
        n_streams = basic_config['n_streams']
        max_seq_len = 256

        key = jax.random.PRNGKey(4)
        x_stream = jax.random.normal(key, (seq_len, n_streams, dim))

        # Initialize cache
        cache = block.init_cache(max_seq_len)

        # Forward pass with cache
        output, new_cache = block(x_stream, cache=cache)

        assert output.shape == (seq_len, n_streams, dim)
        assert jnp.all(jnp.isfinite(output))
        assert new_cache is not None
        assert isinstance(new_cache, SparseMambaBlockInferenceCache)

    def test_autoregressive_generation(self, block, basic_config):
        """Test autoregressive token-by-token generation."""
        dim = basic_config['dim']
        n_streams = basic_config['n_streams']
        num_tokens = 10
        max_seq_len = 128

        cache = block.init_cache(max_seq_len)

        outputs = []
        for i in range(num_tokens):
            key = jax.random.PRNGKey(100 + i)
            token = jax.random.normal(key, (n_streams, dim))
            token_seq = token[None, :, :]  # (1, n_streams, dim)

            output, cache = block(token_seq, cache=cache)

            assert output.shape == (1, n_streams, dim)
            assert jnp.all(jnp.isfinite(output))
            outputs.append(output[0])

        assert len(outputs) == num_tokens

    def test_gradient_computation(self, block, basic_config):
        """Test that gradients can be computed."""
        seq_len = 128
        dim = basic_config['dim']
        n_streams = basic_config['n_streams']

        key = jax.random.PRNGKey(5)
        key_x, key_y = jax.random.split(key)

        x_stream = jax.random.normal(key_x, (seq_len, n_streams, dim))
        target = jax.random.normal(key_y, (seq_len, n_streams, dim))

        def loss_fn(model, x, y):
            pred, _ = model(x)
            return jnp.mean((pred - y) ** 2)

        # Compute gradients
        grads = jax.grad(loss_fn)(block, x_stream, target)

        # Check that all gradients are finite
        grad_leaves = jax.tree_util.tree_leaves(grads)
        grad_arrays = [g for g in grad_leaves if isinstance(g, jnp.ndarray)]

        assert len(grad_arrays) > 0
        assert all(jnp.all(jnp.isfinite(g)) for g in grad_arrays)

    def test_stacked_blocks(self, basic_config):
        """Test multiple stacked blocks."""
        seq_len = 128
        dim = basic_config['dim']
        n_streams = basic_config['n_streams']
        num_layers = 3

        key = jax.random.PRNGKey(6)
        keys = jax.random.split(key, num_layers + 1)
        key_data = keys[0]
        layer_keys = keys[1:]

        # Create stacked blocks
        blocks = [
            SparseMambaTransformerBlock(**basic_config, key=k)
            for k in layer_keys
        ]

        x_stream = jax.random.normal(key_data, (seq_len, n_streams, dim))

        # Forward through all layers
        current = x_stream
        for block in blocks:
            current, _ = block(current)
            assert current.shape == (seq_len, n_streams, dim)
            assert jnp.all(jnp.isfinite(current))

    def test_different_stream_counts(self, basic_config):
        """Test with different numbers of streams."""
        seq_len = 128
        dim = basic_config['dim']

        for n_streams in [1, 2, 4, 8]:
            key = jax.random.PRNGKey(10 + n_streams)
            key_model, key_data = jax.random.split(key)

            config = {**basic_config, 'n_streams': n_streams}
            block = SparseMambaTransformerBlock(**config, key=key_model)

            x_stream = jax.random.normal(key_data, (seq_len, n_streams, dim))
            output, _ = block(x_stream)

            assert output.shape == (seq_len, n_streams, dim)

    def test_jit_compilation(self, block, basic_config):
        """Test that the block can be JIT compiled."""
        seq_len = 128
        dim = basic_config['dim']
        n_streams = basic_config['n_streams']

        @jax.jit
        def forward(model, x):
            return model(x)

        key = jax.random.PRNGKey(7)
        x_stream = jax.random.normal(key, (seq_len, n_streams, dim))

        output, _ = forward(block, x_stream)

        assert output.shape == (seq_len, n_streams, dim)
        assert jnp.all(jnp.isfinite(output))

    def test_batch_processing(self, block, basic_config):
        """Test processing multiple independent sequences."""
        seq_len = 128
        dim = basic_config['dim']
        n_streams = basic_config['n_streams']

        key = jax.random.PRNGKey(8)
        x_stream = jax.random.normal(key, (seq_len, n_streams, dim))

        # Process same input twice
        output1, _ = block(x_stream)
        output2, _ = block(x_stream)

        # Should be deterministic
        assert jnp.allclose(output1, output2, rtol=1e-5)

    def test_parameter_count(self, block):
        """Test that the model has a reasonable number of parameters."""
        # Count parameters
        params = eqx.filter(block, eqx.is_array)
        param_leaves = jax.tree_util.tree_leaves(params)
        param_arrays = [p for p in param_leaves if isinstance(p, jnp.ndarray)]
        total_params = sum(p.size for p in param_arrays)

        assert total_params > 0
        # Should have parameters from attention, MLP, and mHC modules
        assert total_params > 10000  # Reasonable lower bound

    def test_single_token_sequence(self, block, basic_config):
        """Test with sequence length of 1 (edge case, but should work with cache)."""
        dim = basic_config['dim']
        n_streams = basic_config['n_streams']

        key = jax.random.PRNGKey(9)
        x_stream = jax.random.normal(key, (1, n_streams, dim))

        cache = block.init_cache(max_seq_len=128)
        output, new_cache = block(x_stream, cache=cache)

        assert output.shape == (1, n_streams, dim)
        assert jnp.all(jnp.isfinite(output))
        assert new_cache is not None


class TestSparseMambaBlockInferenceCache:
    """Test suite for SparseMambaBlockInferenceCache."""

    def test_cache_structure(self):
        """Test cache is a proper NamedTuple."""
        from src.sma import SparseMambaInferenceCache

        # Create mock cache components
        mock_attn_cache = None  # Would be SparseMambaInferenceCache in practice

        cache = SparseMambaBlockInferenceCache(attn_cache=mock_attn_cache)

        assert hasattr(cache, 'attn_cache')
        assert cache.attn_cache == mock_attn_cache


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
