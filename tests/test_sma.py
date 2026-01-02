"""
Pytest tests for SparseMambaAttax (Sparse Mamba Attention).
"""

import pytest
import jax
import jax.numpy as jnp
import equinox as eqx

from src.sma import SparseMambaAttax, MambaIndexer, SparseMambaInferenceCache
from src.mhlax import _make_rotary_PE


class TestMambaIndexer:
    """Test suite for MambaIndexer."""

    @pytest.fixture
    def basic_config(self):
        """Basic configuration for indexer tests."""
        return {
            'd_model': 64,
            'indexer_dim': 32,
        }

    @pytest.fixture
    def indexer(self, basic_config):
        """Create a basic indexer."""
        key = jax.random.PRNGKey(42)
        return MambaIndexer(**basic_config, key=key)

    def test_initialization(self, basic_config):
        """Test that the indexer initializes without errors."""
        key = jax.random.PRNGKey(0)
        indexer = MambaIndexer(**basic_config, key=key)

        assert indexer.mamba_block is not None
        assert indexer.q_proj is not None
        assert indexer.k_proj is not None
        assert indexer.scale == basic_config['indexer_dim'] ** -0.5

    def test_get_scores_shape(self, indexer):
        """Test that get_scores produces correct output shape."""
        seq_len = 128
        d_model = 64

        key = jax.random.PRNGKey(1)
        x = jax.random.normal(key, (seq_len, d_model))

        scores, cache = indexer.get_scores(x)

        assert scores.shape == (seq_len, seq_len)
        assert cache is None  # No cache in training mode

    def test_get_scores_causality(self, indexer):
        """Test that scores have causal masking (lower triangular)."""
        seq_len = 128
        d_model = 64

        key = jax.random.PRNGKey(2)
        x = jax.random.normal(key, (seq_len, d_model))

        scores, _ = indexer.get_scores(x)

        # Upper triangular should be -inf
        upper_tri_mask = jnp.triu(jnp.ones((seq_len, seq_len), dtype=bool), k=1)
        upper_tri_values = scores[upper_tri_mask]

        assert jnp.all(jnp.isinf(upper_tri_values))
        assert jnp.all(upper_tri_values < 0)  # Negative infinity

    def test_get_scores_finite_lower_triangle(self, indexer):
        """Test that lower triangle scores are finite."""
        seq_len = 128
        d_model = 64

        key = jax.random.PRNGKey(3)
        x = jax.random.normal(key, (seq_len, d_model))

        scores, _ = indexer.get_scores(x)

        # Lower triangular should be finite
        lower_tri_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
        lower_tri_values = scores[lower_tri_mask]

        assert jnp.all(jnp.isfinite(lower_tri_values))

    def test_get_scores_with_cache(self, indexer):
        """Test get_scores with cache (inference mode)."""
        seq_len = 128
        d_model = 64

        key = jax.random.PRNGKey(4)
        x = jax.random.normal(key, (seq_len, d_model))

        # Initialize cache
        cache = indexer.mamba_block.init_cache()

        scores, new_cache = indexer.get_scores(x, cache=cache)

        assert scores.shape == (seq_len, seq_len)
        assert new_cache is not None


class TestSparseMambaAttax:
    """Test suite for SparseMambaAttax."""

    @pytest.fixture
    def basic_config(self):
        """Basic configuration for SparseMambaAttax tests."""
        return {
            'embed_dim': 128,
            'num_heads': 4,
            'top_k': 16,
            'q_low_rank': 64,
            'kv_low_rank': 128,
            'rope_dim': 32,
            'v_head_dim': 32,
            'indexer_dim': 64,
        }

    @pytest.fixture
    def model(self, basic_config):
        """Create a basic SparseMambaAttax model."""
        key = jax.random.PRNGKey(42)
        return SparseMambaAttax(**basic_config, key=key)

    def test_initialization(self, basic_config):
        """Test that SparseMambaAttax initializes without errors."""
        key = jax.random.PRNGKey(0)
        model = SparseMambaAttax(**basic_config, key=key)

        assert model.indexer is not None
        assert model.mla is not None
        assert model.top_k == basic_config['top_k']
        assert model.num_heads == basic_config['num_heads']
        assert model.v_head_dim == basic_config['v_head_dim']
        assert model.rope_dim == basic_config['rope_dim']

    def test_forward_pass_shape(self, model, basic_config):
        """Test that forward pass produces correct output shape."""
        seq_len = 128
        embed_dim = basic_config['embed_dim']

        key = jax.random.PRNGKey(1)
        x = jax.random.normal(key, (seq_len, embed_dim))

        output, cache = model(x)

        assert output.shape == (seq_len, embed_dim)
        assert cache is None  # No cache in training mode

    def test_forward_pass_values(self, model, basic_config):
        """Test that forward pass produces finite values."""
        seq_len = 128
        embed_dim = basic_config['embed_dim']

        key = jax.random.PRNGKey(2)
        x = jax.random.normal(key, (seq_len, embed_dim))

        output, _ = model(x)

        assert jnp.all(jnp.isfinite(output))
        assert output.dtype == x.dtype

    def test_cache_initialization(self, model):
        """Test cache initialization."""
        max_seq_len = 256

        cache = model.init_cache(max_seq_len)

        assert isinstance(cache, SparseMambaInferenceCache)
        assert cache.indexer_cache is not None
        assert cache.mla_cache is not None

    def test_inference_with_cache(self, model, basic_config):
        """Test inference mode with caching."""
        seq_len = 128
        embed_dim = basic_config['embed_dim']
        max_seq_len = 256

        key = jax.random.PRNGKey(3)
        x = jax.random.normal(key, (seq_len, embed_dim))

        # Initialize cache
        cache = model.init_cache(max_seq_len)

        # Forward pass with cache
        output, new_cache = model(x, cache=cache)

        assert output.shape == (seq_len, embed_dim)
        assert jnp.all(jnp.isfinite(output))
        assert new_cache is not None
        assert isinstance(new_cache, SparseMambaInferenceCache)

    def test_autoregressive_generation(self, model, basic_config):
        """Test autoregressive token-by-token generation."""
        embed_dim = basic_config['embed_dim']
        num_tokens = 15
        max_seq_len = 128

        cache = model.init_cache(max_seq_len)

        outputs = []
        for i in range(num_tokens):
            key = jax.random.PRNGKey(100 + i)
            token = jax.random.normal(key, (1, embed_dim))  # Single token

            output, cache = model(token, cache=cache)

            assert output.shape == (1, embed_dim)
            assert jnp.all(jnp.isfinite(output))
            outputs.append(output)

        assert len(outputs) == num_tokens

        # Check that MLA cache position has been updated
        assert cache.mla_cache.position == num_tokens

    def test_gradient_computation(self, model, basic_config):
        """Test that gradients can be computed."""
        seq_len = 128
        embed_dim = basic_config['embed_dim']

        key = jax.random.PRNGKey(4)
        key_x, key_y = jax.random.split(key)

        x = jax.random.normal(key_x, (seq_len, embed_dim))
        target = jax.random.normal(key_y, (seq_len, embed_dim))

        def loss_fn(model, x, y):
            pred, _ = model(x)
            return jnp.mean((pred - y) ** 2)

        # Compute gradients
        grads = jax.grad(loss_fn)(model, x, target)

        # Check that all gradients are finite
        grad_leaves = jax.tree_util.tree_leaves(grads)
        grad_arrays = [g for g in grad_leaves if isinstance(g, jnp.ndarray)]

        assert len(grad_arrays) > 0
        assert all(jnp.all(jnp.isfinite(g)) for g in grad_arrays)

    def test_sparse_attention_top_k(self, model, basic_config):
        """Test that sparse attention uses top-k selection."""
        seq_len = 128
        embed_dim = basic_config['embed_dim']
        top_k = basic_config['top_k']

        # For sequences longer than top_k, sparse selection should occur
        assert seq_len > top_k

        key = jax.random.PRNGKey(5)
        x = jax.random.normal(key, (seq_len, embed_dim))

        output, _ = model(x)

        # Just verify it runs without errors and produces valid output
        assert output.shape == (seq_len, embed_dim)
        assert jnp.all(jnp.isfinite(output))

    def test_different_sequence_lengths(self, basic_config):
        """Test with different sequence lengths (all multiples of 128)."""
        embed_dim = basic_config['embed_dim']

        for seq_len in [128, 256, 384]:
            key = jax.random.PRNGKey(10 + seq_len)
            key_model, key_data = jax.random.split(key)

            model = SparseMambaAttax(**basic_config, key=key_model)
            x = jax.random.normal(key_data, (seq_len, embed_dim))

            output, _ = model(x)

            assert output.shape == (seq_len, embed_dim)
            assert jnp.all(jnp.isfinite(output))

    def test_jit_compilation(self, model, basic_config):
        """Test that the model can be JIT compiled."""
        seq_len = 128
        embed_dim = basic_config['embed_dim']

        @jax.jit
        def forward(model, x):
            return model(x)

        key = jax.random.PRNGKey(6)
        x = jax.random.normal(key, (seq_len, embed_dim))

        output, _ = forward(model, x)

        assert output.shape == (seq_len, embed_dim)
        assert jnp.all(jnp.isfinite(output))

    def test_deterministic_output(self, model, basic_config):
        """Test that output is deterministic for same input."""
        seq_len = 128
        embed_dim = basic_config['embed_dim']

        key = jax.random.PRNGKey(7)
        x = jax.random.normal(key, (seq_len, embed_dim))

        output1, _ = model(x)
        output2, _ = model(x)

        assert jnp.allclose(output1, output2, rtol=1e-5)

    def test_gather_kv_functionality(self, model):
        """Test the internal _gather_kv method."""
        seq_len = 128
        num_heads = 4
        dim_k = 64
        dim_v = 32
        top_k = 16

        key = jax.random.PRNGKey(8)
        k_key, v_key, idx_key = jax.random.split(key, 3)

        k = jax.random.normal(k_key, (seq_len, num_heads, dim_k))
        v = jax.random.normal(v_key, (seq_len, num_heads, dim_v))

        # Create random indices (must be within seq_len)
        indices = jax.random.randint(idx_key, (seq_len, top_k), 0, seq_len)

        k_gathered, v_gathered = model._gather_kv(k, v, indices)

        assert k_gathered.shape == (seq_len, top_k, num_heads, dim_k)
        assert v_gathered.shape == (seq_len, top_k, num_heads, dim_v)
        assert jnp.all(jnp.isfinite(k_gathered))
        assert jnp.all(jnp.isfinite(v_gathered))

    def test_parameter_count(self, model):
        """Test that the model has a reasonable number of parameters."""
        params = eqx.filter(model, eqx.is_array)
        param_leaves = jax.tree_util.tree_leaves(params)
        param_arrays = [p for p in param_leaves if isinstance(p, jnp.ndarray)]
        total_params = sum(p.size for p in param_arrays)

        assert total_params > 0
        # Should have parameters from indexer and MLA
        assert total_params > 10000  # Reasonable lower bound

    def test_small_top_k(self, basic_config):
        """Test with very small top_k value."""
        config = {**basic_config, 'top_k': 4}
        seq_len = 128
        embed_dim = config['embed_dim']

        key = jax.random.PRNGKey(9)
        key_model, key_data = jax.random.split(key)

        model = SparseMambaAttax(**config, key=key_model)
        x = jax.random.normal(key_data, (seq_len, embed_dim))

        output, _ = model(x)

        assert output.shape == (seq_len, embed_dim)
        assert jnp.all(jnp.isfinite(output))

    def test_large_top_k(self, basic_config):
        """Test with top_k larger than sequence length."""
        config = {**basic_config, 'top_k': 256}
        seq_len = 128
        embed_dim = config['embed_dim']

        key = jax.random.PRNGKey(10)
        key_model, key_data = jax.random.split(key)

        model = SparseMambaAttax(**config, key=key_model)
        x = jax.random.normal(key_data, (seq_len, embed_dim))

        # Should use min(top_k, seq_len)
        output, _ = model(x)

        assert output.shape == (seq_len, embed_dim)
        assert jnp.all(jnp.isfinite(output))

    def test_different_head_configurations(self, basic_config):
        """Test with different numbers of attention heads."""
        seq_len = 128
        embed_dim = basic_config['embed_dim']

        for num_heads in [2, 4, 8]:
            key = jax.random.PRNGKey(20 + num_heads)
            key_model, key_data = jax.random.split(key)

            config = {**basic_config, 'num_heads': num_heads}
            model = SparseMambaAttax(**config, key=key_model)

            x = jax.random.normal(key_data, (seq_len, embed_dim))
            output, _ = model(x)

            assert output.shape == (seq_len, embed_dim)
            assert jnp.all(jnp.isfinite(output))

    def test_cache_persistence_across_tokens(self, model, basic_config):
        """Test that cache properly persists state across multiple tokens."""
        embed_dim = basic_config['embed_dim']
        max_seq_len = 128

        cache = model.init_cache(max_seq_len)
        initial_position = cache.mla_cache.position

        # Process 5 tokens
        for i in range(5):
            key = jax.random.PRNGKey(200 + i)
            token = jax.random.normal(key, (1, embed_dim))
            _, cache = model(token, cache=cache)

        # Position should have advanced by 5
        assert cache.mla_cache.position == initial_position + 5

    def test_training_vs_inference_mode(self, model, basic_config):
        """Test that training mode (no cache) and inference mode (with cache) produce different behavior."""
        seq_len = 128
        embed_dim = basic_config['embed_dim']

        key = jax.random.PRNGKey(11)
        x = jax.random.normal(key, (seq_len, embed_dim))

        # Training mode
        output_train, cache_train = model(x)
        assert cache_train is None

        # Inference mode
        cache = model.init_cache(max_seq_len=256)
        output_inf, cache_inf = model(x, cache=cache)
        assert cache_inf is not None

        # Both should produce valid outputs
        assert output_train.shape == output_inf.shape
        assert jnp.all(jnp.isfinite(output_train))
        assert jnp.all(jnp.isfinite(output_inf))


class TestSparseMambaInferenceCache:
    """Test suite for SparseMambaInferenceCache."""

    def test_cache_structure(self):
        """Test cache is a proper NamedTuple."""
        from src.mambax import Mamba2InferenceCache
        from src.mhlax import MLAKVCache

        # Create mock cache components (with proper structure)
        # Mamba2InferenceCache requires dimension parameters
        mock_indexer_cache = Mamba2InferenceCache(
            d_inner=64,
            d_conv=4,
            nheads=4,
            headdim=16,
            d_state=64,
            ngroups=1
        )
        # Create sin/cos for RoPE (assuming rope_dim = 0 for this test, minimal size)
        sin, cos = _make_rotary_PE(256, 16)
        mock_mla_cache = MLAKVCache(
            keys=jnp.zeros((256, 4, 32)),
            values=jnp.zeros((256, 4, 32)),
            position=0,
            sin=sin,
            cos=cos
        )

        cache = SparseMambaInferenceCache(
            indexer_cache=mock_indexer_cache,
            mla_cache=mock_mla_cache
        )

        assert hasattr(cache, 'indexer_cache')
        assert hasattr(cache, 'mla_cache')
        assert cache.indexer_cache == mock_indexer_cache
        assert cache.mla_cache == mock_mla_cache

    def test_cache_initialization_from_model(self):
        """Test cache initialization from SparseMambaAttax."""
        key = jax.random.PRNGKey(12)
        model = SparseMambaAttax(
            embed_dim=64,
            num_heads=4,
            top_k=8,
            key=key
        )

        cache = model.init_cache(max_seq_len=128)

        assert isinstance(cache, SparseMambaInferenceCache)
        assert cache.indexer_cache is not None
        assert cache.mla_cache is not None
        assert cache.mla_cache.position == 0


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_end_to_end_sequence_processing(self):
        """Test complete sequence processing pipeline."""
        embed_dim = 128
        seq_len = 128
        num_heads = 8
        top_k = 16

        key = jax.random.PRNGKey(13)
        key_model, key_data = jax.random.split(key)

        model = SparseMambaAttax(
            embed_dim=embed_dim,
            num_heads=num_heads,
            top_k=top_k,
            key=key_model
        )

        x = jax.random.normal(key_data, (seq_len, embed_dim))

        # Process sequence
        output, _ = model(x)

        # Verify output
        assert output.shape == (seq_len, embed_dim)
        assert jnp.all(jnp.isfinite(output))

        # Verify we can compute loss
        target = jax.random.normal(key_data, (seq_len, embed_dim))
        loss = jnp.mean((output - target) ** 2)
        assert jnp.isfinite(loss)

    def test_multiple_forward_passes(self):
        """Test multiple forward passes with same model."""
        embed_dim = 64
        seq_len = 128

        key = jax.random.PRNGKey(14)
        model = SparseMambaAttax(embed_dim=embed_dim, num_heads=4, top_k=8, key=key)

        for i in range(5):
            key_data = jax.random.PRNGKey(100 + i)
            x = jax.random.normal(key_data, (seq_len, embed_dim))
            output, _ = model(x)

            assert output.shape == (seq_len, embed_dim)
            assert jnp.all(jnp.isfinite(output))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
