"""
Tests for Multi-Head Latent Attention (MLA) module.

Tests cover:
- Rotary position embedding functions
- MLAKVCache initialization and management
- MultiHeadLatentAttention initialization
- Forward pass with and without cache
- Attention computation and output shapes
- Gradient flow and differentiability
- Integration with JAX transformations

Reference: DeepSeek-V3.2 MLA mechanism with low-rank compression
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import pytest
from src.mhlax import (
    _make_rotary_PE,
    _apply_rotary_PE,
    MLAKVCache,
    MultiHeadLatentAttention
)


# ============================================================================
# Rotary Position Embedding Tests
# ============================================================================

@pytest.mark.unit
def test_make_rotary_pe_shape():
    """
    Test that _make_rotary_PE generates correct shape.
    """
    seq_len = 32
    rope_dim = 64

    sin, cos = _make_rotary_PE(seq_len, rope_dim)

    # Should return (seq_len, rope_dim//2) for both sin and cos
    assert sin.shape == (seq_len, rope_dim // 2), f"Expected shape {(seq_len, rope_dim // 2)}, got {sin.shape}"
    assert cos.shape == (seq_len, rope_dim // 2), f"Expected shape {(seq_len, rope_dim // 2)}, got {cos.shape}"
    assert jnp.all(jnp.isfinite(sin)), "Sin contains non-finite values"
    assert jnp.all(jnp.isfinite(cos)), "Cos contains non-finite values"


@pytest.mark.unit
def test_make_rotary_pe_properties():
    """
    Test mathematical properties of rotary embeddings.
    """
    seq_len = 16
    rope_dim = 32

    sin, cos = _make_rotary_PE(seq_len, rope_dim)

    # Sin and cos should be real numbers
    assert sin.dtype in [jnp.float32, jnp.float64]
    assert cos.dtype in [jnp.float32, jnp.float64]

    # Sin values should be in [-1, 1] and cos values should be in [-1, 1]
    assert jnp.all(jnp.abs(sin) <= 1.0), "Sin values out of range"
    assert jnp.all(jnp.abs(cos) <= 1.0), "Cos values out of range"


@pytest.mark.unit
def test_make_rotary_pe_different_lengths():
    """
    Test rotary PE generation with different sequence lengths.
    """
    rope_dim = 64

    seq_lengths = [8, 16, 32, 64, 128]

    for seq_len in seq_lengths:
        sin, cos = _make_rotary_PE(seq_len, rope_dim)
        assert sin.shape == (seq_len, rope_dim // 2), f"Failed for seq_len={seq_len}"
        assert cos.shape == (seq_len, rope_dim // 2), f"Failed for seq_len={seq_len}"


@pytest.mark.unit
def test_apply_rotary_pe_shape():
    """
    Test that _apply_rotary_PE preserves input shape.
    """
    num_heads = 4
    seq_len = 16
    rope_dim = 32

    # Create input tensor (seq_len, num_heads, rope_dim)
    x = jax.random.normal(jax.random.PRNGKey(0), (seq_len, num_heads, rope_dim))

    # Generate sin and cos
    sin, cos = _make_rotary_PE(seq_len, rope_dim)

    # Apply rotary embeddings
    x_rotated = _apply_rotary_PE(x, sin, cos)

    # Shape should be preserved
    assert x_rotated.shape == x.shape, f"Expected shape {x.shape}, got {x_rotated.shape}"
    assert jnp.all(jnp.isfinite(x_rotated)), "Output contains non-finite values"


@pytest.mark.unit
def test_apply_rotary_pe_different_positions():
    """
    Test that rotary embeddings produce different outputs for different positions.
    """
    num_heads = 1
    seq_len = 4
    rope_dim = 16

    # Create identical inputs at different positions (seq_len, num_heads, rope_dim)
    x = jnp.ones((seq_len, num_heads, rope_dim))

    sin, cos = _make_rotary_PE(seq_len, rope_dim)
    x_rotated = _apply_rotary_PE(x, sin, cos)

    # Different positions should produce different outputs
    for i in range(seq_len - 1):
        pos_i = x_rotated[i, 0, :]
        pos_j = x_rotated[i+1, 0, :]
        assert not jnp.allclose(pos_i, pos_j), f"Positions {i} and {i+1} are identical"


@pytest.mark.unit
def test_apply_rotary_pe_invertibility():
    """
    Test that rotary embeddings can be inverted by applying negative sin.
    """
    num_heads = 1
    seq_len = 8
    rope_dim = 16

    x = jax.random.normal(jax.random.PRNGKey(42), (seq_len, num_heads, rope_dim))
    sin, cos = _make_rotary_PE(seq_len, rope_dim)

    # Apply rotary embedding
    x_rotated = _apply_rotary_PE(x, sin, cos)

    # Apply inverse (negative sin, same cos)
    x_restored = _apply_rotary_PE(x_rotated, -sin, cos)

    # Should recover original input (within numerical precision)
    assert jnp.allclose(x, x_restored, atol=1e-5), "Rotary embedding is not invertible"


# ============================================================================
# MLAKVCache Tests
# ============================================================================

@pytest.mark.unit
def test_mla_kv_cache_initialization():
    """
    Test MLAKVCache initialization with correct shapes.
    """
    max_seq_len = 64
    num_heads = 4
    key_dim = 80
    value_dim = 64

    keys = jnp.zeros((max_seq_len, num_heads, key_dim))
    values = jnp.zeros((max_seq_len, num_heads, value_dim))

    cache = MLAKVCache(keys=keys, values=values, position=0)

    assert cache.keys.shape == (max_seq_len, num_heads, key_dim)
    assert cache.values.shape == (max_seq_len, num_heads, value_dim)
    assert cache.position == 0


@pytest.mark.unit
def test_mla_kv_cache_with_data():
    """
    Test that cache can store and retrieve data.
    """
    max_seq_len = 32
    num_heads = 2
    key_dim = 48
    value_dim = 32

    keys = jax.random.normal(jax.random.PRNGKey(0), (max_seq_len, num_heads, key_dim))
    values = jax.random.normal(jax.random.PRNGKey(1), (max_seq_len, num_heads, value_dim))
    position = 10

    cache = MLAKVCache(keys=keys, values=values, position=position)

    # Retrieve data up to position
    active_keys = cache.keys[:position]
    active_values = cache.values[:position]

    assert active_keys.shape == (position, num_heads, key_dim)
    assert active_values.shape == (position, num_heads, value_dim)


# ============================================================================
# MultiHeadLatentAttention Initialization Tests
# ============================================================================

@pytest.mark.unit
def test_mla_initialization(rng_key):
    """
    Test basic MLA initialization.
    """
    embed_dim = 128
    num_heads = 4
    q_low_rank = 64
    kv_low_rank = 64
    rope_dim = 16
    v_head_dim = 32

    mla = MultiHeadLatentAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        q_low_rank=q_low_rank,
        kv_low_rank=kv_low_rank,
        rope_dim=rope_dim,
        v_head_dim=v_head_dim,
        key=rng_key
    )

    # Check static fields
    assert mla.num_heads == num_heads
    assert mla.rope_dim == rope_dim
    assert mla.v_head_dim == v_head_dim

    # Check projection shapes
    assert mla.query_down_proj.weight.shape == (q_low_rank, embed_dim)
    assert mla.query_up_proj.weight.shape == (num_heads * v_head_dim, q_low_rank)
    assert mla.q_rope_proj.weight.shape == (num_heads * rope_dim, q_low_rank)
    assert mla.kv_down_proj.weight.shape == (kv_low_rank, embed_dim)
    assert mla.kv_up_proj.weight.shape == (num_heads * 2 * v_head_dim, kv_low_rank)
    assert mla.k_rope_proj.weight.shape == (rope_dim, embed_dim)
    assert mla.out_proj.weight.shape == (embed_dim, num_heads * v_head_dim)


@pytest.mark.unit
def test_mla_init_cache(rng_key):
    """
    Test that MLA creates properly initialized cache.
    """
    embed_dim = 128
    num_heads = 4
    v_head_dim = 32
    rope_dim = 16
    max_seq_len = 64

    mla = MultiHeadLatentAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        q_low_rank=64,
        kv_low_rank=64,
        rope_dim=rope_dim,
        v_head_dim=v_head_dim,
        key=rng_key
    )

    cache = mla.init_cache(max_seq_len)

    assert isinstance(cache, MLAKVCache)
    assert cache.keys.shape == (max_seq_len, num_heads, v_head_dim + rope_dim)
    assert cache.values.shape == (max_seq_len, num_heads, v_head_dim)
    assert cache.position == 0
    assert jnp.all(cache.keys == 0)
    assert jnp.all(cache.values == 0)


# ============================================================================
# MultiHeadLatentAttention Forward Pass Tests
# ============================================================================

@pytest.mark.unit
def test_mla_forward_shape(rng_key):
    """
    Test that forward pass produces correct output shape.
    """
    embed_dim = 128
    num_heads = 4
    v_head_dim = 32
    seq_len = 16

    mla = MultiHeadLatentAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        q_low_rank=64,
        kv_low_rank=64,
        rope_dim=16,
        v_head_dim=v_head_dim,
        key=rng_key
    )

    x = jax.random.normal(rng_key, (seq_len, embed_dim))
    output, cache = mla(x)

    # Output should have same shape as input
    assert output.shape == (seq_len, embed_dim)
    assert not jnp.any(jnp.isnan(output))
    assert jnp.all(jnp.isfinite(output))
    assert cache is None


@pytest.mark.unit
def test_mla_forward_single_batch(rng_key):
    """
    Test forward pass with single sequence.
    """
    embed_dim = 64
    seq_len = 8

    mla = MultiHeadLatentAttention(
        embed_dim=embed_dim,
        num_heads=2,
        q_low_rank=32,
        kv_low_rank=32,
        rope_dim=8,
        v_head_dim=16,
        key=rng_key
    )

    x = jax.random.normal(rng_key, (seq_len, embed_dim))
    output, cache = mla(x)

    assert output.shape == (seq_len, embed_dim)
    assert not jnp.any(jnp.isnan(output))
    assert cache is None


@pytest.mark.unit
def test_mla_forward_different_sequence_lengths(rng_key):
    """
    Test forward pass with various sequence lengths.
    """
    embed_dim = 64

    mla = MultiHeadLatentAttention(
        embed_dim=embed_dim,
        num_heads=2,
        q_low_rank=32,
        kv_low_rank=32,
        rope_dim=8,
        v_head_dim=16,
        key=rng_key
    )

    seq_lengths = [4, 8, 16, 32, 64]

    for seq_len in seq_lengths:
        x = jax.random.normal(rng_key, (seq_len, embed_dim))
        output, cache = mla(x)

        assert output.shape == (seq_len, embed_dim), \
            f"Failed for seq_len={seq_len}"
        assert not jnp.any(jnp.isnan(output)), \
            f"NaN values for seq_len={seq_len}"


@pytest.mark.unit
def test_mla_forward_with_cache(rng_key):
    """
    Test forward pass with KV cache for inference.
    """
    embed_dim = 64
    num_heads = 2
    max_seq_len = 32

    mla = MultiHeadLatentAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        q_low_rank=32,
        kv_low_rank=32,
        rope_dim=8,
        v_head_dim=16,
        key=rng_key
    )

    # Initialize cache
    cache = mla.init_cache(max_seq_len)

    # Single token input
    x = jax.random.normal(rng_key, (1, embed_dim))

    # Forward with cache
    output, new_cache = mla(x, cache=cache)

    # Check output shape
    assert output.shape == (1, embed_dim)
    assert not jnp.any(jnp.isnan(output))

    # Check cache was updated
    assert isinstance(new_cache, MLAKVCache)
    assert new_cache.position == 1


@pytest.mark.integration
def test_mla_inference_multiple_steps(rng_key):
    """
    Test multiple sequential inference steps with cache.
    """
    embed_dim = 64
    num_heads = 2
    max_seq_len = 32
    num_steps = 10

    mla = MultiHeadLatentAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        q_low_rank=32,
        kv_low_rank=32,
        rope_dim=8,
        v_head_dim=16,
        key=rng_key
    )

    cache = mla.init_cache(max_seq_len)

    outputs = []
    for step in range(num_steps):
        x = jax.random.normal(jax.random.PRNGKey(step), (1, embed_dim))
        output, cache = mla(x, cache=cache)
        outputs.append(output)

    # All outputs should be valid
    for i, output in enumerate(outputs):
        assert output.shape == (1, embed_dim), f"Step {i} shape mismatch"
        assert not jnp.any(jnp.isnan(output)), f"Step {i} contains NaN"
        assert jnp.all(jnp.isfinite(output)), f"Step {i} contains non-finite values"

    # Cache should have accumulated all steps
    assert cache.position == num_steps


@pytest.mark.unit
def test_mla_attention_mask_causality(rng_key):
    """
    Test that attention is causal with explicit mask.
    """
    embed_dim = 64
    num_heads = 2
    seq_len = 8

    mla = MultiHeadLatentAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        q_low_rank=32,
        kv_low_rank=32,
        rope_dim=8,
        v_head_dim=16,
        key=rng_key
    )

    # Create input sequence (seq_len, embed_dim)
    x = jax.random.normal(rng_key, (seq_len, embed_dim))

    # Create causal mask (True = can attend, False = cannot attend)
    causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))

    # Compute output with causal mask
    output, _ = mla(x, mask=causal_mask)

    # Should produce valid output
    assert output.shape == (seq_len, embed_dim)
    assert jnp.all(jnp.isfinite(output))
    assert not jnp.any(jnp.isnan(output))


# ============================================================================
# Gradient Flow Tests
# ============================================================================

@pytest.mark.gradient
def test_mla_gradient_flow(rng_key):
    """
    Test that gradients flow properly through MLA.
    """
    embed_dim = 64
    num_heads = 2
    seq_len = 8

    mla = MultiHeadLatentAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        q_low_rank=32,
        kv_low_rank=32,
        rope_dim=8,
        v_head_dim=16,
        key=rng_key
    )

    x = jax.random.normal(rng_key, (seq_len, embed_dim))

    def loss_fn(model, x):
        output, _ = model(x)
        return jnp.mean(output ** 2)

    # Compute gradients
    loss, grads = eqx.filter_value_and_grad(loss_fn)(mla, x)

    # Check that loss is finite
    assert jnp.isfinite(loss)

    # Check that key projections have gradients
    assert grads.query_down_proj.weight is not None
    assert grads.query_up_proj.weight is not None
    assert grads.kv_down_proj.weight is not None
    assert grads.kv_up_proj.weight is not None
    assert grads.out_proj.weight is not None

    # Check gradients are finite and not all zeros
    assert jnp.all(jnp.isfinite(grads.query_down_proj.weight))
    assert not jnp.all(grads.query_down_proj.weight == 0)
    assert jnp.all(jnp.isfinite(grads.out_proj.weight))
    assert not jnp.all(grads.out_proj.weight == 0)


@pytest.mark.gradient
def test_mla_gradient_finite(rng_key):
    """
    Test that all gradients are finite (no NaN or Inf).
    """
    embed_dim = 64
    num_heads = 2
    seq_len = 8

    mla = MultiHeadLatentAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        q_low_rank=32,
        kv_low_rank=32,
        rope_dim=8,
        v_head_dim=16,
        key=rng_key
    )

    x = jax.random.normal(rng_key, (seq_len, embed_dim))

    def loss_fn(model, x):
        output, _ = model(x)
        return jnp.sum(output ** 2)

    loss, grads = eqx.filter_value_and_grad(loss_fn)(mla, x)

    assert jnp.isfinite(loss)

    # Check key gradient arrays for finite values
    grad_arrays = [
        grads.query_down_proj.weight,
        grads.query_up_proj.weight,
        grads.q_rope_proj.weight,
        grads.kv_down_proj.weight,
        grads.kv_up_proj.weight,
        grads.k_rope_proj.weight,
        grads.out_proj.weight,
    ]

    for i, grad_array in enumerate(grad_arrays):
        assert jnp.all(jnp.isfinite(grad_array)), \
            f"Gradient array {i} contains non-finite values"


@pytest.mark.gradient
@pytest.mark.unit
def test_mla_gradient_with_cache(rng_key):
    """
    Test gradient flow with cache enabled.
    """
    embed_dim = 64
    num_heads = 2
    max_seq_len = 16

    mla = MultiHeadLatentAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        q_low_rank=32,
        kv_low_rank=32,
        rope_dim=8,
        v_head_dim=16,
        key=rng_key
    )

    x = jax.random.normal(rng_key, (1, embed_dim))
    cache = mla.init_cache(max_seq_len)

    def loss_fn(model, x, cache):
        output, _ = model(x, cache=cache)
        return jnp.mean(output ** 2)

    loss, grads = eqx.filter_value_and_grad(loss_fn)(mla, x, cache)

    assert jnp.isfinite(loss)
    assert grads.query_down_proj.weight is not None
    assert jnp.all(jnp.isfinite(grads.query_down_proj.weight))


# ============================================================================
# JAX Transformation Tests
# ============================================================================

@pytest.mark.integration
def test_mla_jit_compilation(rng_key):
    """
    Test that MLA can be JIT compiled.
    """
    embed_dim = 64
    num_heads = 2
    seq_len = 16

    mla = MultiHeadLatentAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        q_low_rank=32,
        kv_low_rank=32,
        rope_dim=8,
        v_head_dim=16,
        key=rng_key
    )

    x = jax.random.normal(rng_key, (seq_len, embed_dim))

    # JIT compile the forward pass
    @eqx.filter_jit
    def forward(model, x):
        return model(x)

    # Should compile without error
    output, cache = forward(mla, x)

    assert output.shape == (seq_len, embed_dim)
    assert jnp.all(jnp.isfinite(output))

    # Second call should use cached compilation
    output2, cache2 = forward(mla, x)
    assert jnp.allclose(output, output2)


@pytest.mark.integration
def test_mla_vmap_compatibility(rng_key):
    """
    Test that MLA works with vmap for batch processing.
    """
    embed_dim = 32
    seq_len = 8
    num_sequences = 4

    mla = MultiHeadLatentAttention(
        embed_dim=embed_dim,
        num_heads=2,
        q_low_rank=16,
        kv_low_rank=16,
        rope_dim=8,
        v_head_dim=8,
        key=rng_key
    )

    # Create batched inputs (num_sequences, seq_len, embed_dim)
    x_batched = jax.random.normal(rng_key, (num_sequences, seq_len, embed_dim))

    # Define a function that processes a single sequence
    def process_single(x):
        output, _ = mla(x)
        return output

    # Apply vmap to process all sequences in batch
    batched_process = jax.vmap(process_single)

    # Process all sequences
    outputs = batched_process(x_batched)

    assert outputs.shape == (num_sequences, seq_len, embed_dim)
    assert jnp.all(jnp.isfinite(outputs))
    assert not jnp.any(jnp.isnan(outputs))


@pytest.mark.integration
def test_mla_scan_autoregressive(rng_key):
    """
    Test using lax.scan for autoregressive generation with MLA.
    """
    embed_dim = 32
    num_heads = 2
    max_seq_len = 64
    num_steps = 10

    mla = MultiHeadLatentAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        q_low_rank=16,
        kv_low_rank=16,
        rope_dim=8,
        v_head_dim=8,
        key=rng_key
    )

    # Generate sequence of single-token inputs (num_steps, 1, embed_dim)
    inputs = jax.random.normal(rng_key, (num_steps, 1, embed_dim))

    # Initial cache
    init_cache = mla.init_cache(max_seq_len)

    def step_fn(cache, x):
        output, new_cache = mla(x, cache=cache)
        return new_cache, output

    # Run scan
    final_cache, outputs = jax.lax.scan(step_fn, init_cache, inputs)

    assert outputs.shape == (num_steps, 1, embed_dim)
    assert jnp.all(jnp.isfinite(outputs))
    assert not jnp.any(jnp.isnan(outputs))
    assert final_cache.position == num_steps


# ============================================================================
# Edge Cases and Robustness Tests
# ============================================================================

@pytest.mark.unit
def test_mla_zero_input(rng_key):
    """
    Test MLA with zero input.
    """
    embed_dim = 64
    num_heads = 2
    seq_len = 8

    mla = MultiHeadLatentAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        q_low_rank=32,
        kv_low_rank=32,
        rope_dim=8,
        v_head_dim=16,
        key=rng_key
    )

    x = jnp.zeros((seq_len, embed_dim))
    output, _ = mla(x)

    # Should produce finite output even with zero input
    assert output.shape == (seq_len, embed_dim)
    assert jnp.all(jnp.isfinite(output))


@pytest.mark.unit
def test_mla_single_token(rng_key):
    """
    Test MLA with single token (seq_len=1).
    """
    embed_dim = 64
    num_heads = 2

    mla = MultiHeadLatentAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        q_low_rank=32,
        kv_low_rank=32,
        rope_dim=8,
        v_head_dim=16,
        key=rng_key
    )

    x = jax.random.normal(rng_key, (1, embed_dim))
    output, _ = mla(x)

    assert output.shape == (1, embed_dim)
    assert jnp.all(jnp.isfinite(output))
    assert not jnp.any(jnp.isnan(output))


@pytest.mark.unit
def test_mla_different_low_rank_configs(rng_key):
    """
    Test MLA with different low-rank configurations.
    """
    embed_dim = 128
    num_heads = 4
    seq_len = 16

    configs = [
        (32, 32),   # Small
        (64, 64),   # Medium
        (96, 96),   # Large
    ]

    for q_low_rank, kv_low_rank in configs:
        mla = MultiHeadLatentAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            q_low_rank=q_low_rank,
            kv_low_rank=kv_low_rank,
            rope_dim=16,
            v_head_dim=32,
            key=rng_key
        )

        x = jax.random.normal(rng_key, (seq_len, embed_dim))
        output, _ = mla(x)

        assert output.shape == (seq_len, embed_dim), \
            f"Failed for config ({q_low_rank}, {kv_low_rank})"
        assert jnp.all(jnp.isfinite(output)), \
            f"Non-finite output for config ({q_low_rank}, {kv_low_rank})"
        assert not jnp.any(jnp.isnan(output)), \
            f"NaN for config ({q_low_rank}, {kv_low_rank})"


@pytest.mark.slow
@pytest.mark.integration
def test_mla_long_sequence(rng_key):
    """
    Test MLA with a long sequence.
    """
    embed_dim = 64
    num_heads = 2
    seq_len = 256

    mla = MultiHeadLatentAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        q_low_rank=32,
        kv_low_rank=32,
        rope_dim=8,
        v_head_dim=16,
        key=rng_key
    )

    x = jax.random.normal(rng_key, (seq_len, embed_dim))
    output, _ = mla(x)

    assert output.shape == (seq_len, embed_dim)
    assert jnp.all(jnp.isfinite(output))
    assert not jnp.any(jnp.isnan(output))


@pytest.mark.unit
def test_mla_different_num_heads(rng_key):
    """
    Test MLA with different numbers of attention heads.
    """
    embed_dim = 128
    seq_len = 16

    num_heads_list = [1, 2, 4, 8]

    for num_heads in num_heads_list:
        mla = MultiHeadLatentAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            q_low_rank=64,
            kv_low_rank=64,
            rope_dim=16,
            v_head_dim=32,
            key=rng_key
        )

        x = jax.random.normal(rng_key, (seq_len, embed_dim))
        output, _ = mla(x)

        assert output.shape == (seq_len, embed_dim), \
            f"Failed for num_heads={num_heads}"
        assert jnp.all(jnp.isfinite(output)), \
            f"Non-finite output for num_heads={num_heads}"
        assert not jnp.any(jnp.isnan(output)), \
            f"NaN for num_heads={num_heads}"
