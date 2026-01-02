"""
Tests for Mamba2 State Space Model implementation.

Tests cover:
- Mamba2 model initialization and configuration
- Forward pass (training mode with chunked computation)
- Inference mode (step-by-step and scan-based)
- Cache management
- Gradient flow and differentiability
- Shape consistency across different sequence lengths
- Integration with JAX transformations (JIT, vmap, grad)

Reference: Mamba2 architecture with chunked computation and efficient inference
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import pytest
from src.mambax import Mamba2, Mamba2InferenceCache, segsum_binary_op


# ============================================================================
# Utility Function Tests
# ============================================================================

@pytest.mark.unit
def test_segsum_binary_op():
    """
    Test the segsum binary operation used in associative scan.
    This operation combines (gate, value) pairs in the associative scan.
    """
    # Create sample (gate, value) pairs
    g1, v1 = jnp.array(0.5), jnp.array(2.0)
    g2, v2 = jnp.array(0.3), jnp.array(3.0)

    # Apply binary op: (g2*g1, g2*v1 + v2)
    result_g, result_v = segsum_binary_op((g1, v1), (g2, v2))

    expected_g = g2 * g1
    expected_v = g2 * v1 + v2

    assert jnp.allclose(result_g, expected_g), f"Gate mismatch: {result_g} vs {expected_g}"
    assert jnp.allclose(result_v, expected_v), f"Value mismatch: {result_v} vs {expected_v}"


@pytest.mark.unit
def test_segsum_binary_op_batched():
    """
    Test segsum_binary_op with batched inputs.
    """
    batch_size = 4
    g1 = jax.random.uniform(jax.random.PRNGKey(0), (batch_size,))
    v1 = jax.random.normal(jax.random.PRNGKey(1), (batch_size,))
    g2 = jax.random.uniform(jax.random.PRNGKey(2), (batch_size,))
    v2 = jax.random.normal(jax.random.PRNGKey(3), (batch_size,))

    result_g, result_v = segsum_binary_op((g1, v1), (g2, v2))

    assert result_g.shape == (batch_size,)
    assert result_v.shape == (batch_size,)
    assert jnp.all(jnp.isfinite(result_g))
    assert jnp.all(jnp.isfinite(result_v))


# ============================================================================
# Mamba2InferenceCache Tests
# ============================================================================

@pytest.mark.unit
def test_inference_cache_initialization(rng_key):
    """
    Test that inference cache initializes with correct shapes and values.
    """
    d_conv = 4
    d_inner = 128
    d_state = 64
    nheads = 4
    headdim = 32
    ngroups = 1

    cache = Mamba2InferenceCache(
        d_inner=d_inner,
        d_conv=d_conv,
        nheads=nheads,
        headdim=headdim,
        d_state=d_state,
        ngroups=ngroups,
    )

    d_conv_in = d_inner + 2 * ngroups * d_state
    assert cache.conv_state.shape == (d_conv_in, d_conv - 1)
    assert cache.ssm_state.shape == (nheads, headdim, d_state)
    assert jnp.all(cache.conv_state == 0)
    assert jnp.all(cache.ssm_state == 0)


@pytest.mark.unit
def test_mamba2_init_cache(rng_key):
    """
    Test that Mamba2.init_cache creates properly shaped cache.
    """
    d_model = 64

    model = Mamba2(
        d_model=d_model,
        d_state=32,
        d_conv=4,
        expand=2,
        headdim=32,
        key=rng_key
    )

    cache = model.init_cache()

    assert isinstance(cache, Mamba2InferenceCache)
    d_conv_in = model.d_inner + 2 * model.ngroups * model.d_state
    assert cache.conv_state.shape == (d_conv_in, model.d_conv - 1)
    assert cache.ssm_state.shape == (model.nheads, model.headdim, model.d_state)


# ============================================================================
# Mamba2 Initialization Tests
# ============================================================================

@pytest.mark.unit
def test_mamba2_initialization(rng_key):
    """
    Test basic Mamba2 initialization with default parameters.
    """
    d_model = 128
    model = Mamba2(
        d_model=d_model,
        d_state=64,
        d_conv=4,
        expand=2,
        headdim=64,
        key=rng_key
    )

    # Check static fields
    assert model.d_model == d_model
    assert model.d_inner == d_model * 2
    assert model.d_state == 64
    assert model.d_conv == 4
    assert model.headdim == 64
    assert model.nheads == model.d_inner // model.headdim

    # Check learned parameters exist and have correct shapes
    assert model.dt_bias.shape == (model.nheads,)
    assert model.A_log.shape == (model.nheads,)
    assert model.D.shape == (model.nheads,)


@pytest.mark.unit
def test_mamba2_initialization_dimensions(rng_key):
    """
    Test that d_inner must be divisible by headdim.
    """
    d_model = 128
    headdim = 64
    expand = 2

    # This should work (d_inner = 256, divisible by 64)
    model = Mamba2(
        d_model=d_model,
        headdim=headdim,
        expand=expand,
        key=rng_key
    )
    assert model.d_inner % model.headdim == 0

    # This should fail (d_inner = 150, not divisible by 64)
    with pytest.raises(AssertionError):
        Mamba2(
            d_model=75,  # 75 * 2 = 150, not divisible by 64
            headdim=64,
            expand=2,
            key=rng_key
        )


@pytest.mark.unit
def test_mamba2_initialization_groups(rng_key):
    """
    Test that nheads must be divisible by ngroups.
    """
    d_model = 128
    headdim = 32
    expand = 2
    # d_inner = 256, nheads = 256/32 = 8

    # This should work (8 heads, 4 groups)
    model = Mamba2(
        d_model=d_model,
        headdim=headdim,
        expand=expand,
        ngroups=4,
        key=rng_key
    )
    assert model.nheads % model.ngroups == 0

    # This should fail (8 heads, 3 groups)
    with pytest.raises(AssertionError):
        Mamba2(
            d_model=d_model,
            headdim=headdim,
            expand=expand,
            ngroups=3,  # 8 is not divisible by 3
            key=rng_key
        )


@pytest.mark.unit
def test_mamba2_parameter_initialization_ranges(rng_key):
    """
    Test that parameters are initialized within expected ranges.
    """
    model = Mamba2(
        d_model=64,
        d_state=32,
        headdim=32,
        dt_min=0.001,
        dt_max=0.1,
        A_init_range=(1, 16),
        key=rng_key
    )

    # A_log stores positive log(A) where A is in [1, 16]
    # So log(A) should be in [0, log(16)] ≈ [0, 2.77]
    # Negation happens during forward pass: A = -exp(A_log)
    assert jnp.all(model.A_log >= 0), "A_log should be positive (stores log of positive A)"
    assert jnp.all(model.A_log <= jnp.log(16)), "A_log should be <= log(16)"

    # D should be initialized to 1.0
    assert jnp.allclose(model.D, 1.0), "D should be initialized to 1.0"


# ============================================================================
# Mamba2 Forward Pass (Training Mode) Tests
# ============================================================================

@pytest.mark.unit
def test_mamba2_forward_shape(rng_key):
    """
    Test that forward pass preserves sequence shape.
    """
    d_model = 64
    seq_len = 32

    model = Mamba2(
        d_model=d_model,
        d_state=32,
        d_conv=4,
        expand=2,
        headdim=32,
        chunk_size=16,
        key=rng_key
    )

    # Input shape: (seq_len, d_model)
    x = jax.random.normal(rng_key, (seq_len, d_model))

    # Forward pass (training mode, no cache)
    output, cache = model(x)

    # Output should have same shape as input
    assert output.shape == (seq_len, d_model)
    assert not jnp.any(jnp.isnan(output))
    assert jnp.all(jnp.isfinite(output))
    assert cache is None


@pytest.mark.unit
def test_mamba2_forward_different_sequence_lengths(rng_key):
    """
    Test forward pass with various sequence lengths.
    """
    d_model = 64

    model = Mamba2(
        d_model=d_model,
        d_state=32,
        headdim=32,
        chunk_size=16,
        key=rng_key
    )

    seq_lengths = [8, 16, 32, 64, 127]  # Including non-multiple of chunk_size

    for seq_len in seq_lengths:
        x = jax.random.normal(rng_key, (seq_len, d_model))
        output, cache = model(x)

        assert output.shape == (seq_len, d_model), \
            f"Failed for seq_len={seq_len}"
        assert not jnp.any(jnp.isnan(output)), \
            f"NaN values for seq_len={seq_len}"
        assert cache is None


@pytest.mark.unit
def test_mamba2_forward_single_batch(rng_key):
    """
    Test forward pass with single sequence.
    """
    d_model = 64
    seq_len = 32

    model = Mamba2(
        d_model=d_model,
        d_state=32,
        headdim=32,
        key=rng_key
    )

    x = jax.random.normal(rng_key, (seq_len, d_model))
    output, cache = model(x)

    assert output.shape == (seq_len, d_model)
    assert not jnp.any(jnp.isnan(output))
    assert cache is None


@pytest.mark.unit
def test_mamba2_forward_chunked_computation(rng_key):
    """
    Test that chunked computation works correctly.
    """
    d_model = 64
    seq_len = 64
    chunk_size = 16

    model = Mamba2(
        d_model=d_model,
        d_state=32,
        headdim=32,
        chunk_size=chunk_size,
        key=rng_key
    )

    x = jax.random.normal(rng_key, (seq_len, d_model))

    # Should process as 64/16 = 4 chunks
    output, cache = model(x)

    assert output.shape == (seq_len, d_model)
    assert not jnp.any(jnp.isnan(output))
    assert cache is None


# ============================================================================
# Mamba2 Inference Mode Tests
# ============================================================================

@pytest.mark.unit
def test_mamba2_inference_step(rng_key):
    """
    Test single-step inference with cache.
    """
    d_model = 64

    model = Mamba2(
        d_model=d_model,
        d_state=32,
        d_conv=4,
        headdim=32,
        key=rng_key
    )

    # Initialize cache
    cache = model.init_cache()

    # Single token input: (d_model,)
    x = jax.random.normal(rng_key, (d_model,))

    # Run inference step
    output, new_cache = model(x, cache=cache)

    # Check output shape
    assert output.shape == (d_model,)
    assert not jnp.any(jnp.isnan(output))

    # Check that cache was updated
    assert isinstance(new_cache, Mamba2InferenceCache)
    assert not jnp.array_equal(cache.conv_state, new_cache.conv_state)
    assert not jnp.array_equal(cache.ssm_state, new_cache.ssm_state)


@pytest.mark.integration
def test_mamba2_inference_multiple_steps(rng_key):
    """
    Test multiple sequential inference steps.
    """
    d_model = 64
    num_steps = 10

    model = Mamba2(
        d_model=d_model,
        d_state=32,
        d_conv=4,
        headdim=32,
        key=rng_key
    )

    cache = model.init_cache()

    outputs = []
    for i in range(num_steps):
        x = jax.random.normal(jax.random.PRNGKey(i), (d_model,))
        output, cache = model(x, cache=cache)
        outputs.append(output)

    # All outputs should be valid
    for i, output in enumerate(outputs):
        assert output.shape == (d_model,), f"Step {i} shape mismatch"
        assert not jnp.any(jnp.isnan(output)), f"Step {i} contains NaN"
        assert jnp.all(jnp.isfinite(output)), f"Step {i} non-finite"


@pytest.mark.integration
def test_mamba2_inference_vs_training_consistency(rng_key):
    """
    Test that inference mode produces consistent results with training mode
    when processing the same sequence step-by-step vs all-at-once.

    Note: There may be small numerical differences due to chunked computation
    and numerical precision, but results should be similar.
    """
    d_model = 32
    seq_len = 8

    model = Mamba2(
        d_model=d_model,
        d_state=16,
        d_conv=4,
        headdim=16,
        chunk_size=4,
        key=rng_key
    )

    # Generate input sequence
    x_full = jax.random.normal(rng_key, (seq_len, d_model))

    # Training mode: process all at once
    output_training, _ = model(x_full)

    # Inference mode: process step by step
    cache = model.init_cache()
    outputs_inference = []

    for t in range(seq_len):
        x_t = x_full[t, :]  # (d_model,)
        output_t, cache = model(x_t, cache=cache)
        outputs_inference.append(output_t)

    output_inference = jnp.stack(outputs_inference, axis=0)

    # Shapes should match
    assert output_inference.shape == output_training.shape

    # Values should be reasonably close (allowing for numerical differences)
    # Note: Exact match may not be guaranteed due to different computation paths
    assert jnp.allclose(output_inference, output_training, atol=1e-2, rtol=1e-2), \
        "Inference and training outputs differ significantly"


# ============================================================================
# Gradient Flow Tests
# ============================================================================

@pytest.mark.gradient
def test_mamba2_gradient_flow(rng_key):
    """
    Test that gradients flow properly through Mamba2.
    """
    d_model = 32
    seq_len = 16

    model = Mamba2(
        d_model=d_model,
        d_state=16,
        d_conv=4,
        headdim=16,
        key=rng_key
    )

    x = jax.random.normal(rng_key, (seq_len, d_model))

    def loss_fn(model, x):
        output, _ = model(x)
        return jnp.mean(output ** 2)

    # Compute gradients
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x)

    # Check loss is finite
    assert jnp.isfinite(loss)

    # Check that key parameters have gradients
    assert grads.in_proj.weight is not None
    assert grads.out_proj.weight is not None
    assert grads.dt_bias is not None
    assert grads.A_log is not None

    # Check gradients are finite and not all zeros
    assert jnp.all(jnp.isfinite(grads.in_proj.weight))
    assert not jnp.all(grads.in_proj.weight == 0)
    assert jnp.all(jnp.isfinite(grads.out_proj.weight))
    assert not jnp.all(grads.out_proj.weight == 0)
    assert not jnp.all(grads.dt_bias == 0)


@pytest.mark.gradient
def test_mamba2_gradient_finite(rng_key):
    """
    Test that all gradients are finite (no NaN or Inf).
    """
    d_model = 32
    seq_len = 16

    model = Mamba2(
        d_model=d_model,
        d_state=16,
        d_conv=4,
        headdim=16,
        key=rng_key
    )

    x = jax.random.normal(rng_key, (seq_len, d_model))

    def loss_fn(model, x):
        output, _ = model(x)
        return jnp.sum(output ** 2)

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x)

    # Check loss is finite
    assert jnp.isfinite(loss)

    # Check key gradient arrays for finite values
    grad_arrays = [
        grads.in_proj.weight,
        grads.out_proj.weight,
        grads.dt_bias,
        grads.A_log,
        grads.D,
    ]

    for i, grad_array in enumerate(grad_arrays):
        assert jnp.all(jnp.isfinite(grad_array)), \
            f"Gradient array {i} contains non-finite values"


@pytest.mark.gradient
def test_mamba2_gradient_through_inference(rng_key):
    """
    Test that gradients can be computed through inference mode.
    """
    d_model = 32

    model = Mamba2(
        d_model=d_model,
        d_state=16,
        d_conv=4,
        headdim=16,
        key=rng_key
    )

    cache = model.init_cache()
    x = jax.random.normal(rng_key, (d_model,))

    def loss_fn(model, x, cache):
        output, _ = model(x, cache=cache)
        return jnp.mean(output ** 2)

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, cache)

    assert jnp.isfinite(loss)
    assert grads.in_proj.weight is not None
    assert jnp.all(jnp.isfinite(grads.in_proj.weight))


# ============================================================================
# JAX Transformation Tests
# ============================================================================

@pytest.mark.integration
def test_mamba2_jit_compilation(rng_key):
    """
    Test that Mamba2 works with JIT compilation.
    """
    d_model = 32
    seq_len = 16

    model = Mamba2(
        d_model=d_model,
        d_state=16,
        d_conv=4,
        headdim=16,
        key=rng_key
    )

    x = jax.random.normal(rng_key, (seq_len, d_model))

    @eqx.filter_jit
    def forward(model, x):
        return model(x)

    # Should compile and run without error
    output, cache = forward(model, x)

    assert output.shape == (seq_len, d_model)
    assert not jnp.any(jnp.isnan(output))
    assert cache is None

    # Run again to test cached compilation
    output2, cache2 = forward(model, x)
    assert jnp.allclose(output, output2)


@pytest.mark.integration
def test_mamba2_vmap_compatibility(rng_key):
    """
    Test that Mamba2 works with vmap for batch processing.
    """
    d_model = 32
    seq_len = 16
    num_sequences = 4

    model = Mamba2(
        d_model=d_model,
        d_state=16,
        d_conv=4,
        headdim=16,
        key=rng_key
    )

    # Create batched inputs (num_sequences, seq_len, d_model)
    x_batched = jax.random.normal(rng_key, (num_sequences, seq_len, d_model))

    # Define a function that processes a single sequence
    def process_single(x):
        output, _ = model(x)
        return output

    # Apply vmap
    batched_process = jax.vmap(process_single)

    # Process all sequences
    outputs = batched_process(x_batched)

    assert outputs.shape == (num_sequences, seq_len, d_model)
    assert jnp.all(jnp.isfinite(outputs))
    assert not jnp.any(jnp.isnan(outputs))


@pytest.mark.integration
def test_mamba2_scan_for_autoregressive_generation(rng_key):
    """
    Test using lax.scan for autoregressive generation.
    """
    d_model = 32
    num_steps = 10

    model = Mamba2(
        d_model=d_model,
        d_state=16,
        d_conv=4,
        headdim=16,
        key=rng_key
    )

    # Generate sequence of single token inputs (num_steps, d_model)
    inputs = jax.random.normal(rng_key, (num_steps, d_model))

    # Initial cache
    init_cache = model.init_cache()

    def step_fn(cache, x):
        output, new_cache = model(x, cache=cache)
        return new_cache, output

    # Run scan
    final_cache, outputs = jax.lax.scan(step_fn, init_cache, inputs)

    assert outputs.shape == (num_steps, d_model)
    assert jnp.all(jnp.isfinite(outputs))
    assert not jnp.any(jnp.isnan(outputs))
    assert isinstance(final_cache, Mamba2InferenceCache)


# ============================================================================
# Edge Cases and Robustness Tests
# ============================================================================

@pytest.mark.unit
def test_mamba2_zero_input(rng_key):
    """
    Test that Mamba2 handles zero input gracefully.
    """
    d_model = 32
    seq_len = 16

    model = Mamba2(
        d_model=d_model,
        d_state=16,
        d_conv=4,
        headdim=16,
        key=rng_key
    )

    x = jnp.zeros((seq_len, d_model))
    output, cache = model(x)

    assert output.shape == (seq_len, d_model)
    assert jnp.all(jnp.isfinite(output))
    assert cache is None


@pytest.mark.unit
def test_mamba2_very_small_sequence(rng_key):
    """
    Test with very small sequence (edge case).
    """
    d_model = 32
    seq_len = 2

    model = Mamba2(
        d_model=d_model,
        d_state=16,
        d_conv=4,
        headdim=16,
        key=rng_key
    )

    x = jax.random.normal(rng_key, (seq_len, d_model))
    output, cache = model(x)

    assert output.shape == (seq_len, d_model)
    assert not jnp.any(jnp.isnan(output))
    assert jnp.all(jnp.isfinite(output))
    assert cache is None


@pytest.mark.slow
@pytest.mark.integration
def test_mamba2_long_sequence(rng_key):
    """
    Test with a long sequence to verify chunking works.
    """
    d_model = 32
    seq_len = 512
    chunk_size = 64

    model = Mamba2(
        d_model=d_model,
        d_state=16,
        d_conv=4,
        headdim=16,
        chunk_size=chunk_size,
        key=rng_key
    )

    x = jax.random.normal(rng_key, (seq_len, d_model))
    output, cache = model(x)

    assert output.shape == (seq_len, d_model)
    assert not jnp.any(jnp.isnan(output))
    assert jnp.all(jnp.isfinite(output))
    assert cache is None


@pytest.mark.unit
def test_mamba2_different_chunk_sizes(rng_key):
    """
    Test that different chunk sizes produce valid outputs.
    """
    d_model = 32
    seq_len = 64

    chunk_sizes = [16, 32, 64, 128]

    for chunk_size in chunk_sizes:
        model = Mamba2(
            d_model=d_model,
            d_state=16,
            d_conv=4,
            headdim=16,
            chunk_size=chunk_size,
            key=rng_key
        )

        x = jax.random.normal(rng_key, (seq_len, d_model))
        output, cache = model(x)

        assert output.shape == (seq_len, d_model), \
            f"Failed for chunk_size={chunk_size}"
        assert not jnp.any(jnp.isnan(output)), \
            f"NaN for chunk_size={chunk_size}"
        assert jnp.all(jnp.isfinite(output)), \
            f"Non-finite for chunk_size={chunk_size}"
        assert cache is None
