"""
Tests for Manifold-Constrained Hyper-Connection (mHC) module.

Tests cover:
- Sinkhorn-Knopp algorithm for projecting onto Birkhoff polytope
- ManifoldConstrainedHyperConnection initialization and forward pass
- Gradient flow and differentiability
- Edge cases and convergence properties

Reference: https://www.arxiv.org/abs/2512.24880
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import pytest
from src.mHC import sinkhorn_knopp, ManifoldConstrainedHyperConnection


# ============================================================================
# Sinkhorn-Knopp Algorithm Tests
# ============================================================================

@pytest.mark.unit
def test_sinkhorn_knopp_properties():
    """
    Verifies that the Sinkhorn-Knopp implementation produces
    doubly stochastic matrices (rows sum to 1, cols sum to 1).
    """
    key = jax.random.PRNGKey(42)
    n = 10
    # Create a random log-matrix
    log_matrix = jax.random.normal(key, (n, n))

    # Apply Sinkhorn
    doubly_stochastic = sinkhorn_knopp(log_matrix, n_iters=100)

    # Check 1: Non-negativity
    assert jnp.all(doubly_stochastic >= 0), "Matrix contains negative values"

    # Check 2: Rows sum to 1
    row_sums = jnp.sum(doubly_stochastic, axis=1)
    assert jnp.allclose(row_sums, 1.0, atol=1e-4), f"Rows do not sum to 1: {row_sums}"

    # Check 3: Columns sum to 1
    col_sums = jnp.sum(doubly_stochastic, axis=0)
    assert jnp.allclose(col_sums, 1.0, atol=1e-4), f"Columns do not sum to 1: {col_sums}"


@pytest.mark.unit
def test_sinkhorn_convergence_identity():
    """
    Edge case: If the input log-matrix corresponds to a diagonal matrix
    (high values on diag, low elsewhere), Sinkhorn should ideally preserve it
    or converge close to Identity.
    """
    n = 5
    # Create a matrix that looks like Identity in log space
    # High value on diagonal, very negative off-diagonal
    log_matrix = jnp.eye(n) * 10.0 - 10.0 * (1 - jnp.eye(n))

    res = sinkhorn_knopp(log_matrix, n_iters=50)

    # Should be very close to Identity
    assert jnp.allclose(res, jnp.eye(n), atol=0.1)


@pytest.mark.unit
def test_sinkhorn_convergence_iterations():
    """
    Test that more iterations lead to better convergence.
    """
    key = jax.random.PRNGKey(123)
    n = 8
    log_matrix = jax.random.normal(key, (n, n))

    # Test with few iterations
    result_few = sinkhorn_knopp(log_matrix, n_iters=5)
    row_sums_few = jnp.sum(result_few, axis=1)
    col_sums_few = jnp.sum(result_few, axis=0)
    error_few = jnp.max(jnp.abs(row_sums_few - 1.0)) + jnp.max(jnp.abs(col_sums_few - 1.0))

    # Test with many iterations
    result_many = sinkhorn_knopp(log_matrix, n_iters=100)
    row_sums_many = jnp.sum(result_many, axis=1)
    col_sums_many = jnp.sum(result_many, axis=0)
    error_many = jnp.max(jnp.abs(row_sums_many - 1.0)) + jnp.max(jnp.abs(col_sums_many - 1.0))

    # More iterations should give better (lower) error
    assert error_many < error_few


@pytest.mark.unit
def test_sinkhorn_uniform_convergence():
    """
    Test that a uniform log-matrix converges to uniform doubly stochastic matrix.
    """
    n = 6
    # Uniform log-matrix (all zeros)
    log_matrix = jnp.zeros((n, n))

    result = sinkhorn_knopp(log_matrix, n_iters=50)

    # Should converge to uniform distribution (each element = 1/n)
    expected = jnp.ones((n, n)) / n
    assert jnp.allclose(result, expected, atol=1e-3)


# ============================================================================
# ManifoldConstrainedHyperConnection Initialization Tests
# ============================================================================

@pytest.mark.unit
def test_mhc_initialization(mhc_config, rng_key):
    """
    Verifies that the gating factors (alphas) are initialized
    to 0.01 as specified in the paper/appendix.
    """
    # Define a dummy layer
    dummy_layer = lambda x: x

    model = ManifoldConstrainedHyperConnection(
        layer_f=dummy_layer,
        n_streams=mhc_config["n_streams"],
        dim=mhc_config["dim"],
        key=rng_key
    )

    # Check alpha initialization (gating factors)
    assert model.alpha_pre == 0.01, "alpha_pre should be initialized to 0.01"
    assert model.alpha_post == 0.01, "alpha_post should be initialized to 0.01"
    assert model.alpha_res == 0.01, "alpha_res should be initialized to 0.01"

    # Check bias initialization (should be zeros)
    assert jnp.all(model.b_pre == 0.0), "b_pre should be initialized to zeros"
    assert jnp.all(model.b_post == 0.0), "b_post should be initialized to zeros"
    assert jnp.all(model.b_res == 0.0), "b_res should be initialized to zeros"


@pytest.mark.unit
def test_mhc_initialization_shapes(rng_key):
    """
    Verify that all weight matrices and biases have correct shapes.
    """
    n_streams = 3
    dim = 16
    dummy_layer = lambda x: x

    model = ManifoldConstrainedHyperConnection(
        layer_f=dummy_layer,
        n_streams=n_streams,
        dim=dim,
        key=rng_key
    )

    input_dim = n_streams * dim

    # Check linear layer shapes
    assert model.phi_pre.weight.shape == (n_streams, input_dim)
    assert model.phi_post.weight.shape == (n_streams, input_dim)
    assert model.phi_res.weight.shape == (n_streams**2, input_dim)

    # Check bias shapes
    assert model.b_pre.shape == (n_streams,)
    assert model.b_post.shape == (n_streams,)
    assert model.b_res.shape == (n_streams, n_streams)


# ============================================================================
# ManifoldConstrainedHyperConnection Forward Pass Tests
# ============================================================================

@pytest.mark.unit
def test_mhc_forward_pass_shape(rng_key):
    """
    Verifies the forward pass runs and preserves the (n, C) shape.
    """
    n_streams = 4
    dim = 32

    # Create a simple Linear layer as the inner function F
    # F expects (dim,) -> (dim,)
    linear_layer = eqx.nn.Linear(dim, dim, key=rng_key)

    mhc = ManifoldConstrainedHyperConnection(
        layer_f=linear_layer,
        n_streams=n_streams,
        dim=dim,
        key=rng_key
    )

    # Input is (n, C)
    x_input = jax.random.normal(rng_key, (n_streams, dim))

    # Run forward
    x_out = mhc(x_input, key=rng_key)

    # Check shape
    assert x_out.shape == (n_streams, dim), f"Expected shape {(n_streams, dim)}, got {x_out.shape}"
    # Check it's not returning NaNs
    assert not jnp.any(jnp.isnan(x_out)), "Output contains NaN values"
    # Check it's finite
    assert jnp.all(jnp.isfinite(x_out)), "Output contains non-finite values"


@pytest.mark.unit
def test_mhc_different_input_sizes(rng_key):
    """
    Test mHC with different numbers of streams and dimensions.
    """
    test_configs = [
        (2, 16),
        (4, 32),
        (8, 64),
        (3, 24),  # Non-power-of-2 sizes
    ]

    for n_streams, dim in test_configs:
        linear_layer = eqx.nn.Linear(dim, dim, key=rng_key)

        mhc = ManifoldConstrainedHyperConnection(
            layer_f=linear_layer,
            n_streams=n_streams,
            dim=dim,
            key=rng_key
        )

        x_input = jax.random.normal(rng_key, (n_streams, dim))
        x_out = mhc(x_input, key=rng_key)

        assert x_out.shape == (n_streams, dim)
        assert not jnp.any(jnp.isnan(x_out))


@pytest.mark.unit
def test_mhc_weights_in_valid_range(rng_key):
    """
    Test that the computed weights (pre, post, res) are in valid ranges.
    - h_pre_weights should be in [0, 1] (sigmoid output)
    - h_post_weights should be in [0, 2] (2 * sigmoid output)
    - h_res_matrix should be doubly stochastic (rows and cols sum to 1)
    """
    n_streams = 4
    dim = 32

    linear_layer = eqx.nn.Linear(dim, dim, key=rng_key)
    mhc = ManifoldConstrainedHyperConnection(
        layer_f=linear_layer,
        n_streams=n_streams,
        dim=dim,
        key=rng_key
    )

    x_input = jax.random.normal(rng_key, (n_streams, dim))

    # We'll need to inspect intermediate values, so let's replicate the forward pass
    x_flat = x_input.reshape((-1,))
    x_norm = mhc.rms_norm(x_flat)

    h_tilde_pre = mhc.alpha_pre * mhc.phi_pre(x_norm) + mhc.b_pre
    h_tilde_post = mhc.alpha_post * mhc.phi_post(x_norm) + mhc.b_post
    h_tilde_res_flat = mhc.alpha_res * mhc.phi_res(x_norm)
    h_tilde_res = h_tilde_res_flat.reshape((n_streams, n_streams)) + mhc.b_res

    h_pre_weights = jax.nn.sigmoid(h_tilde_pre)
    h_post_weights = 2 * jax.nn.sigmoid(h_tilde_post)
    h_res_matrix = sinkhorn_knopp(h_tilde_res, n_iters=20)

    # Check ranges
    assert jnp.all((h_pre_weights >= 0) & (h_pre_weights <= 1))
    assert jnp.all((h_post_weights >= 0) & (h_post_weights <= 2))

    # Check doubly stochastic property
    row_sums = jnp.sum(h_res_matrix, axis=1)
    col_sums = jnp.sum(h_res_matrix, axis=0)
    assert jnp.allclose(row_sums, 1.0, atol=1e-3)
    assert jnp.allclose(col_sums, 1.0, atol=1e-3)


# ============================================================================
# Gradient Flow Tests
# ============================================================================

@pytest.mark.gradient
def test_mhc_gradient_flow(rng_key):
    """
    Verifies that the module is differentiable and gradients flow properly.
    """
    n_streams = 2
    dim = 8

    layer = eqx.nn.Linear(dim, dim, key=rng_key)
    model = ManifoldConstrainedHyperConnection(layer, n_streams, dim, rng_key)

    x = jax.random.normal(rng_key, (n_streams, dim))
    rngkey, subkey= jax.random.split(rng_key)
    target = jax.random.normal(subkey, (n_streams, dim))

    @jax.grad
    def loss_fn(model, x):
        y = model(x)
        return jnp.sum(jnp.square(y-target))

    # Should calculate gradients without error
    grads = loss_fn(model, x)

    # phi_res gradient will now be non-zero because changing mixing affects the squared error
    assert jnp.linalg.norm(grads.phi_res.weight) > 1e-6, "phi_res gradient is zero!"
    assert jnp.linalg.norm(grads.phi_pre.weight) > 1e-6
    assert jnp.linalg.norm(grads.phi_post.weight) > 1e-6

    # Optional: Verify the row/col zero-sum property of Sinkhorn gradients (b_res)
    # This confirms the manifold constraint is active
    grad_b_res = grads.b_res
    row_sums = jnp.sum(grad_b_res, axis=1)
    assert jnp.allclose(row_sums, 0.0, atol=1e-5)


@pytest.mark.gradient
def test_mhc_gradient_finite(rng_key):
    """
    Test that all gradients are finite (no NaN or Inf).
    """
    n_streams = 4
    dim = 16

    layer = eqx.nn.Linear(dim, dim, key=rng_key)
    model = ManifoldConstrainedHyperConnection(layer, n_streams, dim, rng_key)

    x = jax.random.normal(rng_key, (n_streams, dim))

    def loss_fn(model, x):
        y = model(x)
        return jnp.sum(y**2)  # L2 loss

    grads = jax.grad(loss_fn)(model, x)

    # Collect all gradient arrays
    grad_arrays = [
        grads.phi_pre.weight,
        grads.phi_post.weight,
        grads.phi_res.weight,
        grads.b_pre,
        grads.b_post,
        grads.b_res,
        grads.alpha_pre,
        grads.alpha_post,
        grads.alpha_res,
    ]

    for grad_array in grad_arrays:
        assert jnp.all(jnp.isfinite(grad_array)), f"Gradient contains non-finite values"


@pytest.mark.gradient
def test_mhc_gradient_with_complex_layer(rng_key):
    """
    Test gradient flow with a more complex inner layer (MLP).
    """
    n_streams = 3
    dim = 32

    # Create a simple MLP as the layer function
    class SimpleMLP(eqx.Module):
        layers: list

        def __init__(self, dim, key):
            keys = jax.random.split(key, 2)
            self.layers = [
                eqx.nn.Linear(dim, dim * 2, key=keys[0]),
                eqx.nn.Linear(dim * 2, dim, key=keys[1]),
            ]

        def __call__(self, x):
            x = self.layers[0](x)
            x = jax.nn.relu(x)
            x = self.layers[1](x)
            return x

    mlp = SimpleMLP(dim, rng_key)
    model = ManifoldConstrainedHyperConnection(mlp, n_streams, dim, rng_key)

    x = jax.random.normal(rng_key, (n_streams, dim))

    def loss_fn(model, x):
        y = model(x)
        return jnp.mean(y**2)

    # Compute gradients
    grads = jax.grad(loss_fn)(model, x)

    # Check mHC-specific gradients exist
    assert grads.phi_pre.weight is not None
    assert grads.phi_res.weight is not None

    # Check inner layer gradients exist
    assert grads.layer_f.layers[0].weight is not None
    assert grads.layer_f.layers[1].weight is not None


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.integration
def test_mhc_residual_connection_property(rng_key):
    """
    Test that mHC properly combines residual and function branches.
    The output should be: branch_res + branch_fn
    """
    n_streams = 2
    dim = 8

    # Use identity function to simplify analysis
    identity_layer = lambda x: x

    model = ManifoldConstrainedHyperConnection(
        layer_f=identity_layer,
        n_streams=n_streams,
        dim=dim,
        key=rng_key
    )

    x_input = jax.random.normal(rng_key, (n_streams, dim))
    x_output = model(x_input)

    # Output should be different from input (due to weighting and mixing)
    assert not jnp.allclose(x_output, x_input)

    # But shape should be preserved
    assert x_output.shape == x_input.shape


@pytest.mark.integration
def test_mhc_jit_compilation(rng_key):
    """
    Test that mHC can be JIT compiled successfully.
    """
    n_streams = 4
    dim = 32

    layer = eqx.nn.Linear(dim, dim, key=rng_key)
    model = ManifoldConstrainedHyperConnection(layer, n_streams, dim, rng_key)

    x = jax.random.normal(rng_key, (n_streams, dim))

    # JIT compile the forward pass
    @jax.jit
    def forward(model, x):
        return model(x)

    # Should compile and run without error
    output = forward(model, x)

    assert output.shape == (n_streams, dim)
    assert not jnp.any(jnp.isnan(output))


@pytest.mark.integration
def test_mhc_vmap_compatibility(rng_key):
    """
    Test that mHC works with vmap for batch processing.
    """
    n_streams = 2
    dim = 16
    batch_size = 4

    layer = eqx.nn.Linear(dim, dim, key=rng_key)
    model = ManifoldConstrainedHyperConnection(layer, n_streams, dim, rng_key)

    # Create batched input
    x_batch = jax.random.normal(rng_key, (batch_size, n_streams, dim))

    # Apply vmap
    batched_forward = jax.vmap(lambda x: model(x), in_axes=0)

    # Should process batch without error
    output_batch = batched_forward(x_batch)

    assert output_batch.shape == (batch_size, n_streams, dim)
    assert not jnp.any(jnp.isnan(output_batch))
