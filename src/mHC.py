import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Array, PRNGKeyArray
from typing import Optional, Tuple, Any

@eqx.filter_jit
def sinkhorn_knopp(
        log_matrix: Float[Array, "seq n n"],
        n_iters: int = 20
        )-> Float[Array, "seq n n"]:
    matrix = jnp.exp(log_matrix)

    def loop_body(_, m):
        m = m / jnp.sum(m, axis=-2, keepdims=True) # Normalize col
        m = m / jnp.sum(m, axis=-1, keepdims=True) # Normalize row
        return m

    matrix = jax.lax.fori_loop(0, n_iters, loop_body, matrix)
    return matrix

class ManifoldConstrainedHyperConnection(eqx.Module):
    """
    Sequence-aware mHC wrapper that handles (Output, Cache) tuples from inner layers.
    """
    n_streams: int = eqx.field(static=True)
    dim: int = eqx.field(static=True)
    layer_f: eqx.Module

    phi_pre: eqx.nn.Linear
    phi_post: eqx.nn.Linear
    phi_res: eqx.nn.Linear

    b_pre: Float[Array, "n"]
    b_post: Float[Array, "n"]
    b_res : Float[Array, "n n"]

    alpha_pre: Float[Array, ""]
    alpha_post: Float[Array, ""]
    alpha_res: Float[Array, ""]

    rms_norm: eqx.nn.RMSNorm

    def __init__(
            self,
            layer_f: eqx.Module,
            n_streams: int,
            dim: int,
            key: PRNGKeyArray
            ):
        self.layer_f = layer_f
        self.n_streams = n_streams
        self.dim = dim

        k_pre, k_post, k_res = jax.random.split(key, 3)
        input_dim = n_streams * dim

        self.phi_pre = eqx.nn.Linear(input_dim, n_streams, use_bias=False, key=k_pre)
        self.phi_post = eqx.nn.Linear(input_dim, n_streams, use_bias=False, key=k_post)
        self.phi_res = eqx.nn.Linear(input_dim, n_streams**2, use_bias=False, key=k_res)

        self.b_pre = jnp.zeros((n_streams,))
        self.b_post = jnp.zeros((n_streams,))
        self.b_res = jnp.zeros((n_streams, n_streams))

        # Initialize alphas small as per paper
        self.alpha_pre = jnp.array(0.01)
        self.alpha_post = jnp.array(0.01)
        self.alpha_res = jnp.array(0.01)

        self.rms_norm = eqx.nn.RMSNorm(input_dim)

    def __call__(
            self,
            x_stream: Float[Array, "seq n dim"],
            **kwargs
            ) -> Tuple[Float[Array, "seq n dim"], Any]:
        """
        Args:
            x_stream: The expanded residual stream (seq, n, dim)
            **kwargs: Arguments passed to layer_f (e.g., mask, cache)
        """
        seq_len, n, d = x_stream.shape

        # 1. Flatten n streams for gating calculation
        x_flat = x_stream.reshape(seq_len, n * d)

        # 2. RMSNorm on the flattened stream
        # Note: vmap over sequence length
        x_norm = jax.vmap(self.rms_norm)(x_flat)

        # 3. Compute Dynamic Mappings (Seq, n)
        # alpha * Linear(x) + b
        h_tilde_pre = self.alpha_pre * jax.vmap(self.phi_pre)(x_norm) + self.b_pre
        h_tilde_post = self.alpha_post * jax.vmap(self.phi_post)(x_norm) + self.b_post

        h_tilde_res_flat = self.alpha_res * jax.vmap(self.phi_res)(x_norm)
        h_tilde_res = h_tilde_res_flat.reshape(seq_len, n, n) + self.b_res

        # 4. Manifold Projections
        h_pre_weights = jax.nn.sigmoid(h_tilde_pre)      # (seq, n)
        h_post_weights = 2 * jax.nn.sigmoid(h_tilde_post)# (seq, n)
        h_res_matrix = sinkhorn_knopp(h_tilde_res)       # (seq, n, n)

        # 5. Pre-Mapping: Aggregate streams to layer input
        # Sum_i (weight_i * stream_i) -> (seq, dim)
        layer_input = jnp.einsum("sn,snd->sd", h_pre_weights, x_stream)

        # 6. Execute Inner Layer (Attention or MLP)
        layer_out_raw = self.layer_f(layer_input, **kwargs)

        # Handle layers that return (output, cache) vs just output
        if isinstance(layer_out_raw, tuple):
            layer_out, new_cache = layer_out_raw
        else:
            layer_out, new_cache = layer_out_raw, None

        # 7. Post-Mapping: Broadcast layer output back to streams
        # weight_i * layer_out -> (seq, n, dim)
        branch_fn = jnp.einsum("sn,sd->snd", h_post_weights, layer_out)

        # 8. Residual Mapping: Mix streams
        # Matrix multiply (n,n) by (n, dim) per token
        branch_res = jnp.einsum("sij,sjd->sid", h_res_matrix, x_stream)

        # 9. Combine
        x_next = branch_res + branch_fn

        return x_next, new_cache
