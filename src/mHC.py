import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Int, Array, PRNGKeyArray, jaxtyped
from beartype import beartype as typechecker
from typing import Optional, Union, List, Callable
import wadler_lindig as wl

import pytest


@eqx.filter_jit
@jaxtyped(typechecker=typechecker)
def sinkhorn_knopp(
        log_matrix: Float[Array, "n n"],
        n_iters: int = 20
        )-> Float[Array, "n n"]:
    """
    Projects a matrix onto a Birkhoff polytope (doubly stochastic matrices).
    Ref: Arxiv 2512.24880v1 Section 4.2, Eq. (9)
    """
    matrix = jnp.exp(log_matrix)

    @jax.jit
    def loop_body(_, m):
        m = m / jnp.sum(m, axis=0, keepdims=True)
        m = m / jnp.sum(m, axis=1, keepdims=True)
        return m

    matrix = jax.lax.fori_loop(0, n_iters, loop_body, matrix)

    return matrix


class ManifoldConstrainedHyperConnection(eqx.Module):
    """
    mHC: Manifold-Constrained Hyper-Connection in JAX/Equinox
    https://www.arxiv.org/abs/2512.24880
    """

    n_streams: int = eqx.field(static=True)
    dim: int = eqx.field(static=True)
    layer_f: Callable

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
            layer_f: Callable,
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

        self.alpha_pre = jnp.array(0.01)
        self.alpha_post = jnp.array(0.01)
        self.alpha_res = jnp.array(0.01)

        self.rms_norm = eqx.nn.RMSNorm(input_dim)

    @jaxtyped(typechecker=typechecker)
    def __call__(
            self,
            x_l : Float[Array, "n C"],
            key: Optional[PRNGKeyArray] = None
            ) -> Float[Array, "n C"]:

        x_flat = x_l.reshape((-1,))

        x_norm = self.rms_norm(x_flat)

        # dynamic mapping
        h_tilde_pre = self.alpha_pre * self.phi_pre(x_norm) + self.b_pre
        h_tilde_post= self.alpha_post * self.phi_post(x_norm) + self.b_post

        h_tilde_res_flat = self.alpha_res * self.phi_res(x_norm)
        h_tilde_res = h_tilde_res_flat.reshape((self.n_streams, self.n_streams)) + self.b_res

        # manifold proj
        h_pre_weights = jax.nn.sigmoid(h_tilde_pre)

        h_post_weights = 2 * jax.nn.sigmoid(h_tilde_post)

        h_res_matrix = sinkhorn_knopp(h_tilde_res, n_iters=20)

        # residual mix
        branch_res = h_res_matrix @ x_l

        layer_input = jnp.einsum("n,nc -> c", h_pre_weights, x_l)

        layer_out = self.layer_f(layer_input)

        branch_fn = jnp.einsum("n,c -> nc", h_post_weights, layer_out)

        return branch_res + branch_fn
