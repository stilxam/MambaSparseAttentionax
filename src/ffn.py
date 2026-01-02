import jax
import equinox as eqx
from jaxtyping import Float, Array, PRNGKeyArray

class SwiGLUMLP(eqx.Module):
    w1: eqx.nn.Linear
    w2: eqx.nn.Linear
    w3: eqx.nn.Linear

    def __init__(self, dim: int, hidden_dim: int, key: PRNGKeyArray):
        k1, k2, k3 = jax.random.split(key, 3)
        self.w1 = eqx.nn.Linear(dim, hidden_dim, use_bias=False, key=k1)
        self.w2 = eqx.nn.Linear(dim, hidden_dim, use_bias=False, key=k2)
        self.w3 = eqx.nn.Linear(hidden_dim, dim, use_bias=False, key=k3)

    def __call__(self, x: Float[Array, "seq dim"], **kwargs) -> Float[Array, "seq dim"]:
        x_w1 = jax.vmap(self.w1)(x)
        x_w2 = jax.vmap(self.w2)(x)
        hidden = jax.nn.silu(x_w1) * x_w2
        return jax.vmap(self.w3)(hidden)
