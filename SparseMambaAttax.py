from src.mambax import Mamba2
from src.mhlax import MultiHeadLatentAttention, _make_rotary_PE, _apply_rotary_PE


import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Array, PRNGKeyArray, Bool, Int
from typing import Optional, Tuple

class MambaIndexer(eqx.Module):
    """
    Replaces the 'Lightning Indexer' with a Mamba2 block.
    """
    mamba_block: Mamba2
    q_proj: eqx.nn.Linear
    k_proj: eqx.nn.Linear

    scale: float = eqx.field(static=True)

    def __init__(
        self,
        d_model: int,
        indexer_dim: int,
        key: PRNGKeyArray
    ):
        k_mamba, k_q, k_k = jax.random.split(key, 3)

        self.mamba_block = Mamba2(
            d_model=d_model,
            d_state=64,
            expand=1,
            headdim=32,
            key=k_mamba
        )

        self.q_proj = eqx.nn.Linear(d_model, indexer_dim, use_bias=False, key=k_q)
        self.k_proj = eqx.nn.Linear(d_model, indexer_dim, use_bias=False, key=k_k)

        self.scale = indexer_dim ** -0.5

    def get_scores(self, x: Float[Array, "seq d_model"]) -> Float[Array, "seq seq"]:
        seq_len, _ = x.shape

        x_mamba = self.mamba_block(x)

        q_idx = jax.vmap(self.q_proj)(x_mamba)

        k_idx = jax.vmap(self.k_proj)(x)

        scores = jnp.einsum("ti, si -> ts", q_idx, k_idx) * self.scale

        mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
        scores = jnp.where(mask, scores, -jnp.inf)

        return scores


class SparseMambaAttax(eqx.Module):
    """
    DeepSeek Sparse Attention where the 'Indexer' is powered by Mamba2.
    """
    indexer: MambaIndexer
    mla: MultiHeadLatentAttention

    top_k: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    v_head_dim: int = eqx.field(static=True)
    rope_dim: int = eqx.field(static=True)

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        top_k: int,
        q_low_rank: int = 128,
        kv_low_rank: int = 128,
        rope_dim: int = 32,
        v_head_dim: int = 64,
        indexer_dim: int = 64,
        *,
        key: PRNGKeyArray
    ):
        k_idx, k_mla = jax.random.split(key)

        self.top_k = top_k
        self.num_heads = num_heads
        self.v_head_dim = v_head_dim
        self.rope_dim = rope_dim

        self.indexer = MambaIndexer(embed_dim, indexer_dim, key=k_idx)

        self.mla = MultiHeadLatentAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            q_low_rank=q_low_rank,
            kv_low_rank=kv_low_rank,
            rope_dim=rope_dim,
            v_head_dim=v_head_dim,
            key=k_mla
        )

    def _gather_kv(
        self,
        k: Float[Array, "seq heads dim_k"],
        v: Float[Array, "seq heads dim_v"],
        indices: Int[Array, "seq top_k"]
    ) -> Tuple[Float[Array, "seq top_k heads dim_k"], Float[Array, "seq top_k heads dim_v"]]:
        """
        Gathers K and V blocks based on the indices provided by the Indexer.
        """

        def gather_single_step(idx_row: Int[Array, "top_k"]):
            k_selected = jnp.take(k, idx_row, axis=0) # (top_k, heads, dim_k)
            v_selected = jnp.take(v, idx_row, axis=0) # (top_k, heads, dim_v)
            return k_selected, v_selected

        k_gathered, v_gathered = jax.vmap(gather_single_step)(indices)
        return k_gathered, v_gathered

    def __call__(
        self,
        x: Float[Array, "seq embed"],
        inference_params=None,
        key: Optional[PRNGKeyArray] = None
    ) -> Float[Array, "seq embed"]:

        seq_len, _ = x.shape

        idx_scores = self.indexer.get_scores(x) # (seq, seq)

        # Select Top-K indices
        k_val = min(self.top_k, seq_len)
        _, indices = jax.lax.top_k(idx_scores, k_val) # (seq, k)


        c_q = jax.vmap(self.mla.query_down_proj)(x)
        q_content = jax.vmap(self.mla.query_up_proj)(c_q).reshape(seq_len, self.num_heads, self.v_head_dim)
        q_rope = jax.vmap(self.mla.q_rope_proj)(c_q).reshape(seq_len, self.num_heads, self.rope_dim)

        c_kv = jax.vmap(self.mla.kv_down_proj)(x)
        kv_combined = jax.vmap(self.mla.kv_up_proj)(c_kv)
        k_content_flat, v_flat = jnp.split(kv_combined, 2, axis=-1)

        k_content = k_content_flat.reshape(seq_len, self.num_heads, self.v_head_dim)
        v = v_flat.reshape(seq_len, self.num_heads, self.v_head_dim)

        k_rope_shared = jax.vmap(self.mla.k_rope_proj)(x)[:, None, :] # (seq, 1, R)
        sin, cos = _make_rotary_PE(seq_len, self.rope_dim)

        q_rope_rot = _apply_rotary_PE(q_rope, sin, cos)
        k_rope_rot_shared = _apply_rotary_PE(k_rope_shared, sin, cos)
        k_rope_rot = jnp.tile(k_rope_rot_shared, (1, self.num_heads, 1))

        q_final = jnp.concatenate([q_content, q_rope_rot], axis=-1)
        k_final = jnp.concatenate([k_content, k_rope_rot], axis=-1)


        k_sparse, v_sparse = self._gather_kv(k_final, v, indices)

        q_curr = q_final[:, None, :, :]

        k_sparse_t = jnp.transpose(k_sparse, (0, 2, 1, 3))
        v_sparse_t = jnp.transpose(v_sparse, (0, 2, 1, 3))
        q_curr_t = jnp.transpose(q_curr, (0, 2, 1, 3))

        dim_k = k_final.shape[-1]

        attn_logits = jnp.matmul(q_curr_t, jnp.swapaxes(k_sparse_t, -1, -2))
        attn_logits = attn_logits * (dim_k ** -0.5)

        attn_weights = jax.nn.softmax(attn_logits, axis=-1).astype(v_sparse.dtype)

        attn_out = jnp.matmul(attn_weights, v_sparse_t)
        attn_out = attn_out.reshape(seq_len, self.num_heads * self.v_head_dim)

        output = jax.vmap(self.mla.out_proj)(attn_out)

        return output

if __name__ == "__main__":
    SEQ_LEN = 128
    D_MODEL = 64
    TOP_K = 16

    key = jax.random.PRNGKey(42)
    key_model, key_data = jax.random.split(key)

    model = SparseMambaAttax(
        embed_dim=D_MODEL,
        num_heads=4,
        top_k=TOP_K,
        q_low_rank=32,
        kv_low_rank=32,
        key=key_model
    )

    x = jax.random.normal(key_data, (SEQ_LEN, D_MODEL))
    y = model(x)

    print(f"Success. Input: {x.shape}, Output: {y.shape}")
    def loss_fn(m, x):
        out = m(x)
        return jnp.sum(out)

    grads = jax.grad(loss_fn)(model, x)
    assert jnp.all(grads != 0) and jnp.all(jnp.isfinite(grads))
    print("Gradients computed successfully (non-zero/non-nan).")
