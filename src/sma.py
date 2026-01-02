import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Array, PRNGKeyArray, Bool, Int
from typing import Optional, Tuple, NamedTuple

from .mambax import Mamba2, Mamba2InferenceCache
from .mhlax import MultiHeadLatentAttention, MLAKVCache, _make_rotary_PE, _apply_rotary_PE

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

    def get_scores(
        self,
        x: Float[Array, "seq d_model"],
        cache: Optional[Mamba2InferenceCache] = None
    ) -> Tuple[Float[Array, "seq seq"], Optional[Mamba2InferenceCache]]:
        seq_len, _ = x.shape

        x_mamba, new_cache = self.mamba_block(x, cache=cache)

        q_idx = jax.vmap(self.q_proj)(x_mamba)

        k_idx = jax.vmap(self.k_proj)(x)

        scores = jnp.einsum("ti, si -> ts", q_idx, k_idx) * self.scale

        mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
        scores = jnp.where(mask, scores, -jnp.inf)

        return scores, new_cache


class SparseMambaInferenceCache(NamedTuple):
    """Cache for SparseMambaAttax inference mode."""
    indexer_cache: Mamba2InferenceCache
    mla_cache: MLAKVCache


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

    def init_cache(self, max_seq_len: int = 2048) -> SparseMambaInferenceCache:
        """
        Initialize cache for inference mode.

        Args:
            max_seq_len: Maximum sequence length for KV cache

        Returns:
            SparseMambaInferenceCache with zero-initialized states
        """
        indexer_cache = self.indexer.mamba_block.init_cache()
        mla_cache = self.mla.init_cache(max_seq_len)
        return SparseMambaInferenceCache(indexer_cache=indexer_cache, mla_cache=mla_cache)

    def __call__(
        self,
        x: Float[Array, "seq embed"],
        cache: Optional[SparseMambaInferenceCache] = None,
        key: Optional[PRNGKeyArray] = None
    ) -> Tuple[Float[Array, "seq embed"], Optional[SparseMambaInferenceCache]]:

        seq_len, _ = x.shape

        # Get caches if provided
        indexer_cache = cache.indexer_cache if cache is not None else None
        mla_cache = cache.mla_cache if cache is not None else None

        idx_scores, new_indexer_cache = self.indexer.get_scores(x, cache=indexer_cache) # (seq, seq)

        # In inference mode with KV cache, use MLA's built-in caching
        if mla_cache is not None:
            # Use MLA's KV caching - no need for manual K/V computation
            # Create causal mask for attention
            total_seq = mla_cache.position + seq_len
            mask = jnp.tril(jnp.ones((seq_len, total_seq), dtype=bool))

            output, new_mla_cache = self.mla(x, mask=mask, cache=mla_cache)
        else:
            # Training mode - use sparse attention
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
            new_mla_cache = None

        # Create new cache if in inference mode
        new_cache = None
        if cache is not None:
            new_cache = SparseMambaInferenceCache(
                indexer_cache=new_indexer_cache,
                mla_cache=new_mla_cache
            )

        return output, new_cache

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
    y, _ = model(x)

    print(f"Success. Input: {x.shape}, Output: {y.shape}")

    # Test inference mode with caching (sequence)
    print("\nTesting inference mode with caching (sequence)...")
    cache = model.init_cache(max_seq_len=256)
    y_inf, new_cache = model(x, cache=cache)
    print(f"Inference mode: Input: {x.shape}, Output: {y_inf.shape}")
    print(f"Cache updated: {new_cache is not None}")
    print(f"MLA KV cache position: {new_cache.mla_cache.position}")

    # Test single-token inference (autoregressive generation)
    print("\nTesting single-token inference (autoregressive)...")
    cache = model.init_cache(max_seq_len=256)
    tokens = []
    for i in range(10):
        token = jax.random.normal(jax.random.PRNGKey(i), (D_MODEL,))
        # For single token, we need to expand to sequence
        token_seq = token[None, :]  # (1, D_MODEL)
        output, cache = model(token_seq, cache=cache)
        tokens.append(output)
        print(f"  Token {i+1}: Input {token.shape} -> Output {output.shape}, MLA KV pos: {cache.mla_cache.position}")

    print(f"Generated {len(tokens)} tokens with persistent cache")

    # Test gradient computation
    print("\nTesting gradient computation...")
    def loss_fn(m, x):
        out, _ = m(x)
        return jnp.sum(out)

    grads = jax.grad(loss_fn)(model, x)
    # Check gradients are finite by flattening the PyTree
    grad_leaves = jax.tree_util.tree_leaves(grads)
    all_finite = all(jnp.all(jnp.isfinite(g)) for g in grad_leaves if isinstance(g, jnp.ndarray))
    assert all_finite, "Gradients contain NaN or Inf values"
    print("Gradients computed successfully (all finite).")

    # Benchmark: Compare MLA caching benefits directly
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON: MLA KV Cache in SparseMambaAttax")
    print("="*60)

    import time

    # Test the MLA component directly to show cache benefits
    print("\nTesting MLA component performance...")

    num_gen_tokens = 40
    warmup_tokens = 5

    # Create a simpler model for MLA-only testing
    from .mhlax import MultiHeadLatentAttention

    key_mla_test = jax.random.PRNGKey(999)
    mla_test = MultiHeadLatentAttention(
        embed_dim=D_MODEL,
        num_heads=4,
        q_low_rank=32,
        kv_low_rank=32,
        rope_dim=32,
        v_head_dim=16,
        key=key_mla_test
    )

    print(f"\nGenerating {num_gen_tokens} tokens autoregressively (MLA only)...")

    # Method 1: WITHOUT cache - recompute attention over full sequence
    print("\n1. WITHOUT KV Cache (full sequence attention each step):")

    generated_tokens = []

    # Warmup
    for i in range(warmup_tokens):
        token = jax.random.normal(jax.random.PRNGKey(600 + i), (1, D_MODEL))
        generated_tokens.append(token)
        full_seq = jnp.concatenate(generated_tokens, axis=0)
        curr_len = full_seq.shape[0]
        mask = jnp.tril(jnp.ones((curr_len, curr_len), dtype=bool))
        out, _ = mla_test(full_seq, mask=mask)

    # Reset for actual benchmark
    generated_tokens = generated_tokens[-1:]
    start_no_cache = time.time()

    for i in range(num_gen_tokens):
        token = jax.random.normal(jax.random.PRNGKey(600 + warmup_tokens + i), (1, D_MODEL))
        generated_tokens.append(token)
        full_seq = jnp.concatenate(generated_tokens, axis=0)
        curr_len = full_seq.shape[0]
        mask = jnp.tril(jnp.ones((curr_len, curr_len), dtype=bool))
        out, _ = mla_test(full_seq, mask=mask)
        _ = out[-1]  # Only need last token

    end_no_cache = time.time()
    time_no_cache = end_no_cache - start_no_cache
    tokens_per_sec_no_cache = num_gen_tokens / time_no_cache

    print(f"   Time: {time_no_cache:.4f}s")
    print(f"   Tokens/sec: {tokens_per_sec_no_cache:.2f}")
    print(f"   Final sequence length: {len(generated_tokens)}")

    # Method 2: WITH KV cache - incremental attention
    print("\n2. WITH KV Cache (incremental attention):")

    cache_test = mla_test.init_cache(max_seq_len=warmup_tokens + num_gen_tokens + 10)

    # Warmup
    for i in range(warmup_tokens):
        token = jax.random.normal(jax.random.PRNGKey(700 + i), (1, D_MODEL))
        out, cache_test = mla_test(token, cache=cache_test)

    # Actual benchmark
    start_with_cache = time.time()

    for i in range(num_gen_tokens):
        token = jax.random.normal(jax.random.PRNGKey(700 + warmup_tokens + i), (1, D_MODEL))
        out, cache_test = mla_test(token, cache=cache_test)
        _ = out

    end_with_cache = time.time()
    time_with_cache = end_with_cache - start_with_cache
    tokens_per_sec_with_cache = num_gen_tokens / time_with_cache

    print(f"   Time: {time_with_cache:.4f}s")
    print(f"   Tokens/sec: {tokens_per_sec_with_cache:.2f}")
    print(f"   Cache position: {cache_test.position}")

    # Summary
    speedup = time_no_cache / time_with_cache
    print("\n" + "="*60)
    print(f"SPEEDUP: {speedup:.2f}x faster with KV caching")
    print(f"Efficiency gain: {(1 - 1/speedup)*100:.1f}% reduction in computation")
    print("="*60)
    print("\nKey insight:")
    print("  Without cache: O(n²) attention recomputed for growing sequence")
    print("  With cache: O(n) incremental updates to cached K/V")
    print(f"\nFull SparseMambaAttax cache components:")
    print(f"  - Mamba2 indexer cache: conv_state + ssm_state")
    print(f"  - MLA KV cache: keys + values")
