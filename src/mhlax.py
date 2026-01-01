"""
Multi-Head Latent Attention (MLA) with Rotary Position Embeddings (RoPE).

This module implements low-rank multi-head attention inspired by DeepSeek-V3.2.
Key features:
- Low-rank query and key-value projections for memory efficiency
- Rotary Position Embeddings (RoPE) for position awareness
- Shared RoPE keys across attention heads
- Separate handling of content and positional information

Reference: "DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models"
(DeepSeek-AI, 2025)
"""

import jax
import jax.numpy as jnp
import equinox as eqx

from jaxtyping import Float, Int, Bool, PRNGKeyArray, Array
from typing import Optional, Tuple, NamedTuple


def _make_rotary_PE(seq_len: int, dim: int,)-> Tuple[Float[Array, "seq dim_half"], Float[Array, "seq dim_half"]]:
    """
    Generate sin and cos arrays for Rotary Position Embeddings (RoPE).

    Computes the frequency basis for rotary embeddings using the formula:
    freq_i = 1 / (10000^(2i/dim)) for i in [0, dim/2)

    Args:
        seq_len: Length of the sequence
        dim: Dimension of the embeddings (must be even)

    Returns:
        Tuple of (sin, cos) arrays, each of shape (seq_len, dim//2)
    """
    inv_freq = 1.0 / jnp.power(10_000, (jnp.arange(0, dim, 2)/dim))

    t = jnp.arange(seq_len)
    freqs = jnp.outer(t, inv_freq)
    return jnp.sin(freqs), jnp.cos(freqs)


def _apply_rotary_PE(
    x: Float[Array, "seq ... dim"],
    sin: Float[Array, "seq dim_half"],
    cos: Float[Array, "seq dim_half"]
) -> Float[Array, "seq ... dim"]:
    """
    Apply Rotary Position Embeddings to input tensor.

    Rotates pairs of dimensions using the formula:
    RoPE(x) = [x1*cos - x2*sin, x1*sin + x2*cos]

    This provides relative position information without absolute position tokens.

    Args:
        x: Input tensor of shape (seq_len, ..., dim) where dim must be even
        sin: Sine components of shape (seq_len, dim//2)
        cos: Cosine components of shape (seq_len, dim//2)

    Returns:
        Tensor of same shape as x with rotary embeddings applied
    """

    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]

    sin = jnp.concatenate([sin, sin], axis=-1)
    cos = jnp.concatenate([cos, cos], axis=-1)

    ndim_expand = x.ndim - 2

    sin_expanded = sin.reshape(sin.shape[0], *([1] * ndim_expand), sin.shape[1])
    cos_expanded = cos.reshape(cos.shape[0], *([1] * ndim_expand), cos.shape[1])

    rotated_x = jnp.concatenate((-x2, x1), axis=-1)
    return (x * cos_expanded) + (rotated_x * sin_expanded)


class MLAKVCache(NamedTuple):
    """
    Cache for KV pairs in Multi-Head Latent Attention inference mode.

    Stores accumulated key and value tensors for autoregressive generation.

    Attributes:
        keys: Cached keys of shape (cached_seq, num_heads, key_dim)
              where key_dim = v_head_dim + rope_dim
        values: Cached values of shape (cached_seq, num_heads, v_head_dim)
        position: Current position in the sequence (number of cached tokens)
    """
    keys: Float[Array, "cached_seq num_heads key_dim"]
    values: Float[Array, "cached_seq num_heads v_head_dim"]
    position: int


class MultiHeadLatentAttention(eqx.Module):
    """
    Multi-Head Latent Attention with low-rank compression and RoPE.

    Implements the MLA mechanism from DeepSeek-V3.2 which uses low-rank
    bottlenecks to reduce KV cache size while maintaining model quality.

    Architecture:
    1. Query path: embed → q_low_rank → [q_content, q_rope]
    2. KV path: embed → kv_low_rank → [k_content, v]
    3. Keys get additional shared RoPE projection
    4. Final keys = [k_content, k_rope] with RoPE applied
    5. Values padded with zeros to match key dimension

    Key features:
    - Low-rank bottlenecks reduce memory from O(L·H·D) to O(L·R)
      where L=seq_len, H=num_heads, D=head_dim, R=low_rank
    - RoPE keys shared across heads for efficiency
    - Separate content and position streams

    Attributes:
        query_down_proj: Projects query to low-rank (embed_dim → q_low_rank)
        query_up_proj: Projects to query content (q_low_rank → num_heads * v_head_dim)
        q_rope_proj: Projects to query RoPE (q_low_rank → num_heads * rope_dim)
        kv_down_proj: Projects KV to low-rank (embed_dim → kv_low_rank)
        kv_up_proj: Projects to keys and values (kv_low_rank → num_heads * 2 * v_head_dim)
        k_rope_proj: Shared key RoPE projection (embed_dim → rope_dim)
        out_proj: Output projection (num_heads * v_head_dim → embed_dim)
        num_heads: Number of attention heads
        rope_dim: Dimension of rotary embeddings
        v_head_dim: Dimension of value vectors per head
    """

    query_down_proj: eqx.nn.Linear
    query_up_proj: eqx.nn.Linear
    q_rope_proj: eqx.nn.Linear

    kv_down_proj: eqx.nn.Linear
    kv_up_proj: eqx.nn.Linear

    k_rope_proj: eqx.nn.Linear

    out_proj: eqx.nn.Linear

    num_heads: int = eqx.field(static=True)
    rope_dim: int = eqx.field(static=True)
    v_head_dim: int = eqx.field(static=True)

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        q_low_rank: int,
        kv_low_rank: int,
        rope_dim: int,
        v_head_dim: int,
        *,
        key: PRNGKeyArray,
    ):
        """
        Initialize Multi-Head Latent Attention module.

        Args:
            embed_dim: Model embedding dimension
            num_heads: Number of attention heads
            q_low_rank: Query low-rank bottleneck dimension
            kv_low_rank: Key-Value low-rank bottleneck dimension
            rope_dim: Dimension for rotary position embeddings
            v_head_dim: Value dimension per head
            key: PRNG key for parameter initialization

        Note:
            Final key dimension will be v_head_dim + rope_dim.
            Values are padded with zeros to match this dimension.
        """
        keys = jax.random.split(key, 7)
        self.query_down_proj = eqx.nn.Linear(embed_dim, q_low_rank, use_bias=False,  key=keys[0])
        self.query_up_proj = eqx.nn.Linear(q_low_rank, num_heads * v_head_dim, use_bias=False, key=keys[1])
        self.q_rope_proj = eqx.nn.Linear(q_low_rank, num_heads * rope_dim, use_bias=False, key=keys[2])

        self.kv_down_proj = eqx.nn.Linear(embed_dim, kv_low_rank, use_bias=False, key=keys[3])
        self.kv_up_proj = eqx.nn.Linear(kv_low_rank, num_heads * (v_head_dim * 2), use_bias=False, key=keys[4])
        self.k_rope_proj = eqx.nn.Linear(embed_dim, rope_dim, use_bias=False, key=keys[5])

        self.out_proj = eqx.nn.Linear(num_heads * v_head_dim, embed_dim, use_bias=False, key=keys[6])

        self.num_heads = num_heads
        self.v_head_dim = v_head_dim
        self.rope_dim = rope_dim

    def init_cache(self, max_seq_len: int) -> MLAKVCache:
        """
        Initialize KV cache for inference mode.

        Args:
            max_seq_len: Maximum sequence length to support

        Returns:
            MLAKVCache with zero-initialized keys and values
        """
        key_dim = self.v_head_dim + self.rope_dim
        keys = jnp.zeros((max_seq_len, self.num_heads, key_dim), dtype=jnp.bfloat16)
        values = jnp.zeros((max_seq_len, self.num_heads, self.v_head_dim), dtype=jnp.bfloat16)
        return MLAKVCache(keys=keys, values=values, position=0)

    def compute_attention_content(
        self,
        x: Float[Array, "seq embed"],
        mask: Optional[Bool[Array, "seq seq"]],
        cache: Optional[MLAKVCache] = None
    ) -> Tuple[Float[Array, "seq val_dim_total"], Optional[MLAKVCache]]:
        """
        Compute multi-head attention with low-rank projections.

        Processing steps:
        1. Project queries through low-rank bottleneck
        2. Split into content and RoPE components
        3. Project keys/values through shared low-rank bottleneck
        4. Apply RoPE to position components
        5. Concatenate content and position for final Q, K
        6. Pad values to match key dimension
        7. Compute scaled dot-product attention

        If cache is provided:
        - Keys and values are appended to the cache
        - Attention is computed over all cached + current tokens
        - Returns updated cache

        Args:
            x: Input sequence of shape (seq_len, embed_dim)
            mask: Optional attention mask of shape (seq_len, seq_len)
                  True indicates positions that can be attended to
            cache: Optional KV cache for inference mode

        Returns:
            Tuple of:
            - Attention output of shape (seq_len, num_heads * v_head_dim)
            - Updated cache (or None if not using cache)
        """
        seq_len, _ = x.shape

        # Query path: embed → low_rank → [content, rope]
        c_q = jax.vmap(self.query_down_proj)(x)
        q_content = jax.vmap(self.query_up_proj)(c_q).reshape(seq_len, self.num_heads, self.v_head_dim)
        q_rope = jax.vmap(self.q_rope_proj)(c_q).reshape(seq_len, self.num_heads, self.rope_dim)

        # KV path: embed → low_rank → [k_content, v]
        c_kv = jax.vmap(self.kv_down_proj)(x)
        kv_combined = jax.vmap(self.kv_up_proj)(c_kv)
        k_content_flat, v_flat = jnp.split(kv_combined, 2, axis=-1)

        k_content = k_content_flat.reshape(seq_len, self.num_heads, self.v_head_dim)
        v = v_flat.reshape(seq_len, self.num_heads, self.v_head_dim)

        # Determine position offset for RoPE based on cache
        position_offset = cache.position if cache is not None else 0

        # Shared RoPE key projection (one per position, broadcast to all heads)
        k_rope_shared = jax.vmap(self.k_rope_proj)(x)[:, None, :]  # (seq, 1, rope_dim)
        sin, cos = _make_rotary_PE(position_offset + seq_len, self.rope_dim)

        # Slice sin/cos to current positions
        sin = sin[position_offset:position_offset + seq_len]
        cos = cos[position_offset:position_offset + seq_len]

        # Apply rotary embeddings to position components
        q_rope_rot = _apply_rotary_PE(q_rope, sin, cos)
        k_rope_rot_shared = _apply_rotary_PE(k_rope_shared, sin, cos)
        k_rope_rot = jnp.tile(k_rope_rot_shared, (1, self.num_heads, 1))

        # Concatenate content and position to form final Q and K
        q_final = jnp.concatenate([q_content, q_rope_rot], axis=-1)
        k_final = jnp.concatenate([k_content, k_rope_rot], axis=-1)

        # Handle KV caching
        new_cache = None
        if cache is not None:
            # Append new keys and values to cache
            k_cached = cache.keys.at[position_offset:position_offset + seq_len].set(k_final.astype(jnp.bfloat16))
            v_cached = cache.values.at[position_offset:position_offset + seq_len].set(v.astype(jnp.bfloat16))

            # Use cached K, V up to current position
            k_final = k_cached[:position_offset + seq_len]
            v = v_cached[:position_offset + seq_len]

            # Update cache position
            new_cache = MLAKVCache(
                keys=k_cached,
                values=v_cached,
                position=position_offset + seq_len
            )

        # Pad values with zeros to match key dimension (v_head_dim + rope_dim)
        zeros_padding = jnp.zeros((v.shape[0], self.num_heads, self.rope_dim), dtype=v.dtype)
        v_padded = jnp.concatenate([v, zeros_padding], axis=-1)

        # Prepare mask for attention (expand to 4D for broadcasting)
        if mask is not None:
            mask_4d = mask[None, None, :, :]
        else:
            mask_4d = None

        # Compute attention in bfloat16 for efficiency
        dtype_attn = jnp.bfloat16
        q_attn = q_final.astype(dtype_attn)
        k_attn = k_final.astype(dtype_attn)
        v_attn = v_padded.astype(dtype_attn)

        # JAX's dot_product_attention expects (Seq, Heads, Dim) for unbatched input
        attn_out_padded = jax.nn.dot_product_attention(
            query=q_attn,
            key=k_attn,
            value=v_attn,
            mask=mask_4d,
            implementation="cudnn"
        )

        # Remove padding from output (keep only first v_head_dim dimensions)
        attn_out = attn_out_padded.astype(dtype_attn)[..., :self.v_head_dim]

        return attn_out.reshape(seq_len, self.num_heads * self.v_head_dim), new_cache

    def __call__(
            self,
            x: Float[Array, "seq embed"],
            mask: Optional[Bool[Array, "seq seq"]] = None,
            cache: Optional[MLAKVCache] = None,
            key: Optional[PRNGKeyArray] = None
        ) -> Tuple[Float[Array, "seq embed"], Optional[MLAKVCache]]:
        """
        Forward pass through multi-head latent attention.

        Args:
            x: Input sequence of shape (seq_len, embed_dim)
            mask: Optional attention mask of shape (seq_len, seq_len)
                  True = can attend, False = cannot attend
            cache: Optional KV cache for inference mode
            key: Optional PRNG key (unused, for interface compatibility)

        Returns:
            Tuple of:
            - Output sequence of shape (seq_len, embed_dim)
            - Updated cache (or None if not using cache)
        """
        # Compute attention (with or without gradient checkpointing based on cache)
        if cache is None:
            attn_flat, new_cache = eqx.filter_checkpoint(self.compute_attention_content)(x, mask, cache)
        else:
            # Don't checkpoint in inference mode
            attn_flat, new_cache = self.compute_attention_content(x, mask, cache)

        # Project back to embedding dimension
        output = jax.vmap(self.out_proj)(attn_flat)
        return output, new_cache

if __name__ == "__main__":
    key = jax.random.PRNGKey(0)

    seq_len = 64
    embed_dim = 128
    num_heads = 4
    rope_dim = 16
    v_head_dim = 32

    # Note: K dimension will be v_head_dim + rope_dim = 48
    # V dimension is 32. We pad V to 48 internally.

    model = MultiHeadLatentAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        q_low_rank=64,
        kv_low_rank=64,
        rope_dim=rope_dim,
        v_head_dim=v_head_dim,
        key=key,
    )

    x = jax.random.normal(key, (seq_len, embed_dim))

    def loss_fn(model, x, mask):
        out, _ = model(x, mask)
        return jnp.sum(out)


    # Causal Mask
    # Shape (seq, seq). Lower triangle is True.
    mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))


    print("Running forward pass...")
    out, _ = model(x, mask)

    grads = jax.grad(loss_fn)(model, x, mask)
    print(f"Output shape: {out.shape}")
    print(f"Grad shape: {grads}")

    assert out.shape == (seq_len, embed_dim), f"Shape Mismatch! Expected {(seq_len, embed_dim)}, got {out.shape}"
    print("✅ Success! Output shape matches input shape.")

    # Test KV caching
    print("\nTesting KV caching...")
    max_seq = 128
    cache = model.init_cache(max_seq)

    # Process tokens one by one
    cache_test = cache
    for i in range(5):
        token = jax.random.normal(jax.random.PRNGKey(100 + i), (1, embed_dim))
        token_out, cache_test = model(token, cache=cache_test)
        print(f"  Token {i+1}: Input {token.shape} -> Output {token_out.shape}, Cache position: {cache_test.position}")

    print("✅ KV caching works!")

    # Benchmark: MLA with and without cache
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON: MLA with vs without KV Cache")
    print("="*60)

    import time

    num_tokens = 50
    warmup_tokens = 5

    print(f"\nGenerating {num_tokens} tokens autoregressively...")

    # Benchmark WITHOUT cache (recompute full sequence each time)
    print("\n1. WITHOUT KV Cache (recomputing full attention each step):")
    start_no_cache = time.time()

    generated_seq = []
    for i in range(warmup_tokens + num_tokens):
        token = jax.random.normal(jax.random.PRNGKey(200 + i), (1, embed_dim))
        generated_seq.append(token)

        # Concatenate all tokens so far
        full_seq = jnp.concatenate(generated_seq, axis=0)

        # Create causal mask for full sequence
        curr_len = full_seq.shape[0]
        mask_full = jnp.tril(jnp.ones((curr_len, curr_len), dtype=bool))

        # Compute attention over full sequence (no cache)
        out, _ = model(full_seq, mask=mask_full, cache=None)

        # We only care about the last token's output
        _ = out[-1]  # Force computation

        if i == warmup_tokens - 1:
            # Start timing after warmup
            generated_seq = generated_seq[-1:]  # Keep only last token
            start_no_cache = time.time()

    end_no_cache = time.time()
    time_no_cache = end_no_cache - start_no_cache
    tokens_per_sec_no_cache = num_tokens / time_no_cache

    print(f"   Time: {time_no_cache:.4f}s")
    print(f"   Tokens/sec: {tokens_per_sec_no_cache:.2f}")

    # Benchmark WITH cache (incremental attention)
    print("\n2. WITH KV Cache (incremental attention):")

    cache_bench = model.init_cache(max_seq_len=warmup_tokens + num_tokens + 10)

    # Warmup
    for i in range(warmup_tokens):
        token = jax.random.normal(jax.random.PRNGKey(300 + i), (1, embed_dim))
        out, cache_bench = model(token, cache=cache_bench)

    # Actual benchmark
    start_with_cache = time.time()

    for i in range(num_tokens):
        token = jax.random.normal(jax.random.PRNGKey(300 + warmup_tokens + i), (1, embed_dim))
        out, cache_bench = model(token, cache=cache_bench)
        _ = out  # Force computation

    end_with_cache = time.time()
    time_with_cache = end_with_cache - start_with_cache
    tokens_per_sec_with_cache = num_tokens / time_with_cache

    print(f"   Time: {time_with_cache:.4f}s")
    print(f"   Tokens/sec: {tokens_per_sec_with_cache:.2f}")

    # Summary
    speedup = time_no_cache / time_with_cache
    print("\n" + "="*60)
    print(f"SPEEDUP: {speedup:.2f}x faster with KV caching")
    print(f"Efficiency gain: {(1 - 1/speedup)*100:.1f}% reduction in computation")
    print("="*60)
