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
from typing import Optional, Tuple


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


    def compute_attention_content(
        self,
        x: Float[Array, "seq embed"],
        mask: Optional[Bool[Array, "seq seq"]]
    ) -> Float[Array, "seq val_dim_total"]:
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

        Args:
            x: Input sequence of shape (seq_len, embed_dim)
            mask: Optional attention mask of shape (seq_len, seq_len)
                  True indicates positions that can be attended to

        Returns:
            Attention output of shape (seq_len, num_heads * v_head_dim)
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

        # Shared RoPE key projection (one per position, broadcast to all heads)
        k_rope_shared = jax.vmap(self.k_rope_proj)(x)[:, None, :]  # (seq, 1, rope_dim)
        sin, cos = _make_rotary_PE(seq_len, self.rope_dim)

        # Apply rotary embeddings to position components
        q_rope_rot = _apply_rotary_PE(q_rope, sin, cos)
        k_rope_rot_shared = _apply_rotary_PE(k_rope_shared, sin, cos)
        k_rope_rot = jnp.tile(k_rope_rot_shared, (1, self.num_heads, 1))

        # Concatenate content and position to form final Q and K
        q_final = jnp.concatenate([q_content, q_rope_rot], axis=-1)
        k_final = jnp.concatenate([k_content, k_rope_rot], axis=-1)

        # Pad values with zeros to match key dimension (v_head_dim + rope_dim)
        zeros_padding = jnp.zeros((seq_len, self.num_heads, self.rope_dim), dtype=v.dtype)
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

        return attn_out.reshape(seq_len, self.num_heads * self.v_head_dim)

    def __call__(
            self,
            x: Float[Array, "seq embed"],
            mask: Optional[Bool[Array, "seq seq"]] = None,
            key: Optional[PRNGKeyArray] = None
        ) -> Float[Array, "seq embed"]:
        """
        Forward pass through multi-head latent attention.

        Args:
            x: Input sequence of shape (seq_len, embed_dim)
            mask: Optional attention mask of shape (seq_len, seq_len)
                  True = can attend, False = cannot attend
            key: Optional PRNG key (unused, for interface compatibility)

        Returns:
            Output sequence of shape (seq_len, embed_dim)
        """
        # Compute attention with gradient checkpointing
        attn_flat = eqx.filter_checkpoint(self.compute_attention_content)(x, mask)

        # Project back to embedding dimension
        output = jax.vmap(self.out_proj)(attn_flat)
        return output

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
        out = model(x, mask)
        return jnp.sum(out)


    # Causal Mask
    # Shape (seq, seq). Lower triangle is True.
    mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))


    print("Running forward pass...")
    out = model(x, mask)

    grads = jax.grad(loss_fn)(model, x, mask)
    print(f"Output shape: {out.shape}")
    print(f"Grad shape: {grads}")

    assert out.shape == (seq_len, embed_dim), f"Shape Mismatch! Expected {(seq_len, embed_dim)}, got {out.shape}"
    print("✅ Success! Output shape matches input shape.")
