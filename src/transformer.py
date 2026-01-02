
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Array, PRNGKeyArray, Bool
from typing import Optional, Tuple, NamedTuple, Any

from .sma import SparseMambaAttax, SparseMambaInferenceCache
from .mHC import ManifoldConstrainedHyperConnection
from .ffn import SwiGLUMLP


class SparseMambaBlockInferenceCache(NamedTuple):
    attn_cache: Any # SparseMambaInferenceCache

class SparseMambaTransformerBlock(eqx.Module):
    """
    Transformer block where:
    1. Attention is SparseMambaAttax (Mamba2 Indexer + MLA)
    2. FFN is SwiGLU
    3. Residual connections are managed by mHC (Manifold Constrained Hyper-Connections)
    """
    attn_mhc: ManifoldConstrainedHyperConnection
    mlp_mhc: ManifoldConstrainedHyperConnection

    def __init__(
        self,
        dim: int,
        n_streams: int,
        # SparseMamba Args
        num_heads: int,
        top_k: int,
        indexer_dim: int = 64,
        # MLP Args
        mlp_ratio: int = 4,
        *,
        key: PRNGKeyArray
    ):
        k_attn, k_mlp, k_mhc_1, k_mhc_2 = jax.random.split(key, 4)

        # 1. Define Attention Layer
        # Note: q_low_rank/kv_low_rank/rope_dim usually scaled with dim/heads
        head_dim = dim // num_heads
        rope_dim = head_dim // 2

        attn_layer = SparseMambaAttax(
            embed_dim=dim,
            num_heads=num_heads,
            top_k=top_k,
            q_low_rank=max(128, dim // 4),
            kv_low_rank=512,
            rope_dim=rope_dim,
            v_head_dim=head_dim,
            indexer_dim=indexer_dim,
            key=k_attn
        )

        # 2. Wrap Attention in mHC
        self.attn_mhc = ManifoldConstrainedHyperConnection(
            layer_f=attn_layer,
            n_streams=n_streams,
            dim=dim,
            key=k_mhc_1
        )

        # 3. Define MLP Layer
        mlp_hidden = int(dim * mlp_ratio * 2/3) # Standard LLaMA/DeepSeek scaling
        mlp_layer = SwiGLUMLP(
            dim=dim,
            hidden_dim=mlp_hidden,
            key=k_mlp
        )

        # 4. Wrap MLP in mHC
        self.mlp_mhc = ManifoldConstrainedHyperConnection(
            layer_f=mlp_layer,
            n_streams=n_streams,
            dim=dim,
            key=k_mhc_2
        )

    def __call__(
        self,
        x_stream: Float[Array, "seq n dim"],
        mask: Optional[Bool[Array, "seq seq"]] = None,
        cache: Optional[SparseMambaBlockInferenceCache] = None,
        key: Optional[PRNGKeyArray] = None,
    ) -> Tuple[Float[Array, "seq n dim"], Optional[SparseMambaBlockInferenceCache]]:

        # Unpack cache
        attn_sub_cache = cache.attn_cache if cache is not None else None

        # 1. Attention Block (wrapped in mHC)
        # Note: SparseMambaAttax has built-in causal masking via the indexer, so we don't pass mask
        x_stream, new_attn_sub_cache = self.attn_mhc(
            x_stream,
            cache=attn_sub_cache,
            key=key
        )

        # 2. MLP Block (wrapped in mHC)
        # MLP is stateless, ignores cache
        x_stream, _ = self.mlp_mhc(x_stream)

        # Reconstruct cache
        new_cache = None
        if cache is not None:
            new_cache = SparseMambaBlockInferenceCache(attn_cache=new_attn_sub_cache)

        return x_stream, new_cache

    def init_cache(self, max_seq_len: int) -> SparseMambaBlockInferenceCache:
        """Initialize caches for the inner SparseMambaAttax layer."""
        # Access the inner layer of the mHC wrapper
        attn_layer: SparseMambaAttax = self.attn_mhc.layer_f
        return SparseMambaBlockInferenceCache(
            attn_cache=attn_layer.init_cache(max_seq_len)
        )
