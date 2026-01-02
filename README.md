# MambaSparseAttentionax

MambaSparseAttentionax is a JAX/Equinox implementation of a hybrid Transformer architecture. It combines State Space Models (Mamba2), Multi-Head Latent Attention (MLA), and Manifold-Constrained Hyper-Connections (mHC) to create a high-efficiency, multi-stream neural architecture.

## Architecture Overview

This repository implements a specialized transformer block, the SparseMambaTransformerBlock, which deviates from standard architectures in three key areas:

1. **Sparse Mamba-Indexed Attention**: Instead of a global dense attention matrix, the model uses a Mamba2-based indexer to compute importance scores, followed by a sparse selection mechanism.
2. **Multi-Head Latent Attention (MLA)**: Inspired by DeepSeek-V3, this uses low-rank bottlenecks for queries and KV pairs to significantly reduce the memory footprint of the KV cache while maintaining representational power.
3. **Manifold-Constrained Hyper-Connections (mHC)**: Residual connections are treated as multiple parallel streams. A learnable, doubly-stochastic matrix (projected via the Sinkhorn-Knopp algorithm) dynamically mixes these streams, allowing the model to route information across different computational tracks.

## Computational Efficiency

The architecture is designed to balance the parallel training strengths of Transformers with the efficient incremental inference of State Space Models.

### Training Efficiency
During training, the model achieves near-linear scaling with respect to sequence length:
* **Mamba2 Blocks**: Instead of sequential recurrence, these blocks utilize a parallel associative scan. This allows the model to compute the entire sequence state in O(L log L) or O(L) time, avoiding the sequential bottleneck of standard RNNs.
* **Sparse Attention**: By using a Mamba-based indexer to select only the Top-K relevant keys, the attention mechanism avoids the O(L^2) cost of global dense attention. The complexity is reduced to O(L * k), where k is the sparsity factor.
* **mHC Routing**: The manifold-constrained routing adds a small constant overhead per layer but remains O(L) in time and memory, as it operates on the flattened stream dimensions.

### Inference Efficiency
In autoregressive mode, the model transitions to a recurrent representation:
* **Constant Time Step**: Mamba2 blocks provide O(1) inference per token by maintaining a fixed-size recurrent state (convolutional history and SSM hidden state).
* **Compressed KV Cache**: MLA reduces the KV cache memory requirement by projecting keys and values into a low-rank latent space. This allows for significantly larger batch sizes or longer context windows compared to standard Multi-Head Attention.
* **Fast Scan**: The implementation utilizes jax.lax.scan for sequence generation, which lowers the overhead of the Python interpreter and allows XLA to optimize the generation loop as a single fused operation.

## Memory Optimization and Stability

To prevent memory explosions and numerical instability—common issues in long-sequence hybrid models—the following optimizations are implemented:

### Chunked Associative Scan
To avoid materializing the full (Seq, Heads, HeadDim, StateDim) state tensor, the Mamba2 implementation employs a chunked computation strategy. The sequence is divided into segments (default size 128), and the associative scan is performed within these chunks while carrying only the final state across boundaries. This bounds the intermediate memory usage to O(ChunkSize).

### Low-Rank Bottlenecks
Memory usage is optimized through latent space compression in the MLA module. Queries and Key-Value pairs are compressed into a bottleneck dimension significantly smaller than the model dimension. This is particularly effective for the KV cache, as the model stores latent vectors rather than full high-dimensional head states.

### Gradient Checkpointing
For long sequences, the attention and MLP layers within the mHC wrapper can be processed using gradient checkpointing. This discards intermediate activations during the forward pass and recomputes them during the backward pass, trading computation for a significant reduction in peak memory consumption.

### Numerical Precision Management
The repository employs a mixed-precision strategy:
* **Matrix Multiplications**: Performed in bfloat16 to leverage Tensor Cores and reduce memory bandwidth.
* **SSM Recurrence and Sinkhorn**: The core SSM state updates and the Sinkhorn-Knopp iterations for mHC are maintained in float32 to prevent vanishing or exploding gradients and to ensure the doubly-stochastic constraints of the residual streams are precisely met.

### Manifold Projections
The mHC module uses the Sinkhorn-Knopp algorithm to project the residual mixing weights onto the Birkhoff Polytope. This ensures that the total energy of the residual streams is preserved (rows and columns sum to 1), preventing latent representations from diverging in magnitude across depth.

## Installation

### Using Nix (Recommended)
This repository uses Nix Flakes to provide a reproducible development environment with CUDA 12 and CUDNN support.
1. Ensure Nix is installed with Flakes enabled.
2. Run:
   ```bash
   nix develop
   ```
This configures Python 3.13, JAX, CUDA paths, and LD_LIBRARY_PATH automatically.

### Manual Installation
Required dependencies:
* jax
* jaxlib (with CUDA support)
* equinox
* jaxtyping
* beartype
* optax

## Usage

### Basic Forward Pass
```python
import jax
import jax.numpy as jnp
from src.transformer import SparseMambaTransformerBlock

key = jax.random.PRNGKey(0)
dim = 128
n_streams = 4

block = SparseMambaTransformerBlock(
    dim=dim,
    n_streams=n_streams,
    num_heads=4,
    top_k=16,
    key=key
)

# Input shape: (sequence_length, num_streams, dimension)
x = jax.random.normal(key, (128, n_streams, dim))
output, _ = block(x)
```

### Autoregressive Inference
```python
# Initialize cache for a maximum sequence length
cache = block.init_cache(max_seq_len=1024)

# Process tokens one by one
for i in range(10):
    token = jax.random.normal(key, (1, n_streams, dim))
    token_out, cache = block(token, cache=cache)
```

## Testing

The repository contains a comprehensive test suite using pytest.
Run all tests:
```bash
pytest
```

## Repository Structure

* `src/`: Core implementation.
    * `mambax.py`: Mamba2 SSM with chunked scan.
    * `mhlax.py`: Multi-Head Latent Attention.
    * `mHC.py`: Manifold-Constrained Hyper-Connections.
    * `sma.py`: Sparse Mamba Attention.
    * `transformer.py`: Integrated Transformer block.
    * `ffn.py`: SwiGLU FFN implementation.
* `tests/`: Pytest suite.
* `transformer_example.py`: Demonstration script.
* `flake.nix`: Reproducible development environment.

## References

* Dao, T., & Gu, A. (2024). Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality.
* DeepSeek-AI. (2025). DeepSeek-V3 Technical Report.
* Arxiv 2512.24880: Manifold-Constrained Hyper-Connection.
