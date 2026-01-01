# MambaSparseAttentionax

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![JAX](https://img.shields.io/badge/JAX-0.8.0%2B-orange)](https://github.com/google/jax)
[![Nix](https://img.shields.io/badge/Built%20with-Nix-5277C3.svg?logo=nixos)](https://nixos.org)

A memory-efficient JAX/Equinox implementation of **sparse attention** using **Mamba2** as the "Lightning Indexer" component, inspired by [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556)'s sparse attention architecture.

## Key Features

- **Memory Efficient**: O(L·K) complexity instead of O(L²) for standard attention
- **Mamba2 Indexer**: Uses state-space models for intelligent token selection
- **Low-Rank MLA**: Multi-Head Latent Attention with compressed KV projections
- **RoPE Embeddings**: Rotary position encoding for better length generalization
- **Reproducible**: Nix flake for deterministic environment setup
- **Training Ready**: Gradient checkpointing for long sequences, bfloat16 precision

## Architecture

### Components

1. **`Mamba2`** (`src/mambax.py`): Efficient SSM block with chunked computation
2. **`MultiHeadLatentAttention`** (`src/mhlax.py`): Low-rank attention with RoPE
3. **`SparseMambaAttax`** (`SparseMambaAttax.py`): Sparse attention combining both components

```
Input → MambaIndexer → Top-K Selection → MLA (Sparse) → Output
         (Mamba2)         (Learned)      (Low-rank KV)
```

### Memory Complexity

| Component | Memory | Notes |
|-----------|--------|-------|
| Standard Attention | O(L²) | Full pairwise attention |
| **Sparse Attention** | **O(L·K)** | K << L selected tokens |
| Mamba2 SSM | O(L·D·N) | Chunked to prevent explosion |

Where L = sequence length, K = top-k tokens, D = model dimension, N = state dimension.

## Installation

### Option 1: Nix Flake (Recommended for Reproducibility)

This project includes a Nix flake with CUDA support for completely reproducible builds:

```bash
# Clone the repository
git clone https://github.com/yourusername/MambaSparseAttentionax.git
cd MambaSparseAttentionax

# Enter the development environment
nix develop

# Run examples
python SparseMambaAttax.py
```

The flake automatically configures:
- Python 3.13 with all dependencies
- CUDA toolkit (12.x) and cuDNN
- JAX with GPU support
- All required Python packages (equinox, jaxtyping, optax, etc.)

### Option 2: Manual Installation

```bash
# For CPU
pip install jax equinox jaxtyping

# For GPU (CUDA 12.x)
pip install jax[cuda12] equinox jaxtyping
```

## Quick Start

### Basic Usage

```python
import jax
import jax.numpy as jnp
from SparseMambaAttax import SparseMambaAttax

# Initialize model
key = jax.random.PRNGKey(0)
model = SparseMambaAttax(
    embed_dim=512,
    num_heads=8,
    top_k=32,          # Attend to top-32 tokens
    q_low_rank=128,
    kv_low_rank=128,
    rope_dim=32,
    v_head_dim=64,
    key=key
)

# Forward pass
x = jax.random.normal(key, (128, 512))  # (seq_len, embed_dim)
output = model(x)  # (128, 512)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
```

### Training Example

```python
import optax

# Define loss function
def loss_fn(model, x, labels):
    logits = model(x)
    return jnp.mean((logits - labels) ** 2)

# Compute gradients
grad_fn = jax.value_and_grad(loss_fn)
loss, grads = grad_fn(model, x, labels)

# Update with optimizer
optimizer = optax.adam(1e-4)
opt_state = optimizer.init(model)
updates, opt_state = optimizer.update(grads, opt_state)
model = optax.apply_updates(model, updates)
```

### Batch Processing

```python
# Process batches with vmap
batch_size = 4
x_batch = jax.random.normal(key, (batch_size, 128, 512))

# Vectorize over batch dimension
output_batch = jax.vmap(model)(x_batch)  # (4, 128, 512)
```

### Using Individual Components

```python
# Standalone Mamba2 block
from src.mambax import Mamba2

mamba = Mamba2(d_model=512, d_state=128, headdim=64, expand=2, key=key)
out = mamba(x)

# Standalone Multi-Head Latent Attention
from src.mhlax import MultiHeadLatentAttention

mla = MultiHeadLatentAttention(
    embed_dim=512,
    num_heads=8,
    q_low_rank=128,
    kv_low_rank=128,
    rope_dim=32,
    v_head_dim=64,
    key=key
)

# Create causal mask
seq_len = x.shape[0]
mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
out = mla(x, mask=mask)
```

## Implementation Details

### Mamba2 Block

- **Chunked Computation**: Sequences split into 128-token chunks to prevent memory explosion during the associative scan
- **Associative Scan**: Parallel prefix scan using `jax.lax.associative_scan` for efficient recurrence computation
- **Mixed Precision**: bfloat16 for matrix multiplications, float32 for SSM state updates for numerical stability
- **Gradient Checkpointing**: Automatic checkpointing with `jax.checkpoint` for memory-efficient training
- **Causal Processing**: Maintains autoregressive properties for language modeling tasks

### Sparse Attention Mechanism

The sparse attention mechanism works in three stages:

1. **Indexing**: Mamba2 processes the input sequence to generate importance scores for each token pair
2. **Selection**: Top-K most relevant tokens are selected using `jax.lax.top_k` for each query position
3. **Attention**: Multi-Head Latent Attention is computed only over the selected K tokens

**Key advantages:**
- Learned sparsity pattern adapts to input content
- Maintains causal masking for autoregressive generation
- Significantly reduces memory footprint for long sequences

### Multi-Head Latent Attention (MLA)

- **Low-Rank Compression**: Queries and Keys/Values are projected through low-rank bottlenecks
- **Shared RoPE Keys**: Rotary embeddings shared across heads for efficiency
- **Separate Content and Position**: Content vectors and position encodings handled separately
- **Value Padding**: Values padded to match key dimensionality during attention computation

## Performance Characteristics

### Memory Usage

Approximate memory consumption for `seq_len=4096`, `d_model=512`, `num_heads=8`:

- **Standard Attention**: ~8GB (storing full attention matrix)
- **Sparse Attention (K=64)**: ~1GB (only selected tokens)
- **Memory savings**: Approximately 8x reduction

Memory scales as:
- Standard: `O(batch_size * seq_len² * sizeof(float))`
- Sparse: `O(batch_size * seq_len * top_k * sizeof(float))`


## Project Structure

```
MambaSparseAttentionax/
├── src/
│   ├── mambax.py          # Mamba2 SSM implementation
│   └── mhlax.py           # Multi-Head Latent Attention
├── SparseMambaAttax.py                # Sparse attention main module
├── flake.nix              # Nix development environment
├── README.md              # This file
└── LICENSE                # Apache 2.0 license
```

## Development

### Running Tests

Each module includes basic sanity checks in the `__main__` block:

```bash
# Enter Nix shell
nix develop

# Run sanity check
python src/mambax.py
python src/mhlax.py
python SparseMambaAttax.py
```

### Code Organization

- **Type Annotations**: All functions use `jaxtyping` for precise array shape specifications
- **Equinox Modules**: Pure functional modules with immutable state
- **Static Fields**: Shape parameters marked with `eqx.field(static=True)` for JIT compilation

## Limitations and Future Work

### Current Limitations

- **No KV Caching**: Current implementation recomputes attention for all tokens during inference
- **Fixed Chunk Size**: Mamba2 uses hardcoded 128-token chunks (sequence length must be divisible)
- **Training Focus**: Optimized for training; autoregressive generation not yet optimized

### Planned Improvements

- [ ] KV caching for efficient autoregressive inference
- [ ] Comprehensive benchmarking suite
- [ ] Variable sequence lengths with padding
- [ ] Multi-GPU support using `jax.pmap` or `jax.sharding`
- [ ] Attention visualization tools

## References

This implementation is inspired by and builds upon:

- **DeepSeek-V3.2** - DeepSeek-AI (2025): [Pushing the Frontier of Open Large Language Models](https://arxiv.org/abs/2512.02556)
- **Mamba2** - Dao & Gu (2024): [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060)
- **Equinox** - Kidger & Garcia (2021): [Neural networks in JAX via callable PyTrees and filtered transformations](https://github.com/patrick-kidger/equinox)
- **RoPE** - Su et al. (2021): [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)

## Citation

If you use this code in your research, please cite the original papers:

```bibtex
@misc{deepseekai2025deepseekv32,
  title={DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models}, 
  author={DeepSeek-AI and Aixin Liu and Aoxue Mei and Bangcai Lin and Bing Xue and Bingxuan Wang and Bingzheng Xu and Bochao Wu and Bowei Zhang and Chaofan Lin and Chen Dong and Chengda Lu and Chenggang Zhao and Chengqi Deng and Chenhao Xu and Chong Ruan and Damai Dai and Daya Guo and Dejian Yang and Deli Chen and Erhang Li and Fangqi Zhou and Fangyun Lin and Fucong Dai and Guangbo Hao and Guanting Chen and Guowei Li and H. Zhang and Hanwei Xu and Hao Li and Haofen Liang and Haoran Wei and Haowei Zhang and Haowen Luo and Haozhe Ji and Honghui Ding and Hongxuan Tang and Huanqi Cao and Huazuo Gao and Hui Qu and Hui Zeng and Jialiang Huang and Jiashi Li and Jiaxin Xu and Jiewen Hu and Jingchang Chen and Jingting Xiang and Jingyang Yuan and Jingyuan Cheng and Jinhua Zhu and Jun Ran and Junguang Jiang and Junjie Qiu and Junlong Li and Junxiao Song and Kai Dong and Kaige Gao and Kang Guan and Kexin Huang and Kexing Zhou and Kezhao Huang and Kuai Yu and Lean Wang and Lecong Zhang and Lei Wang and Liang Zhao and Liangsheng Yin and Lihua Guo and Lingxiao Luo and Linwang Ma and Litong Wang and Liyue Zhang and M. S. Di and M. Y Xu and Mingchuan Zhang and Minghua Zhang and Minghui Tang and Mingxu Zhou and Panpan Huang and Peixin Cong and Peiyi Wang and Qiancheng Wang and Qihao Zhu and Qingyang Li and Qinyu Chen and Qiushi Du and Ruiling Xu and Ruiqi Ge and Ruisong Zhang and Ruizhe Pan and Runji Wang and Runqiu Yin and Runxin Xu and Ruomeng Shen and Ruoyu Zhang and S. H. Liu and Shanghao Lu and Shangyan Zhou and Shanhuang Chen and Shaofei Cai and Shaoyuan Chen and Shengding Hu and Shengyu Liu and Shiqiang Hu and Shirong Ma and Shiyu Wang and Shuiping Yu and Shunfeng Zhou and Shuting Pan and Songyang Zhou and Tao Ni and Tao Yun and Tian Pei and Tian Ye and Tianyuan Yue and Wangding Zeng and Wen Liu and Wenfeng Liang and Wenjie Pang and Wenjing Luo and Wenjun Gao and Xi Gao and Xiangwen Wang and Xiao Bi and Xiaodong Liu and Xiaohan Wang and Xiaokang Chen and Xiaokang Zhang and Xiaotao Nie and Xin Cheng and Xin Liu and Xin Xie and Xingchao Liu and Xingkai Yu and Xingyou Li and Xinyu Yang and Xinyuan Li and Xu Chen and Xuecheng Su and Xuehai Pan and Xuheng Lin and Xuwei Fu and Y. Q. Wang and Yang Zhang and Yanhong Xu and Yanru Ma and Yao Li and Yao Li and Yao Zhao and Yaofeng Sun and Yaohui Wang and Yi Qian and Yi Yu and Yichao Zhang and Yifan Ding and Yifan Shi and Yiliang Xiong and Ying He and Ying Zhou and Yinmin Zhong and Yishi Piao and Yisong Wang and Yixiao Chen and Yixuan Tan and Yixuan Wei and Yiyang Ma and Yiyuan Liu and Yonglun Yang and Yongqiang Guo and Yongtong Wu and Yu Wu and Yuan Cheng and Yuan Ou and Yuanfan Xu and Yuduan Wang and Yue Gong and Yuhan Wu and Yuheng Zou and Yukun Li and Yunfan Xiong and Yuxiang Luo and Yuxiang You and Yuxuan Liu and Yuyang Zhou and Z. F. Wu and Z. Z. Ren and Zehua Zhao and Zehui Ren and Zhangli Sha and Zhe Fu and Zhean Xu and Zhenda Xie and Zhengyan Zhang and Zhewen Hao and Zhibin Gou and Zhicheng Ma and Zhigang Yan and Zhihong Shao and Zhixian Huang and Zhiyu Wu and Zhuoshu Li and Zhuping Zhang and Zian Xu and Zihao Wang and Zihui Gu and Zijia Zhu and Zilin Li and Zipeng Zhang and Ziwei Xie and Ziyi Gao and Zizheng Pan and Zongqing Yao and Bei Feng and Hui Li and J. L. Cai and Jiaqi Ni and Lei Xu and Meng Li and Ning Tian and R. J. Chen and R. L. Jin and S. S. Li and Shuang Zhou and Tianyu Sun and X. Q. Li and Xiangyue Jin and Xiaojin Shen and Xiaosha Chen and Xinnan Song and Xinyi Zhou and Y. X. Zhu and Yanping Huang and Yaohui Li and Yi Zheng and Yuchen Zhu and Yunxian Ma and Zhen Huang and Zhipeng Xu and Zhongyu Zhang and Dongjie Ji and Jian Liang and Jianzhong Guo and Jin Chen and Leyi Xia and Miaojun Wang and Mingming Li and Peng Zhang and Ruyi Chen and Shangmian Sun and Shaoqing Wu and Shengfeng Ye and T. Wang and W. L. Xiao and Wei An and Xianzu Wang and Xiaowen Sun and Xiaoxiang Wang and Ying Tang and Yukun Zha and Zekai Zhang and Zhe Ju and Zhen Zhang and Zihua Qu},
  year={2025},
  eprint={2512.02556},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2512.02556}
}

@article{mamba2024,
  title={Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality},
  author={Dao, Tri and Gu, Albert},
  journal={arXiv preprint arXiv:2405.21060},
  year={2024}
}

@article{kidger2021equinox,
  title={{E}quinox: neural networks in {JAX} via callable {P}y{T}rees and filtered transformations},
  author={Patrick Kidger and Cristian Garcia},
  year={2021},
  journal={Differentiable Programming workshop at Neural Information Processing Systems 2021}
}

@article{rope2021,
  title={RoFormer: Enhanced Transformer with Rotary Position Embedding},
  author={Su, Jianlin and Lu, Yu and Pan, Shengfeng and Murtadha, Ahmed and Wen, Bo and Liu, Yunfeng},
  journal={arXiv preprint arXiv:2104.09864},
  year={2021}
}
```

## Contributing

Contributions are welcome! Please see areas for improvement listed above. Some guidelines:

- Follow existing code style (type hints, docstrings, etc.)
- Add tests for new functionality
- Update documentation as needed
- Ensure code runs in the Nix environment

For major changes, please open an issue to discuss the proposed changes.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Built with:
- [JAX](https://github.com/google/jax) - High-performance numerical computing
- [Equinox](https://github.com/patrick-kidger/equinox) - Elegant neural networks in JAX
- [Nix](https://nixos.org) - Reproducible development environments

Special thanks to the authors of Mamba2 and DeepSeek-V3.2 for their innovative architectures.
