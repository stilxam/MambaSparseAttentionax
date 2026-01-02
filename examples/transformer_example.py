"""
Example usage of SparseMambaTransformerBlock.

This script demonstrates:
1. Basic forward pass with multi-stream input
2. Inference mode with caching
3. Autoregressive generation
4. Gradient computation
5. Multiple stacked blocks (mini transformer)
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import time
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.transformer import SparseMambaTransformerBlock, SparseMambaBlockInferenceCache


def example_1_basic_forward_pass():
    """Example 1: Basic forward pass with multi-stream residual."""
    print("=" * 80)
    print("EXAMPLE 1: Basic Forward Pass")
    print("=" * 80)

    # Configuration
    SEQ_LEN = 128
    DIM = 128
    N_STREAMS = 4
    NUM_HEADS = 4
    TOP_K = 16

    key = jax.random.PRNGKey(42)
    key_model, key_data = jax.random.split(key)

    # Create a single transformer block
    block = SparseMambaTransformerBlock(
        dim=DIM,
        n_streams=N_STREAMS,
        num_heads=NUM_HEADS,
        top_k=TOP_K,
        indexer_dim=64,
        mlp_ratio=4,
        key=key_model
    )

    # Create multi-stream input: (seq_len, n_streams, dim)
    x_stream = jax.random.normal(key_data, (SEQ_LEN, N_STREAMS, DIM))

    print(f"Input shape: {x_stream.shape}")
    print(f"  - Sequence length: {SEQ_LEN}")
    print(f"  - Number of streams: {N_STREAMS}")
    print(f"  - Dimension per stream: {DIM}")

    # Forward pass
    output, _ = block(x_stream)

    print(f"\nOutput shape: {output.shape}")
    print(f"Output stats:")
    print(f"  - Mean: {jnp.mean(output):.6f}")
    print(f"  - Std: {jnp.std(output):.6f}")
    print(f"  - Min: {jnp.min(output):.6f}")
    print(f"  - Max: {jnp.max(output):.6f}")
    print("\n✓ Basic forward pass completed successfully!\n")


def example_2_with_causal_mask():
    """Example 2: Forward pass with causal attention mask."""
    print("=" * 80)
    print("EXAMPLE 2: Forward Pass with Causal Mask")
    print("=" * 80)

    SEQ_LEN = 128
    DIM = 64
    N_STREAMS = 2

    key = jax.random.PRNGKey(123)
    key_model, key_data = jax.random.split(key)

    block = SparseMambaTransformerBlock(
        dim=DIM,
        n_streams=N_STREAMS,
        num_heads=4,
        top_k=8,
        key=key_model
    )

    x_stream = jax.random.normal(key_data, (SEQ_LEN, N_STREAMS, DIM))

    # Create causal mask (lower triangular)
    causal_mask = jnp.tril(jnp.ones((SEQ_LEN, SEQ_LEN), dtype=bool))

    print(f"Input shape: {x_stream.shape}")
    print(f"Causal mask shape: {causal_mask.shape}")
    print(f"Causal mask (first 5x5):")
    print(causal_mask[:5, :5].astype(int))

    # Forward pass with mask
    output, _ = block(x_stream, mask=causal_mask)

    print(f"\nOutput shape: {output.shape}")
    print("✓ Forward pass with causal mask completed!\n")


def example_3_inference_caching():
    """Example 3: Inference mode with KV caching."""
    print("=" * 80)
    print("EXAMPLE 3: Inference Mode with Caching")
    print("=" * 80)

    SEQ_LEN = 128
    DIM = 64
    N_STREAMS = 2
    MAX_SEQ_LEN = 256

    key = jax.random.PRNGKey(456)
    key_model, key_data = jax.random.split(key)

    block = SparseMambaTransformerBlock(
        dim=DIM,
        n_streams=N_STREAMS,
        num_heads=4,
        top_k=8,
        key=key_model
    )

    # Initialize cache
    cache = block.init_cache(max_seq_len=MAX_SEQ_LEN)
    print(f"Cache initialized for max_seq_len={MAX_SEQ_LEN}")
    print(f"Cache type: {type(cache).__name__}")

    # Process a sequence with cache
    x_stream = jax.random.normal(key_data, (SEQ_LEN, N_STREAMS, DIM))
    output, new_cache = block(x_stream, cache=cache)

    print(f"\nProcessed sequence of length {SEQ_LEN}")
    print(f"Output shape: {output.shape}")
    print(f"Cache updated: {new_cache is not None}")

    if new_cache is not None and hasattr(new_cache.attn_cache, 'mla_cache'):
        mla_cache = new_cache.attn_cache.mla_cache
        if hasattr(mla_cache, 'position'):
            print(f"MLA cache position: {mla_cache.position}")

    print("\n✓ Inference caching completed!\n")


def example_4_autoregressive_generation():
    """Example 4: Autoregressive token-by-token generation."""
    print("=" * 80)
    print("EXAMPLE 4: Autoregressive Generation")
    print("=" * 80)

    DIM = 64
    N_STREAMS = 2
    NUM_TOKENS = 20
    MAX_SEQ_LEN = 100

    key = jax.random.PRNGKey(789)
    key_model = jax.random.PRNGKey(790)

    block = SparseMambaTransformerBlock(
        dim=DIM,
        n_streams=N_STREAMS,
        num_heads=4,
        top_k=8,
        key=key_model
    )

    # Initialize cache
    cache = block.init_cache(max_seq_len=MAX_SEQ_LEN)

    print(f"Generating {NUM_TOKENS} tokens autoregressively...")
    print(f"Each token: ({N_STREAMS}, {DIM}) -> expands to (1, {N_STREAMS}, {DIM})")

    generated_outputs = []

    for i in range(NUM_TOKENS):
        # Generate a random token (in practice, this would be from a previous layer)
        token_key = jax.random.fold_in(key, i)
        token = jax.random.normal(token_key, (N_STREAMS, DIM))

        # Expand to sequence dimension: (n_streams, dim) -> (1, n_streams, dim)
        token_seq = token[None, :, :]

        # Process with cache
        output, cache = block(token_seq, cache=cache)

        # Extract the single output token
        output_token = output[0]  # (n_streams, dim)
        generated_outputs.append(output_token)

        if (i + 1) % 5 == 0:
            print(f"  Generated {i + 1}/{NUM_TOKENS} tokens")

    # Stack all outputs
    all_outputs = jnp.stack(generated_outputs, axis=0)  # (num_tokens, n_streams, dim)

    print(f"\nGenerated sequence shape: {all_outputs.shape}")
    print("✓ Autoregressive generation completed!\n")


def example_5_gradient_computation():
    """Example 5: Compute gradients for training."""
    print("=" * 80)
    print("EXAMPLE 5: Gradient Computation")
    print("=" * 80)

    SEQ_LEN = 128
    DIM = 64
    N_STREAMS = 2

    key = jax.random.PRNGKey(111)
    key_model, key_data, key_target = jax.random.split(key, 3)

    block = SparseMambaTransformerBlock(
        dim=DIM,
        n_streams=N_STREAMS,
        num_heads=4,
        top_k=8,
        key=key_model
    )

    x_stream = jax.random.normal(key_data, (SEQ_LEN, N_STREAMS, DIM))
    target = jax.random.normal(key_target, (SEQ_LEN, N_STREAMS, DIM))

    print(f"Input shape: {x_stream.shape}")
    print(f"Target shape: {target.shape}")

    # Define a simple loss function
    def loss_fn(model, x, y):
        pred, _ = model(x)
        return jnp.mean((pred - y) ** 2)

    # Compute loss
    loss_value = loss_fn(block, x_stream, target)
    print(f"\nInitial loss: {loss_value:.6f}")

    # Compute gradients
    print("Computing gradients...")
    grads = jax.grad(loss_fn)(block, x_stream, target)

    # Check gradient statistics
    grad_leaves = jax.tree_util.tree_leaves(grads)
    grad_arrays = [g for g in grad_leaves if isinstance(g, jnp.ndarray)]

    all_finite = all(jnp.all(jnp.isfinite(g)) for g in grad_arrays)
    total_params = sum(g.size for g in grad_arrays)
    grad_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in grad_arrays))

    print(f"\nGradient statistics:")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - All gradients finite: {all_finite}")
    print(f"  - Gradient norm: {grad_norm:.6f}")
    print(f"  - Number of gradient arrays: {len(grad_arrays)}")

    if all_finite:
        print("\n✓ Gradient computation successful!\n")
    else:
        print("\n✗ Warning: Some gradients are NaN or Inf\n")


def example_6_stacked_blocks():
    """Example 6: Multiple stacked transformer blocks (mini transformer)."""
    print("=" * 80)
    print("EXAMPLE 6: Stacked Transformer Blocks")
    print("=" * 80)

    SEQ_LEN = 128
    DIM = 64
    N_STREAMS = 3
    NUM_LAYERS = 4

    key = jax.random.PRNGKey(222)
    keys = jax.random.split(key, NUM_LAYERS + 1)
    key_data = keys[0]
    layer_keys = keys[1:]

    # Create multiple stacked blocks
    blocks = [
        SparseMambaTransformerBlock(
            dim=DIM,
            n_streams=N_STREAMS,
            num_heads=4,
            top_k=8,
            key=k
        )
        for k in layer_keys
    ]

    print(f"Created {NUM_LAYERS} stacked transformer blocks")
    print(f"Input shape: ({SEQ_LEN}, {N_STREAMS}, {DIM})")

    # Input
    x_stream = jax.random.normal(key_data, (SEQ_LEN, N_STREAMS, DIM))

    # Forward pass through all layers
    activations = [x_stream]
    current = x_stream

    for i, block in enumerate(blocks):
        current, _ = block(current)
        activations.append(current)
        print(f"  Layer {i + 1} output shape: {current.shape}")

    # Analyze activation statistics per layer
    print(f"\nActivation statistics:")
    for i, act in enumerate(activations):
        print(f"  Layer {i}: mean={jnp.mean(act):.6f}, std={jnp.std(act):.6f}")

    print("\n✓ Stacked blocks forward pass completed!\n")


def example_7_performance_comparison():
    """Example 7: Compare performance with and without caching."""
    print("=" * 80)
    print("EXAMPLE 7: Performance Comparison (Caching vs No Caching)")
    print("=" * 80)

    DIM = 128
    N_STREAMS = 2
    NUM_TOKENS = 20  # Changed to work with batch processing
    WARMUP = 0  # Skip warmup for simplicity

    key = jax.random.PRNGKey(333)
    key_model, key_data = jax.random.split(key)

    block = SparseMambaTransformerBlock(
        dim=DIM,
        n_streams=N_STREAMS,
        num_heads=8,
        top_k=16,
        key=key_model
    )

    print(f"Generating {NUM_TOKENS} tokens (with {WARMUP} warmup tokens)")
    print(f"Model config: dim={DIM}, n_streams={N_STREAMS}")

    # Method 1: WITH caching (incremental)
    print("\n1. WITH caching (autoregressive with cache):")
    cache = block.init_cache(max_seq_len=512)

    # Benchmark - generate tokens one at a time
    start_cached = time.time()
    for i in range(NUM_TOKENS):
        token = jax.random.normal(jax.random.PRNGKey(2000 + i), (N_STREAMS, DIM))
        token_seq = token[None, :, :]
        _, cache = block(token_seq, cache=cache)
    end_cached = time.time()

    time_cached = end_cached - start_cached
    tokens_per_sec_cached = NUM_TOKENS / time_cached

    print(f"   Time: {time_cached:.4f}s")
    print(f"   Tokens/sec: {tokens_per_sec_cached:.2f}")

    # Method 2: WITHOUT caching (batch processing with fixed sequence length)
    print("\n2. WITHOUT caching (batch processing, seq_len=128):")

    # Process tokens in batches of 128 to satisfy Mamba2 chunking requirement
    # We'll process the same number of tokens but in batch mode
    num_batches = max(1, NUM_TOKENS // 128)
    tokens_per_batch = 128

    start_no_cache = time.time()
    for batch_idx in range(num_batches):
        # Create a batch of 128 tokens
        batch_tokens = []
        for i in range(tokens_per_batch):
            token = jax.random.normal(jax.random.PRNGKey(4000 + batch_idx * tokens_per_batch + i), (N_STREAMS, DIM))
            batch_tokens.append(token)
        batch_seq = jnp.stack(batch_tokens, axis=0)  # (128, n_streams, dim)
        _, _ = block(batch_seq)
    end_no_cache = time.time()

    total_tokens_no_cache = num_batches * tokens_per_batch

    time_no_cache = end_no_cache - start_no_cache
    tokens_per_sec_no_cache = total_tokens_no_cache / time_no_cache

    print(f"   Time: {time_no_cache:.4f}s")
    print(f"   Tokens/sec: {tokens_per_sec_no_cache:.2f}")
    print(f"   Total tokens processed: {total_tokens_no_cache}")
    print(f"   Batches: {num_batches} x {tokens_per_batch}")

    # Summary
    print(f"\n{'=' * 80}")
    print(f"Comparison:")
    print(f"  Cached (incremental): {NUM_TOKENS} tokens in {time_cached:.4f}s")
    print(f"  Batch (no cache): {total_tokens_no_cache} tokens in {time_no_cache:.4f}s")
    print(f"\nNote: This compares incremental inference (cache) vs batch processing.")
    print(f"Cache is beneficial for autoregressive generation (1 token at a time).")
    print(f"Batch processing is efficient for parallel token processing (training).")
    print(f"{'=' * 80}\n")


def main():
    """Run all examples."""
    print("\n")
    print("*" * 80)
    print("*" + " " * 78 + "*")
    print("*" + "  SparseMambaTransformerBlock Examples".center(78) + "*")
    print("*" + " " * 78 + "*")
    print("*" * 80)
    print("\n")

    examples = [
        ("Basic Forward Pass", example_1_basic_forward_pass),
        ("Forward Pass with Causal Mask", example_2_with_causal_mask),
        ("Inference Mode with Caching", example_3_inference_caching),
        ("Autoregressive Generation", example_4_autoregressive_generation),
        ("Gradient Computation", example_5_gradient_computation),
        ("Stacked Transformer Blocks", example_6_stacked_blocks),
        ("Performance Comparison", example_7_performance_comparison),
    ]

    for i, (name, func) in enumerate(examples, 1):
        try:
            func()
        except Exception as e:
            print(f"\n✗ Example {i} ({name}) failed with error:")
            print(f"  {type(e).__name__}: {e}\n")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
