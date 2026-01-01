"""
Mamba2 State Space Model implementation in JAX/Equinox.

This module implements the Mamba2 architecture from "Transformers are SSMs:
Generalized Models and Efficient Algorithms Through Structured State Space Duality"
(Dao & Gu, 2024).

Key features:
- Chunked associative scan for memory-efficient sequence processing
- Input-dependent state space parameters (B, C, dt)
- Mixed precision computation (bfloat16 for matmuls, float32 for recurrence)
- Gradient checkpointing for long sequences
- Efficient inference with recurrent state caching
- Support for both single-token and scan-based inference
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import math
from jaxtyping import Float, Array, PRNGKeyArray
from typing import Optional, Tuple


def segsum_binary_op(
    e_i: Tuple[Float[Array, "..."], Float[Array, "..."]],
    e_j: Tuple[Float[Array, "..."], Float[Array, "..."]]
) -> Tuple[Float[Array, "..."], Float[Array, "..."]]:
    """
    Associative binary operator for parallel prefix scan in SSM computation.

    Implements the linear recurrence: h_t = A_t * h_{t-1} + u_t
    where each tuple element represents (decay_t, update_t).

    This operator enables parallel computation of the recurrence using
    jax.lax.associative_scan instead of sequential loops.

    Args:
        e_i: Tuple of (A_i, u_i) representing earlier state
        e_j: Tuple of (A_j, u_j) representing later state

    Returns:
        Tuple of (A_combined, u_combined) where:
        - A_combined = A_j * A_i (cumulative decay)
        - u_combined = A_j * u_i + u_j (accumulated update)
    """
    a_i, u_i = e_i
    a_j, u_j = e_j
    return a_j * a_i, a_j * u_i + u_j


class Mamba2InferenceCache(eqx.Module):
    """
    Cache for incremental inference in Mamba2.

    Stores the recurrent state needed for autoregressive generation:
    - conv_state: Recent history for causal convolution
    - ssm_state: Hidden state of the SSM recurrence

    Attributes:
        conv_state: Convolution cache of shape (d_conv_in, d_conv-1)
                   where d_conv_in = d_inner + 2 * ngroups * d_state
        ssm_state: SSM hidden state of shape (nheads, headdim, d_state)
    """
    conv_state: Float[Array, "d_conv_in d_conv_size"]
    ssm_state: Float[Array, "nheads headdim d_state"]

    def __init__(
        self,
        d_inner: int,
        d_conv: int,
        nheads: int,
        headdim: int,
        d_state: int,
        ngroups: int = 1,
    ):
        """
        Initialize cache with zeros.

        Args:
            d_inner: Inner dimension (d_model * expand)
            d_conv: Convolution kernel size
            nheads: Number of attention heads
            headdim: Dimension per head
            d_state: SSM state dimension
            ngroups: Number of groups for B, C parameters
        """
        d_conv_in = d_inner + 2 * ngroups * d_state
        self.conv_state = jnp.zeros((d_conv_in, d_conv - 1), dtype=jnp.float32)
        self.ssm_state = jnp.zeros((nheads, headdim, d_state), dtype=jnp.float32)


class Mamba2(eqx.Module):
    """
    Mamba2 State Space Model block with chunked computation and efficient inference.

    Implements the Mamba2 architecture which uses state space duality to achieve
    efficient sequence modeling. The model processes sequences in chunks to prevent
    memory explosion during the associative scan operation.

    Architecture overview:
    1. Combined input projection for all SSM parameters (x, z, B, C, dt)
    2. 1D convolution for local context (causal)
    3. State space computation with diagonal SSM per head
    4. Chunked associative scan for memory efficiency
    5. Output projection with gating and normalization

    Key differences from Mamba1:
    - Combined input projections instead of separate
    - Diagonal SSM structure (one per head) instead of shared
    - Input-dependent B, C broadcast across head groups
    - Chunked computation to handle long sequences
    - Efficient recurrent inference mode

    Attributes:
        in_proj: Projects input to [z, x, B, C, dt]
        out_proj: Projects output back to d_model
        conv1d: Causal 1D convolution for local context
        norm: RMSNorm for output normalization
        dt_bias: Learned bias for timestep parameter (nheads,)
        A_log: Log of diagonal SSM matrix (nheads,)
        D: Skip connection parameter (nheads,)
        d_model: Input/output dimension
        d_inner: Internal dimension (d_model * expand)
        d_state: SSM state dimension
        d_conv: Convolution kernel size
        headdim: Dimension per head
        nheads: Number of heads
        ngroups: Number of groups for B, C broadcasting
        activation: Activation function name
        chunk_size: Size of chunks for training (default 128)
    """

    in_proj: eqx.nn.Linear
    out_proj: eqx.nn.Linear
    conv1d: eqx.nn.Conv1d
    norm: eqx.nn.RMSNorm

    dt_bias: Float[Array, "nheads"]
    A_log: Float[Array, "nheads"]
    D: Float[Array, "nheads"]

    d_model: int = eqx.field(static=True)
    d_inner: int = eqx.field(static=True)
    d_state: int = eqx.field(static=True)
    d_conv: int = eqx.field(static=True)
    headdim: int = eqx.field(static=True)
    nheads: int = eqx.field(static=True)
    ngroups: int = eqx.field(static=True)
    activation: str = eqx.field(static=True)
    chunk_size: int = eqx.field(static=True)

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
        chunk_size: int = 128,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        A_init_range: Tuple[float, float] = (1, 16),
        *,
        key: PRNGKeyArray,
    ):
        """
        Initialize Mamba2 block.

        Args:
            d_model: Model dimension (input/output size)
            d_state: SSM state dimension (default: 128)
            d_conv: Convolution kernel size (default: 4)
            expand: Expansion factor for inner dimension (default: 2)
            headdim: Dimension per head (default: 64)
            ngroups: Number of groups for B, C parameters (default: 1)
            chunk_size: Chunk size for training (default: 128)
            dt_min: Minimum value for timestep initialization (default: 0.001)
            dt_max: Maximum value for timestep initialization (default: 0.1)
            dt_init_floor: Floor value for timestep after clipping (default: 1e-4)
            A_init_range: (min, max) range for A matrix initialization (default: (1, 16))
            key: PRNG key for random initialization
        """
        k_in, k_out, k_conv, k_dt, k_A = jax.random.split(key, 5)

        self.d_model = d_model
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.headdim = headdim
        self.ngroups = ngroups
        self.nheads = self.d_inner // headdim
        self.activation = "silu"
        self.chunk_size = chunk_size

        assert self.d_inner % self.headdim == 0, "d_inner must be divisible by headdim"
        assert self.nheads % self.ngroups == 0, "nheads must be divisible by ngroups"

        # 1. Projections
        # Input order: [z, x, B, C, dt]
        # Shapes:
        # z: d_inner
        # x: d_inner
        # B: ngroups * d_state
        # C: ngroups * d_state
        # dt: nheads
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = eqx.nn.Linear(d_model, d_in_proj, use_bias=False, key=k_in)

        # 2. Convolution (Conv1d on x, B, C concatenated)
        # Input to conv is size: d_inner + 2 * ngroups * d_state
        d_conv_in = self.d_inner + 2 * self.ngroups * self.d_state
        self.conv1d = eqx.nn.Conv1d(
            in_channels=d_conv_in,
            out_channels=d_conv_in,
            kernel_size=d_conv,
            groups=d_conv_in,
            padding=d_conv - 1,
            use_bias=True,
            key=k_conv
        )

        # 3. Output Projection
        self.out_proj = eqx.nn.Linear(self.d_inner, d_model, use_bias=False, key=k_out)
        self.norm = eqx.nn.RMSNorm(self.d_inner)

        # 4. Parameters (A, D, dt_bias)
        # Initialize dt bias
        dt = jnp.exp(
            jax.random.uniform(k_dt, (self.nheads,), minval=math.log(dt_min), maxval=math.log(dt_max))
        )
        dt = jnp.clip(dt, a_min=dt_init_floor)
        # Inverse softplus to store as bias
        self.dt_bias = dt + jnp.log(-jnp.expm1(-dt))

        # Initialize A
        A = jax.random.uniform(k_A, (self.nheads,), minval=A_init_range[0], maxval=A_init_range[1])
        self.A_log = jnp.log(A)

        # Initialize D
        self.D = jnp.ones(self.nheads)

    def __call__(
        self,
        u: Float[Array, "seq d_model"] | Float[Array, "d_model"],
        cache: Optional[Mamba2InferenceCache] = None,
        durations = None,
    ) -> tuple[Float[Array, "..."], Optional[Mamba2InferenceCache]]:
        """
        Forward pass supporting both training and inference modes.

        Training mode (cache=None):
            - Input shape: (seq_len, d_model)
            - Uses chunked associative scan
            - Returns: (output, None)

        Inference mode (cache provided):
            - Single token: input shape (d_model,)
            - Sequence: input shape (seq_len, d_model) with lax.scan
            - Uses recurrent state updates
            - Returns: (output, updated_cache)

        Args:
            u: Input sequence or single token
            cache: Optional cache for inference mode
            durations: Optional duration parameters for timestep modulation

        Returns:
            output: Model output
            new_cache: Updated cache (None in training mode)
        """
        is_inference = cache is not None

        if is_inference:
            # Inference mode
            if u.ndim == 1:
                # Single token inference
                return self._inference_step(u, cache, durations)
            else:
                # Sequence inference with scan
                return self._inference_scan(u, cache, durations)
        else:
            # Training mode
            assert u.ndim == 2, "Training mode expects sequence input (seq, d_model)"
            output = self._training_forward(u, durations)
            return output, None

    def _inference_step(
        self,
        u: Float[Array, "d_model"],
        cache: Mamba2InferenceCache,
        durations,
    ) -> tuple[Float[Array, "d_model"], Mamba2InferenceCache]:
        """
        Process single token with recurrent state.

        Implements the SSM recurrence:
            h_t = exp(dt * A) * h_{t-1} + dt * B * x_t
            y_t = C^T * h_t + D * x_t

        Args:
            u: Single input token of shape (d_model,)
            cache: Current recurrent state
            durations: Optional duration parameter

        Returns:
            output: Output token of shape (d_model,)
            new_cache: Updated cache
        """
        # 1. Input projection
        u = u.astype(jnp.bfloat16)
        zxbcdt = self.in_proj(u)

        dim_z = self.d_inner
        dim_x = self.d_inner
        dim_B = self.ngroups * self.d_state
        dim_C = self.ngroups * self.d_state

        z, x, B, C, dt = jnp.split(
            zxbcdt,
            [dim_z, dim_z + dim_x, dim_z + dim_x + dim_B, dim_z + dim_x + dim_B + dim_C],
        )

        # 2. Causal convolution update
        # Concatenate x, B, C to match training mode
        xBC = jnp.concatenate([x, B, C], axis=-1)

        # Apply convolution: weighted sum over window
        # conv_state contains [x[t-3], x[t-2], x[t-1]]
        # Current window is [conv_state, xBC] = [x[t-3], x[t-2], x[t-1], x[t]]
        conv_weights = self.conv1d.weight.squeeze(1)  # (d_conv_in, d_conv)
        conv_bias = self.conv1d.bias.squeeze()  # (d_conv_in,)

        xBC_conv = jnp.sum(
            cache.conv_state * conv_weights[:, :-1],
            axis=-1
        ) + xBC * conv_weights[:, -1] + conv_bias

        xBC_conv = jax.nn.silu(xBC_conv)

        # Update conv state: shift left and append current input
        new_conv_state = jnp.concatenate([
            cache.conv_state[:, 1:],  # Drop oldest
            xBC[:, None]  # Add newest
        ], axis=-1)

        # Split back into x, B, C
        x, B, C = jnp.split(xBC_conv, [dim_x, dim_x + dim_B], axis=-1)

        # 3. Reshape for SSM heads
        x = x.reshape(self.nheads, self.headdim)

        # 4. Broadcast B and C across head groups
        heads_per_group = self.nheads // self.ngroups

        B = B.reshape(self.ngroups, 1, self.d_state)
        B = jnp.broadcast_to(B, (self.ngroups, heads_per_group, self.d_state))
        B = B.reshape(self.nheads, self.d_state)

        C = C.reshape(self.ngroups, 1, self.d_state)
        C = jnp.broadcast_to(C, (self.ngroups, heads_per_group, self.d_state))
        C = C.reshape(self.nheads, self.d_state)

        # 5. SSM recurrence
        A = -jnp.exp(self.A_log)  # (nheads,)

        if durations is not None:
            dt_input = durations
            dt = jax.nn.softplus(dt_input + self.dt_bias)
        else:
            dt = jax.nn.softplus(dt + self.dt_bias)  # (nheads,)

        dA = jnp.exp(dt * A)  # Discretized A

        # State update: h_t = dA * h_{t-1} + dt * B * x_t
        dt_B = jnp.einsum("h,hn->hn", dt, B)  # (nheads, d_state)
        u_t = jnp.einsum("hn,hp->hpn", dt_B, x)  # (nheads, headdim, d_state)

        new_ssm_state = (
            dA[:, None, None] * cache.ssm_state + u_t
        ).astype(jnp.float32)

        # 6. Compute output: y = C^T * h_t + D * x_t
        y = jnp.einsum("hn,hpn->hp", C, new_ssm_state.astype(B.dtype))

        # Skip connection
        y = y + self.D[:, None] * x
        y = y.reshape(self.d_inner)

        # 7. Normalize, gate, and project to output
        y = self.norm(y) * jax.nn.silu(z)
        out = self.out_proj(y)

        # Create new cache with updated state
        new_cache = eqx.tree_at(
            lambda c: (c.conv_state, c.ssm_state),
            cache,
            (new_conv_state, new_ssm_state)
        )

        return out, new_cache

    def _inference_scan(
        self,
        u: Float[Array, "seq d_model"],
        cache: Mamba2InferenceCache,
        durations = None,
    ) -> tuple[Float[Array, "seq d_model"], Mamba2InferenceCache]:
        """
        Process sequence of tokens using lax.scan for efficient inference.

        This is much faster than a Python loop for generating sequences.

        Args:
            u: Input sequence of shape (seq_len, d_model)
            cache: Initial recurrent state
            durations: Optional duration parameters

        Returns:
            outputs: Output sequence of shape (seq_len, d_model)
            final_cache: Final state after processing sequence
        """
        def scan_fn(carry_cache, x_t):
            # Process single token
            output, new_cache = self._inference_step(x_t, carry_cache, None)
            return new_cache, output

        # Run scan over sequence
        final_cache, outputs = jax.lax.scan(scan_fn, cache, u)

        return outputs, final_cache

    def _training_forward(
        self,
        u: Float[Array, "seq d_model"],
        durations = None,
    ) -> Float[Array, "seq d_model"]:
        """
        Forward pass through Mamba2 block using chunked computation.

        Processes input sequence through:
        1. Input projection to get z, x, B, C, dt
        2. Causal convolution on x, B, C
        3. Chunked state space computation
        4. Output projection with gating

        The sequence is split into chunks to prevent memory
        explosion during the associative scan. State is carried across chunks.

        Args:
            u: Input sequence of shape (seq_len, d_model)
            durations: Optional duration parameters for timestep modulation

        Returns:
            Output sequence of shape (seq_len, d_model)

        Note:
            Current implementation requires seq_len to be divisible by chunk_size.
        """
        seq_len, _ = u.shape

        # Cast inputs to bfloat16 for matrix mults, but keep state logic precise later
        u = u.astype(jnp.bfloat16)

        # 1. Input Projection
        zxbcdt = jax.vmap(self.in_proj)(u)

        dim_z = self.d_inner
        dim_x = self.d_inner
        dim_B = self.ngroups * self.d_state
        dim_C = self.ngroups * self.d_state

        # Split
        z, x, B, C, dt = jnp.split(
            zxbcdt,
            [dim_z, dim_z + dim_x, dim_z + dim_x + dim_B, dim_z + dim_x + dim_B + dim_C],
            axis=-1
        )

        # 2. Convolution (Standard)
        xBC = jnp.concatenate([x, B, C], axis=-1)
        xBC_T = jnp.transpose(xBC, (1, 0))
        xBC_conv = self.conv1d(xBC_T)[..., :seq_len]  # Slice to seq_len
        xBC_conv = jax.nn.silu(jnp.transpose(xBC_conv, (1, 0)))
        x, B, C = jnp.split(xBC_conv, [dim_x, dim_x + dim_B], axis=-1)

        # Reshape for SSM
        x = x.reshape(seq_len, self.nheads, self.headdim)

        # Broadcast B and C
        heads_per_group = self.nheads // self.ngroups
        def broadcast_groups(tensor):
            t = tensor.reshape(seq_len, self.ngroups, 1, self.d_state)
            return jnp.broadcast_to(
                t, (seq_len, self.ngroups, heads_per_group, self.d_state)
            ).reshape(seq_len, self.nheads, self.d_state)

        B = broadcast_groups(B)
        C = broadcast_groups(C)

        # SSM Parameters
        A = -jnp.exp(self.A_log)  # (nheads)

        if durations is not None:
            dt_input = durations[:, None]
            dt = jax.nn.softplus(dt_input + self.dt_bias)
        else:
            dt = jax.nn.softplus(dt + self.dt_bias)  # (seq, nheads)

        dA = jnp.exp(dt * A)  # (seq, nheads)

        # Chunked SSM Computation
        # ------------------------
        # Split sequence into chunks to prevent memory explosion from
        # materializing (Seq, Head, HeadDim, StateDim) tensor all at once.
        # Use jax.lax.scan to loop over chunks, carrying state H_prev across chunks.

        assert seq_len % self.chunk_size == 0, f"Seq len must be divisible by chunk_size ({self.chunk_size})"
        num_chunks = seq_len // self.chunk_size

        # Reshape inputs into chunks: (NumChunks, ChunkSize, ...)
        def to_chunks(t):
            return t.reshape(num_chunks, self.chunk_size, *t.shape[1:])

        chunk_dA = to_chunks(dA)
        chunk_dt = to_chunks(dt)
        chunk_B = to_chunks(B)
        chunk_x = to_chunks(x)
        chunk_C = to_chunks(C)

        # Initialize state to zeros
        init_state = jnp.zeros((self.nheads, self.headdim, self.d_state), dtype=jnp.float32)

        @jax.checkpoint
        def process_chunk(carry_state, args):
            """
            Process a single chunk of the sequence.

            Steps:
            1. Compute update vectors u_t and decay factors
            2. Run associative scan within chunk (assuming zero initial state)
            3. Apply carried state from previous chunks
            4. Compute output by contracting state with C
            """
            c_dA, c_dt, c_B, c_x, c_C = args

            # Compute update vectors (bfloat16 -> float32 for stability)
            # u_t = dt_t * B_t * x_t represents the contribution of input at time t
            dt_B = jnp.einsum("sh, shn -> shn", c_dt, c_B)
            u_t = jnp.einsum("shn, shp -> shpn", dt_B, c_x).astype(jnp.float32)
            decay_t = c_dA[:, :, None, None].astype(jnp.float32)

            # Intra-chunk associative scan
            # Computes state evolution within chunk, starting from zero
            _, scan_out = jax.lax.associative_scan(
                segsum_binary_op, (decay_t, u_t), axis=0
            )

            # Apply cross-chunk state carry
            # Compute cumulative decay to properly weight the carried state
            cumul_decay = jax.lax.associative_scan(
                lambda a, b: a * b, decay_t, axis=0
            )

            # Final state = intra-chunk state + decayed carry from previous chunks
            actual_state = scan_out + (cumul_decay * carry_state[None, ...])

            # Compute output: contract state with C projection
            y_chunk = jnp.einsum("shpn, shn -> shp", actual_state.astype(c_C.dtype), c_C)

            # New carry is the final state of this chunk
            new_carry = actual_state[-1]

            return new_carry, y_chunk

        # Process all chunks sequentially with state carryover
        final_state, y_chunks = jax.lax.scan(
            process_chunk,
            init_state,
            (chunk_dA, chunk_dt, chunk_B, chunk_x, chunk_C)
        )

        # Reshape chunks back to full sequence
        y = y_chunks.reshape(seq_len, self.nheads, self.headdim)

        # Skip connection and output projection
        y = y + (self.D[None, :, None] * x)  # Add skip connection
        y = y.reshape(seq_len, self.d_inner)

        # Normalize, gate with z, and project to output
        y = jax.vmap(self.norm)(y) * jax.nn.silu(z)
        out = jax.vmap(self.out_proj)(y)

        return out

    def init_cache(self) -> Mamba2InferenceCache:
        """
        Create an empty cache for inference.

        Returns:
            Initialized cache with zero states
        """
        return Mamba2InferenceCache(
            d_inner=self.d_inner,
            d_conv=self.d_conv,
            nheads=self.nheads,
            headdim=self.headdim,
            d_state=self.d_state,
            ngroups=self.ngroups,
        )


# --- Example Usage ---

if __name__ == "__main__":
    print("=" * 80)
    print("Mamba2 Complete Implementation Demo")
    print("=" * 80)

    key = jax.random.PRNGKey(0)
    key_model, key_data = jax.random.split(key)

    # Configuration
    batch_size = 2
    seq_len = 256
    d_model = 128

    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Model dimension: {d_model}")

    # Initialize model
    model = Mamba2(
        d_model=d_model,
        d_state=64,
        headdim=64,
        expand=2,
        chunk_size=128,
        key=key_model
    )

    print(f"\nModel initialized with:")
    print(f"  Inner dimension: {model.d_inner}")
    print(f"  Number of heads: {model.nheads}")
    print(f"  State dimension: {model.d_state}")
    print(f"  Chunk size: {model.chunk_size}")

    # === TRAINING MODE ===
    print("\n" + "=" * 80)
    print("TRAINING MODE")
    print("=" * 80)

    x_train = jax.random.normal(key_data, (batch_size, seq_len, d_model))
    print(f"\nInput shape: {x_train.shape}")

    # Forward pass (batched with vmap)
    out_train, _ = jax.vmap(model)(x_train)
    print(f"Output shape: {out_train.shape}")

    # Compute gradients
    loss_fn = lambda m, x: jnp.mean(jax.vmap(m)(x)[0] ** 2)
    grad_fn = jax.jit(jax.value_and_grad(loss_fn))
    loss, grads = grad_fn(model, x_train)
    print(f"\nLoss: {loss:.6f}")
    print("✓ Gradients computed successfully")

    # === INFERENCE MODE ===
    print("\n" + "=" * 80)
    print("INFERENCE MODE")
    print("=" * 80)

    # 1. Single token inference
    print("\n1. Single Token Inference:")
    cache = model.init_cache()
    single_token = jax.random.normal(key_data, (d_model,))

    output, cache = model(single_token, cache=cache)
    print(f"   Input shape: {single_token.shape}")
    print(f"   Output shape: {output.shape}")
    print("   ✓ Single token processed")

    # 2. Sequential inference with Python loop
    print("\n2. Sequential Inference (Python loop):")
    cache = model.init_cache()
    test_seq_len = 50
    tokens = jax.random.normal(key_data, (test_seq_len, d_model))

    outputs_loop = []
    for i in range(test_seq_len):
        output, cache = model(tokens[i], cache=cache)
        outputs_loop.append(output)
    outputs_loop = jnp.stack(outputs_loop)
    print(f"   Processed {test_seq_len} tokens")
    print(f"   Output shape: {outputs_loop.shape}")

    # 3. Sequential inference with lax.scan (fast!)
    print("\n3. Sequential Inference (lax.scan - optimized):")
    cache = model.init_cache()

    outputs_scan, final_cache = model(tokens, cache=cache)
    print(f"   Input shape: {tokens.shape}")
    print(f"   Output shape: {outputs_scan.shape}")

    # Verify they match
    max_diff = jnp.max(jnp.abs(outputs_loop - outputs_scan))
    print(f"   Max difference from loop: {max_diff:.2e}")
    assert jnp.allclose(outputs_loop, outputs_scan, atol=1e-5), "Outputs don't match!"
    print("   ✓ Scan and loop outputs match")

    # === PERFORMANCE COMPARISON ===
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)

    # Compile both methods
    @jax.jit
    def inference_loop(model, tokens):
        cache = model.init_cache()
        outputs = []
        for i in range(tokens.shape[0]):
            output, cache = model(tokens[i], cache=cache)
            outputs.append(output)
        return jnp.stack(outputs)
    @jax.jit
    def inference_scan(model, tokens):
        cache = model.init_cache()
        outputs, _ = model(tokens, cache=cache)
        return outputs

# Warm up
    _ = inference_loop(model, tokens[:10])
    _ = inference_scan(model, tokens[:10])

# Time both methods
    import time

    n_runs = 100
    test_tokens = jax.random.normal(key_data, (100, d_model))

# Loop timing
    start = time.time()
    for _ in range(n_runs):
        _ = inference_loop(model, test_tokens).block_until_ready()
    loop_time = (time.time() - start) / n_runs

# Scan timing
    start = time.time()
    for _ in range(n_runs):
        _ = inference_scan(model, test_tokens).block_until_ready()
    scan_time = (time.time() - start) / n_runs

    print(f"\nInference on {test_tokens.shape[0]} tokens:")
    print(f"  Python loop: {loop_time*1000:.2f} ms")
    print(f"  lax.scan:    {scan_time*1000:.2f} ms")
    print(f"  Speedup:     {loop_time/scan_time:.2f}x")

    # === VALIDATION: Training vs Inference Mode ===
    print("\n" + "=" * 80)
    print("VALIDATION: Training vs Inference Mode")
    print("=" * 80)
    print("\nComparing outputs from training mode (chunked scan) vs inference mode (recurrence)...")

    # Use a shorter sequence that's divisible by chunk_size
    val_seq_len = 128  # Must be divisible by chunk_size (128)
    key_val = jax.random.PRNGKey(42)
    val_sequence = jax.random.normal(key_val, (val_seq_len, d_model))

    # 1. Training mode (chunked associative scan)
    print(f"\n1. Training mode (chunked associative scan):")
    print(f"   Input shape: {val_sequence.shape}")
    output_training, _ = model(val_sequence)  # No cache = training mode
    print(f"   Output shape: {output_training.shape}")

    # 2. Inference mode (recurrent state updates)
    print(f"\n2. Inference mode (recurrent state updates):")
    cache = model.init_cache()
    output_inference, _ = model(val_sequence, cache=cache)
    print(f"   Output shape: {output_inference.shape}")

    # 3. Compare outputs
    print(f"\n3. Comparing outputs:")
    max_abs_diff = jnp.max(jnp.abs(output_training - output_inference))
    mean_abs_diff = jnp.mean(jnp.abs(output_training - output_inference))
    rel_error = max_abs_diff / (jnp.max(jnp.abs(output_training)) + 1e-8)

    print(f"   Max absolute difference: {max_abs_diff:.6e}")
    print(f"   Mean absolute difference: {mean_abs_diff:.6e}")
    print(f"   Relative error: {rel_error:.6e}")

    # Check if outputs are close
    # Using relaxed tolerance since we use bfloat16 for computations
    tolerance = 5e-4
    if jnp.allclose(output_training, output_inference, atol=tolerance):
        print(f"   ✓ Outputs match within tolerance ({tolerance:.1e})")
        print(f"   Training mode (chunked scan) and inference mode (recurrence) are consistent!")
    else:
        print(f"   ✗ WARNING: Outputs differ by more than tolerance ({tolerance:.1e})")
        print(f"   This could indicate an implementation mismatch between modes.")

    # Show a few sample values for visual inspection
    print(f"\n4. Sample output values (first 3 timesteps, first 5 dims):")
    print(f"   Training mode:")
    print(f"   {output_training[:3, :5]}")
    print(f"   Inference mode:")
    print(f"   {output_inference[:3, :5]}")

    print("\n" + "=" * 80)
    print("Demo completed successfully!")
    print("=" * 80)
