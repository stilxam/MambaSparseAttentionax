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




class Mamba2(eqx.Module):
    """
    Mamba2 State Space Model block with chunked computation.

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

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
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

        assert self.d_inner % self.headdim == 0, "d_inner must be divisible by headdim"

        # 1. Projections
        # Input order in PyTorch Mamba2: [z, x, B, C, dt]
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
            padding=d_conv - 1, # Causal padding handling happens in __call__ usually, but we set standard here
            use_bias=True,
            key=k_conv
        )

        # 3. Output Projection
        self.out_proj = eqx.nn.Linear(self.d_inner, d_model, use_bias=False, key=k_out)
        self.norm = eqx.nn.RMSNorm(self.d_inner)

        # 4. Parameters (A, D, dt_bias)
        # Initialize dt bias
        # dt = exp(uniform(log(dt_min), log(dt_max)))
        dt = jnp.exp(
            jax.random.uniform(k_dt, (self.nheads,), minval=math.log(dt_min), maxval=math.log(dt_max))
        )
        dt = jnp.clip(dt, a_min=dt_init_floor)
        # Inverse softplus to store as bias
        self.dt_bias = dt + jnp.log(-jnp.expm1(-dt))

        # Initialize A
        # A is usually [1, 16], stored as -log(A) effectively or just log(A) depending on formula.
        # PyTorch Mamba2: A = -exp(A_log)
        A = jax.random.uniform(k_A, (self.nheads,), minval=A_init_range[0], maxval=A_init_range[1])
        self.A_log = jnp.log(A)

        # Initialize D
        self.D = jnp.ones(self.nheads)

    def __call__(
        self,
        u: Float[Array, "seq d_model"],
        durations= None,
        inference_params=None
    ) -> Float[Array, "seq d_model"]:
        """
        Forward pass through Mamba2 block.

        Processes input sequence through:
        1. Input projection to get z, x, B, C, dt
        2. Causal convolution on x, B, C
        3. Chunked state space computation
        4. Output projection with gating

        The sequence is split into chunks of 128 tokens to prevent memory
        explosion during the associative scan. State is carried across chunks.

        Args:
            u: Input sequence of shape (seq_len, d_model)
            durations: Optional duration parameters for timestep modulation
            inference_params: Optional parameters for inference (not yet implemented)

        Returns:
            Output sequence of shape (seq_len, d_model)

        Note:
            Current implementation requires seq_len to be divisible by chunk_size (128).
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
        z, x, B, C, dt = jnp.split(zxbcdt, [dim_z, dim_z + dim_x, dim_z + dim_x + dim_B, dim_z + dim_x + dim_B + dim_C], axis=-1)

        # 2. Convolution (Standard)
        xBC = jnp.concatenate([x, B, C], axis=-1)
        xBC_T = jnp.transpose(xBC, (1, 0))
        xBC_conv = self.conv1d(xBC_T)[..., :seq_len] # Slice to seq_len
        xBC_conv = jax.nn.silu(jnp.transpose(xBC_conv, (1, 0)))
        x, B, C = jnp.split(xBC_conv, [dim_x, dim_x + dim_B], axis=-1)

        # Reshape for SSM
        x = x.reshape(seq_len, self.nheads, self.headdim)

        # Broadcast B and C
        heads_per_group = self.nheads // self.ngroups
        def broadcast_groups(tensor):
            t = tensor.reshape(seq_len, self.ngroups, 1, self.d_state)
            return jnp.broadcast_to(t, (seq_len, self.ngroups, heads_per_group, self.d_state)).reshape(seq_len, self.nheads, self.d_state)

        B = broadcast_groups(B)
        C = broadcast_groups(C)

        # SSM Parameters
        A = -jnp.exp(self.A_log) # (nheads)


        if durations is not None:
            dt_input = durations[:, None]
            dt = jax.nn.softplus(dt_input + self.dt_bias)
        else:
            dt = jax.nn.softplus(dt + self.dt_bias) # (seq, nheads)


        dA = jnp.exp(dt * A) # (seq, nheads)

        # Chunked SSM Computation
        # ------------------------
        # Split sequence into chunks to prevent memory explosion from
        # materializing (Seq, Head, HeadDim, StateDim) tensor all at once.
        # Use jax.lax.scan to loop over chunks, carrying state H_prev across chunks.

        chunk_size = 128
        assert seq_len % chunk_size == 0, "Seq len must be divisible by chunk_size (128)"
        num_chunks = seq_len // chunk_size

        # Reshape inputs into chunks: (NumChunks, ChunkSize, ...)
        def to_chunks(t): return t.reshape(num_chunks, chunk_size, *t.shape[1:])

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
# --- Example Usage ---

if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    key_model, key_data = jax.random.split(key)

    # Config
    B, L, D = 2, 256, 32
    model = Mamba2(d_model=D, d_state=16, headdim=16, expand=2, key=key_model)

    x = jax.random.normal(key_data, (B, L, D))

    out = jax.vmap(model)(x)

    loss_fn = lambda m, x: jnp.mean(jax.vmap(m)(x))
    grad_fn = jax.jit(jax.value_and_grad(loss_fn))
    loss, grads = grad_fn(model, x)
    print(f"Loss: {loss}")
    print("Gradients computed successfully.")
