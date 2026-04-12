# RWKV-6 Recurrent Kernel Fix — Review

## What was fixed

Two changes to `src/models/rwkv6_time_mix.py`:

### 1. Added `time_faaaa` (bonus term `u`)

The RWKV-6 attention formula has a bonus term `u` that gives extra weight
to the current timestep's key-value pair on the attention diagonal:

    y_t = r_t^T @ (S_t + u ⊙ k_t * v_t^T)

This was present in the original `rwkv-block` library and in `blocks.py`
but was missing from the `rwkv6_time_mix.py` reimplementation. Without it
the model under-weights the current frame and must compensate through the
decay/state alone, hurting both accuracy and gradient flow.

The parameter is initialized with a position-dependent zigzag pattern
matching the original RWKV-6 init (line 125–129 in the new code).

### 2. Replaced sequential Python loop with SmerkyG chunked parallel WKV

The old `_forward_recurrent` had a `for t in range(T)` loop that:
- Made T individual CUDA kernel launches per layer
- For T=250 (10s audio after 4× downsample) × 6 layers = **1500 sequential CUDA ops**
- Each op is tiny (matrix multiply of head_size × head_size) → ~99% GPU idle time

The new implementation splits the work into two levels:

**Intra-chunk (parallel):** Within a chunk of 128 frames, compute the
full causal attention matrix using cumulative log-decay sums and batched
matmuls. This is ~2 large parallel ops per chunk instead of 128 sequential
tiny ones.

**Inter-chunk (sequential):** Only ~T/128 steps (typically 2–4 for ASR
frames) carry the state matrix from one chunk to the next. This is the
irreducible sequential part.

The algorithm falls through chunk sizes `[128, 16, 2, 1]` so it handles
any sequence length including odd remainders.

### Expected speedup

The old code: ~1220 s/epoch (measured), ~35 k tokens/sec.
Expected with the fix: ~80–120 s/epoch (matching LION's ~89 s/epoch or
the earlier draft's ~83 s/epoch), ~350–500 k tokens/sec.

**Could not benchmark in this session** — CUDA context was corrupted by
the process kill during the earlier validation run. The benchmark must
be re-run in a fresh Python process (new terminal session or `uv run`).

---

## Why we have both `blocks.py` and the `rwkv6_*.py` files

### `blocks.py` (989 lines) — the original monolith

This was the **draft-era all-in-one file** from the early exploratory phase
of the project. It contains complete implementations of:

- `TransformerEncoder` — matched bidirectional Transformer baseline
- `LinearAttentionEncoder` — ELU+1 linear attention (Katharopoulos et al.)
- `RWKV6TimeMix` + `RWKV6ChannelMix` + `RWKV6Block` + `RWKV6Encoder`
  — standalone RWKV-6, causal only, no mechanisms, no LION
- `MambaBlock` + `MambaEncoder` — standalone Mamba

Key point: `blocks.py` **has no mechanism support** (no LION, no LUCID,
no Delta Rule, no ConvShift) and **only supports causal recurrent mode**.
It was the reference that proved the architectures work, then got
refactored into separate files.

### The `rwkv6_*.py` files — the formal v1 modular architecture

These replaced `blocks.py` for the formal experiment phase:

| File | Lines | Purpose |
|---|---|---|
| `rwkv6_time_mix.py` | 433 | **THE unified TimeMix** — one class supporting all modes (recurrent, lion, bidir_serial) and all mechanisms (conv_shift, headscale, delta_rule, lucid, temperature). This is the active code that all RWKV-6 and LION experiments use. |
| `rwkv6_channel_mix.py` | 59 | The FFN: `sigmoid(r) * value(relu²(key(x)))` |
| `rwkv6_block.py` | 80 | Single encoder layer: `ln0 → ln1 → TimeMix → drop → ln2 → ChannelMix → drop` |
| `rwkv6_encoder.py` | 99 | Stacks N blocks + positional encoding, carry-state support |
| `lion_attention.py` | 186 | LION-specific parallel attention kernels called by `rwkv6_time_mix.py._forward_lion` |

Additionally, the Transformer and Mamba got their own files:
- `transformer.py` — bidirectional Transformer baseline
- `transformer_causal.py` — causal Transformer with KV cache (Phase B)
- `mamba_block.py` — PyTorch Mamba with parallel scan (Phase A/B fixes)
- `mamba_encoder.py` — Mamba encoder with carry-state
- `mamba_cuda_encoder.py` — wraps the `mamba-ssm` CUDA library

### Why `blocks.py` still exists

It's **not imported by the encoder factory** (`encoder.py` imports from
the modular files). It's dead code in the formal v1 pipeline. It exists
because:

1. It was the reference implementation that the modular files were
   extracted from.
2. The SmerkyG chunked WKV in `blocks.py` was the correct version —
   `rwkv6_time_mix.py` was accidentally written with a naive sequential
   loop instead, causing the 14× speed regression. The fix was to port
   `blocks.py`'s `_rwkv6_wkv_subchunk` back into `rwkv6_time_mix.py`.
3. It could be safely deleted now (all its functionality is covered by
   the modular files), but it does no harm and serves as a reference.

### Is `rwkv6_time_mix.py` the "original RWKV"?

**It is our self-contained reimplementation of RWKV-6**, closely matching
the official `RWKV-block` library (`rwkv_block/v6_finch/block/rwkv6_time_mix.py`).
All parameter names, initializations, and the WKV algorithm trace back
to the RWKV project. The additions beyond stock RWKV-6 are:

- LION mode (bidirectional parallel attention, `_forward_lion`)
- Mechanism composition (ConvShift, Headscale, Delta Rule, LUCID, Temperature)
- Bidirectional serial mode (`_forward_bidir_serial`)

These are research extensions that the thesis evaluates. The "recurrent"
mode path with the SmerkyG chunked WKV **is** the same algorithm as the
original RWKV-6 — just reimplemented in pure PyTorch without the CUDA
kernel from `rwkv-block`.

---

## Next steps

1. **Benchmark the fix** in a fresh Python process (CUDA context is
   currently corrupted): `uv run python -c "..."` or restart the terminal.
2. **Re-run `exp02_rwkv6`** through the registry for 2 epochs to confirm
   epoch time is back to ~80–120 s.
3. If confirmed, update `INFRASTRUCTURE_PLAN.md §9` to mark D0 resolved
   and unblock Phase D.
4. **Optional cleanup:** delete `blocks.py` if we're confident the modular
   files cover everything. Or keep it as a reference — it's 989 lines of
   dead code that does no harm.
