# synthetics_v1

Tier-1 synthetic expressivity benchmarks for the master thesis. Reuses the
encoder backbones from `../formal_v1/src/models/` (Transformer, RWKV-6, Mamba,
LION, plus mechanism variants) on:

- **MQAR** — Multi-Query Associative Recall (Arora et al., *Zoology* 2024)
- **State tracking** — parity, modular addition, Dyck-k *(planned)*
- **Selective copy / induction heads** *(planned)*

The point of this tier is to test architectural expressivity along axes that
LibriSpeech CTC ASR (the `formal_v1` backbone task) does not exercise:
associative recall, state tracking, in-context retrieval, length extrapolation.

See `PLAN.md` for the experiment matrix and `CLAUDE.md` for project conventions.
