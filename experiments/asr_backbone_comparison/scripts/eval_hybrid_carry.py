#!/usr/bin/env python3
"""
Option B: Hybrid carry-state evaluation for Bi-WKV models.

Current _forward_carry skips backward blocks entirely → CTC head sees forward-only
output, which it was never trained on → CER 0.71 (catastrophic).

This script implements hybrid carry:
  - Forward blocks: state carried across chunk boundaries (accumulates context)
  - Backward blocks: reset to zero at every chunk boundary (sees only current chunk)
  - Combination: same as training (0.5*(fwd+bwd) or gate)

The CTC head then receives a combined fwd+bwd input at every chunk — same distribution
as training, just without cross-chunk backward context.

Usage:
    python scripts/eval_hybrid_carry.py \
        --run-dir outputs/run-007_biwkv6 \
        --chunk-sizes 2.0 5.0 10.0 \
        --n-utterances 500
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from jiwer import wer as compute_wer

EXP_DIR = Path(__file__).parent.parent
RWKV_DIR = EXP_DIR / "RWKV-block"
SRC_DIR  = EXP_DIR / "src"
sys.path.insert(0, str(RWKV_DIR))
sys.path.insert(0, str(SRC_DIR))

from asr_exp.config import load_config
from asr_exp.data import ASRDataset, CharVocab, load_and_prepare_data
from asr_exp.models import ASRModel
from asr_exp.training.decode import compute_cer, greedy_ctc_decode


# ── Hybrid carry inference ─────────────────────────────────────────────────────

@torch.no_grad()
def hybrid_carry_chunked(model, dataset, vocab, chunk_sec, cfg, device, n_utterances):
    """
    Hybrid carry: forward state carried, backward state reset each chunk.

    Directly accesses encoder internals (layers_fwd, layers_bwd, gates,
    conv_shifts) to implement the combined inference loop.
    """
    enc = model.encoder
    hop_samples  = int(cfg.hop_length_ms * cfg.sample_rate / 1000)
    chunk_frames = int(chunk_sec * cfg.sample_rate / hop_samples)
    n_eval = min(len(dataset), n_utterances)

    all_hyps, all_refs = [], []

    for idx in range(n_eval):
        mel, targets = dataset[idx]          # (n_mels, T_total), (L,)
        T_total = mel.shape[1]

        # Initialise FORWARD carry state (persists across chunks)
        fwd_state = enc.init_state(batch_size=1, device=device)

        chunk_hyp_tokens        = []
        prev_token_across_chunks = -1

        for start in range(0, T_total, chunk_frames):
            end       = min(start + chunk_frames, T_total)
            chunk_mel = mel[:, start:end].unsqueeze(0).to(device)   # (1, n_mels, T_chunk)
            chunk_len = torch.tensor([chunk_mel.shape[2]], dtype=torch.long, device=device)

            # ── frontend (ConvSubsampling) ─────────────────────────────────
            x, lengths = model.frontend(chunk_mel, chunk_len)       # (1, T', D)
            B, T, D = x.shape

            # ── positional encoding ────────────────────────────────────────
            x = enc.pos_enc(x)

            # ── padding mask ───────────────────────────────────────────────
            mask_f = (
                (torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1))
                .unsqueeze(-1).float()
            )

            # ── layer-by-layer hybrid forward ─────────────────────────────
            new_fwd_state = []
            for i in range(enc.n_layers):

                # Forward block — carry state in, new state out
                p_fwd, new_s = enc.layers_fwd[i](x, fwd_state[i])
                new_fwd_state.append(new_s)

                # Backward block — fresh zero state each chunk
                bwd_state_zero = (
                    torch.zeros(B, D, device=device),
                    torch.zeros(
                        B, enc.n_head, enc.head_size, enc.head_size,
                        dtype=torch.float32, device=device,
                    ),
                    torch.zeros(B, D, device=device),
                )
                p_bwd_flipped, _ = enc.layers_bwd[i](x.flip(1), bwd_state_zero)
                p_bwd = p_bwd_flipped.flip(1)

                # Combine — same logic as _forward_bidir
                if enc.use_gate:
                    xres = enc.conv_shifts[i](x) - x
                    G    = torch.sigmoid(enc.gates[i](xres))
                    x    = G * p_fwd + (1.0 - G) * p_bwd
                else:
                    x = 0.5 * (p_fwd + p_bwd)

                x = x * mask_f

            fwd_state = new_fwd_state

            # ── CTC head ───────────────────────────────────────────────────
            logits   = model.ctc_head(x)
            log_probs = F.log_softmax(logits, dim=-1)
            out_lens  = torch.clamp(lengths, max=log_probs.size(1))

            preds = log_probs.argmax(dim=-1)[0, : out_lens[0]].tolist()

            # Collapse repeats within chunk
            collapsed, prev = [], -1
            for p in preds:
                if p != prev:
                    collapsed.append(p)
                prev = p

            # Cross-chunk boundary deduplication
            if collapsed and collapsed[0] == prev_token_across_chunks and collapsed[0] != 0:
                collapsed = collapsed[1:]

            if collapsed:
                prev_token_across_chunks = collapsed[-1]

            chunk_hyp_tokens.extend(collapsed)

        final_tokens = [t for t in chunk_hyp_tokens if t != 0]
        hyp = vocab.decode(final_tokens)
        ref = vocab.decode(targets.tolist())
        all_hyps.append(hyp)
        all_refs.append(ref)

    cer = compute_cer(all_hyps, all_refs)
    wer_val = compute_wer(
        [" ".join(r.split()) if r.strip() else "⟨empty⟩" for r in all_refs],
        [" ".join(h.split()) if h.strip() else "⟨empty⟩" for h in all_hyps],
    )
    return {"cer": cer, "wer": wer_val, "n_evaluated": n_eval}


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir",    default="outputs/run-007_biwkv6",
                        help="Path to the run output directory")
    parser.add_argument("--chunk-sizes", nargs="+", type=float, default=[2.0, 5.0, 10.0])
    parser.add_argument("--n-utterances", type=int, default=500)
    parser.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = EXP_DIR / run_dir

    config_path = run_dir / "config_snapshot.yaml"
    vocab_path  = run_dir / "vocab.json"

    print(f"Loading config from {config_path}")
    cfg  = load_config(str(config_path))
    cfg.data_dir = str(EXP_DIR / "data" / "cv_uk")
    vocab = CharVocab.load(str(vocab_path))
    device = torch.device(args.device)

    print("Loading test data …")
    _, _, test_entries, _ = load_and_prepare_data(cfg, build_vocab=False)
    test_ds = ASRDataset(test_entries, vocab, cfg)
    print(f"  {len(test_ds)} test utterances (using first {args.n_utterances})")

    backbones = ["biwkv6_no_conv_no_gate", "biwkv6"]
    all_results = {}

    for backbone in backbones:
        ckpt_path = run_dir / f"best_{backbone}.pt"
        if not ckpt_path.exists():
            print(f"  [SKIP] checkpoint not found: {ckpt_path}")
            continue

        print(f"\n{'='*60}")
        print(f"  Backbone: {backbone}")
        print(f"  Checkpoint: {ckpt_path}")

        cfg_b = load_config(str(config_path))
        cfg_b.data_dir = str(EXP_DIR / "data" / "cv_uk")

        model = ASRModel(backbone, vocab.size, cfg_b).to(device)
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        print(f"  Loaded weights.")

        results_backbone = {}
        for chunk_sec in args.chunk_sizes:
            print(f"\n  [hybrid carry @ {chunk_sec}s]", end=" ", flush=True)
            r = hybrid_carry_chunked(
                model, test_ds, vocab, chunk_sec, cfg_b, device, args.n_utterances
            )
            print(f"CER={r['cer']:.4f}  WER={r['wer']:.4f}  n={r['n_evaluated']}")
            results_backbone[f"{chunk_sec}s"] = r

        all_results[backbone] = results_backbone

    # ── Print summary table ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  HYBRID CARRY RESULTS SUMMARY")
    print(f"{'='*60}")

    header = f"{'Backbone':<30} {'HC@2s_CER':>10} {'HC@2s_WER':>10} {'HC@5s_CER':>10} {'HC@5s_WER':>10} {'HC@10s_CER':>11} {'HC@10s_WER':>11}"
    print(header)
    print("-" * len(header))

    for backbone, res in all_results.items():
        row_vals = []
        for chunk_sec in args.chunk_sizes:
            key = f"{chunk_sec}s"
            if key in res:
                row_vals += [f"{res[key]['cer']:.4f}", f"{res[key]['wer']:.4f}"]
            else:
                row_vals += ["  N/A  ", "  N/A  "]
        print(f"{backbone:<30} {row_vals[0]:>10} {row_vals[1]:>10} {row_vals[2]:>10} {row_vals[3]:>10} {row_vals[4]:>11} {row_vals[5]:>11}")

    print()

    # Compare vs original carry (from results.json)
    original_results_path = run_dir / "results.json"
    if original_results_path.exists():
        with open(original_results_path) as f:
            orig = json.load(f)
        print("  COMPARISON vs original carry (forward-only):")
        print(f"  {'Backbone':<30} {'orig C@2s':>10} {'HC@2s':>10} {'orig C@10s':>11} {'HC@10s':>11}")
        print("  " + "-" * 65)
        for backbone in backbones:
            if backbone not in orig or backbone not in all_results:
                continue
            orig_c2  = orig[backbone]["chunked_carry"].get("2.0s", {}).get("cer", float("nan"))
            orig_c10 = orig[backbone]["chunked_carry"].get("10.0s", {}).get("cer", float("nan"))
            hc2  = all_results[backbone].get("2.0s", {}).get("cer", float("nan"))
            hc10 = all_results[backbone].get("10.0s", {}).get("cer", float("nan"))
            print(f"  {backbone:<30} {orig_c2:>10.4f} {hc2:>10.4f} {orig_c10:>11.4f} {hc10:>11.4f}")
        print()

    # Save
    out_path = run_dir / "results_hybrid_carry.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Results saved to {out_path}")


if __name__ == "__main__":
    main()
