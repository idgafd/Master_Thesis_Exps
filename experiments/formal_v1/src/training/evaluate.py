"""Full-utterance and chunked evaluation.

Reset mode is batched across utterances for ~20× speedup over the old
single-utterance-at-a-time path. Carry-state mode must stay sequential per
utterance because the state is per-utterance by definition.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from jiwer import wer as compute_wer

from src.config import ExperimentConfig
from src.data.vocab import CharVocab
from src.training.decode import compute_cer, greedy_ctc_decode


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader,
    vocab: CharVocab,
    device: torch.device,
    tag: str = "dev",
) -> dict:
    """Compute CTC loss, CER, WER over a DataLoader (full-utterance)."""
    model.eval()
    ctc_loss_fn = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    all_hyps, all_refs = [], []
    total_loss = 0.0
    n_batches = 0

    for mels, targets, mel_lengths, target_lengths in dataloader:
        mels = mels.to(device)
        targets = targets.to(device)
        mel_lengths = mel_lengths.to(device)
        target_lengths = target_lengths.to(device)

        log_probs, output_lengths, _ = model(mels, mel_lengths)
        output_lengths = torch.clamp(output_lengths, max=log_probs.size(1))

        loss = ctc_loss_fn(
            log_probs.permute(1, 0, 2), targets, output_lengths, target_lengths
        )
        total_loss += loss.item()
        n_batches += 1

        hyps = greedy_ctc_decode(log_probs.cpu(), output_lengths.cpu(), vocab)
        for i in range(targets.size(0)):
            ref = vocab.decode(targets[i, :target_lengths[i]].tolist())
            all_refs.append(ref)
        all_hyps.extend(hyps)

    cer = compute_cer(all_hyps, all_refs)
    wer_val = compute_wer(
        [" ".join(r.split()) if r.strip() else "<empty>" for r in all_refs],
        [" ".join(h.split()) if h.strip() else "<empty>" for h in all_hyps],
    )
    return {"loss": total_loss / max(n_batches, 1), "cer": cer, "wer": wer_val}


@torch.no_grad()
def evaluate_chunked(
    model: nn.Module,
    dataset,
    vocab: CharVocab,
    chunk_sec: float,
    cfg: ExperimentConfig,
    device: torch.device,
    carry_state: bool = False,
    batch_size: int = 16,
    max_utterances: Optional[int] = None,
) -> Optional[dict]:
    """Chunked inference evaluation.

    Reset mode is batched across utterances for speed. Carry-state mode
    processes one utterance at a time (state is per-utterance by definition).

    Args:
        chunk_sec: chunk length in seconds.
        carry_state: propagate encoder state across chunks within an utterance.
        batch_size: for reset mode only.
        max_utterances: cap on number of utterances to evaluate. None = all.
    """
    model.eval()
    hop_samples = cfg.hop_length_samples
    chunk_frames = int(chunk_sec * cfg.sample_rate / hop_samples)

    if carry_state and not model.supports_carry_state:
        return None

    n_eval = len(dataset)
    if max_utterances is not None and max_utterances > 0:
        n_eval = min(n_eval, max_utterances)

    if carry_state:
        return _evaluate_chunked_carry(
            model, dataset, vocab, chunk_frames, device, n_eval
        )
    return _evaluate_chunked_reset_batched(
        model, dataset, vocab, chunk_frames, device, n_eval, batch_size
    )


# ── reset mode: batched across utterances ─────────────────────────────────

def _evaluate_chunked_reset_batched(
    model, dataset, vocab, chunk_frames: int, device, n_eval: int,
    batch_size: int,
) -> dict:
    """Process utterances in mini-batches of matching chunks.

    Each mini-batch is formed from chunks of the same chunk index across
    different utterances. Chunks shorter than `chunk_frames` are right-padded
    with zeros and masked via `lengths`.
    """
    all_hyps: list[str] = ["" for _ in range(n_eval)]
    all_refs: list[str] = []

    # Load all utterances into memory (they're log-mel tensors, small)
    utterances = []
    for idx in range(n_eval):
        mel, targets = dataset[idx]
        utterances.append(mel)
        all_refs.append(vocab.decode(targets.tolist()))

    # Determine how many chunks each utterance has
    n_chunks = [(mel.shape[1] + chunk_frames - 1) // chunk_frames for mel in utterances]
    max_chunks = max(n_chunks) if n_chunks else 0

    # Hyp token buffers, one per utterance
    hyp_tokens: list[list[int]] = [[] for _ in range(n_eval)]
    prev_last: list[int] = [-1 for _ in range(n_eval)]

    # Iterate chunk index across utterances
    for ci in range(max_chunks):
        # Gather utterances that still have a chunk at index ci
        participating = [i for i in range(n_eval) if n_chunks[i] > ci]
        # Process in batches
        for bstart in range(0, len(participating), batch_size):
            batch_idxs = participating[bstart : bstart + batch_size]
            n_mels = utterances[batch_idxs[0]].shape[0]

            batch_mel = torch.zeros(
                len(batch_idxs), n_mels, chunk_frames,
                dtype=torch.float32, device=device,
            )
            batch_lens = torch.zeros(
                len(batch_idxs), dtype=torch.long, device=device
            )
            for j, uidx in enumerate(batch_idxs):
                mel = utterances[uidx]
                t0 = ci * chunk_frames
                t1 = min(t0 + chunk_frames, mel.shape[1])
                chunk = mel[:, t0:t1]
                batch_mel[j, :, : chunk.shape[1]] = chunk.to(device)
                batch_lens[j] = chunk.shape[1]

            log_probs, out_lens, _ = model(batch_mel, batch_lens)
            out_lens = torch.clamp(out_lens, max=log_probs.size(1))
            preds = log_probs.argmax(dim=-1).cpu()
            out_lens_cpu = out_lens.cpu()

            for j, uidx in enumerate(batch_idxs):
                chunk_preds = preds[j, : out_lens_cpu[j]].tolist()
                collapsed: list[int] = []
                prev = -1
                for p in chunk_preds:
                    if p != prev:
                        collapsed.append(p)
                    prev = p
                # Merge across chunk boundaries: avoid duplicating the last
                # non-blank token of the previous chunk.
                if collapsed and collapsed[0] == prev_last[uidx] and collapsed[0] != 0:
                    collapsed = collapsed[1:]
                if collapsed:
                    prev_last[uidx] = collapsed[-1]
                hyp_tokens[uidx].extend(collapsed)

    for i in range(n_eval):
        final = [t for t in hyp_tokens[i] if t != 0]
        all_hyps[i] = vocab.decode(final)

    cer = compute_cer(all_hyps, all_refs)
    wer_val = compute_wer(
        [" ".join(r.split()) if r.strip() else "<empty>" for r in all_refs],
        [" ".join(h.split()) if h.strip() else "<empty>" for h in all_hyps],
    )
    return {"cer": cer, "wer": wer_val, "n_evaluated": n_eval}


# ── carry-state mode: sequential per utterance ────────────────────────────

def _evaluate_chunked_carry(
    model, dataset, vocab, chunk_frames: int, device, n_eval: int,
) -> dict:
    """Carry encoder state across chunks within each utterance."""
    all_hyps: list[str] = []
    all_refs: list[str] = []

    for idx in range(n_eval):
        mel, targets = dataset[idx]
        T_total = mel.shape[1]

        enc_state = model.encoder.init_state(batch_size=1, device=device)
        chunk_hyp_tokens: list[int] = []
        prev_token = -1

        for start in range(0, T_total, chunk_frames):
            end = min(start + chunk_frames, T_total)
            chunk_mel = mel[:, start:end].unsqueeze(0).to(device)
            chunk_len = torch.tensor(
                [chunk_mel.shape[2]], dtype=torch.long, device=device
            )
            log_probs, out_lens, enc_state = model(
                chunk_mel, chunk_len, state=enc_state
            )
            out_lens = torch.clamp(out_lens, max=log_probs.size(1))

            preds = log_probs.argmax(dim=-1)[0, : out_lens[0]].tolist()
            collapsed: list[int] = []
            prev = -1
            for p in preds:
                if p != prev:
                    collapsed.append(p)
                prev = p

            if collapsed and collapsed[0] == prev_token and collapsed[0] != 0:
                collapsed = collapsed[1:]
            if collapsed:
                prev_token = collapsed[-1]
            chunk_hyp_tokens.extend(collapsed)

        final_tokens = [t for t in chunk_hyp_tokens if t != 0]
        all_hyps.append(vocab.decode(final_tokens))
        all_refs.append(vocab.decode(targets.tolist()))

    cer = compute_cer(all_hyps, all_refs)
    wer_val = compute_wer(
        [" ".join(r.split()) if r.strip() else "<empty>" for r in all_refs],
        [" ".join(h.split()) if h.strip() else "<empty>" for h in all_hyps],
    )
    return {"cer": cer, "wer": wer_val, "n_evaluated": n_eval}
