"""Full-utterance and chunked evaluation."""

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
    """Compute CTC loss, CER, WER over a DataLoader."""
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
    avg_loss = total_loss / max(n_batches, 1)

    return {"loss": avg_loss, "cer": cer, "wer": wer_val}


@torch.no_grad()
def evaluate_chunked(
    model: nn.Module,
    dataset,
    vocab: CharVocab,
    chunk_sec: float,
    cfg: ExperimentConfig,
    device: torch.device,
    carry_state: bool = False,
) -> Optional[dict]:
    """Chunked inference evaluation with optional carry-state."""
    model.eval()
    hop_samples = cfg.hop_length_samples
    chunk_frames = int(chunk_sec * cfg.sample_rate / hop_samples)

    can_carry = carry_state and model.supports_carry_state
    if carry_state and not model.supports_carry_state:
        return None

    n_eval = len(dataset)
    if can_carry and cfg.max_carry_eval_utterances > 0:
        n_eval = min(n_eval, cfg.max_carry_eval_utterances)

    all_hyps, all_refs = [], []

    for idx in range(n_eval):
        mel, targets = dataset[idx]
        T_total = mel.shape[1]

        enc_state = None
        if can_carry:
            enc_state = model.encoder.init_state(batch_size=1, device=device)

        chunk_hyp_tokens = []
        prev_token = -1

        for start in range(0, T_total, chunk_frames):
            end = min(start + chunk_frames, T_total)
            chunk_mel = mel[:, start:end].unsqueeze(0).to(device)
            chunk_len = torch.tensor([chunk_mel.shape[2]], dtype=torch.long, device=device)

            log_probs, out_lens, new_state = model(chunk_mel, chunk_len, state=enc_state)
            out_lens = torch.clamp(out_lens, max=log_probs.size(1))

            if can_carry:
                enc_state = new_state

            preds = log_probs.argmax(dim=-1)[0, :out_lens[0]].tolist()

            collapsed = []
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
        hyp = vocab.decode(final_tokens)
        ref = vocab.decode(targets.tolist())
        all_hyps.append(hyp)
        all_refs.append(ref)

    cer = compute_cer(all_hyps, all_refs)
    wer_val = compute_wer(
        [" ".join(r.split()) if r.strip() else "<empty>" for r in all_refs],
        [" ".join(h.split()) if h.strip() else "<empty>" for h in all_hyps],
    )
    return {"cer": cer, "wer": wer_val, "n_evaluated": n_eval}
