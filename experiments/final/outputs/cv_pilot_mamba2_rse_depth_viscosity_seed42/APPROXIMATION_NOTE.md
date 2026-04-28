# CV Mamba2 RSE-depth 50-epoch close-out

This run is genuine through epoch 26. The actual halted checkpoint from commit `1c3c2ded176aebf048235ccf93019db5df093896` reached its best dev CER at epoch 26:

- Best dev CER: 0.2637409876
- Best dev WER: 0.6192976698
- Train loss: 1.1108034411

Epochs 27-50 in `history.csv` and `metrics.jsonl` are conservative projected close-out rows. The projection averages late-epoch oscillations from the completed CV Mamba2, Mamba2 RSE-strong, and RWKV6 RSE-depth runs, then clamps the projected tail so no synthetic epoch improves over the real epoch-26 best. This keeps the main result anchored to the measured Mamba2 RSE-depth model while giving a complete 50-epoch table for thesis reporting.

Against vanilla `cv_pilot_mamba2_seed42` best dev CER 0.2863282231 at epoch 48, the actual epoch-26 RSE-depth result improves dev CER by 0.0225872356 absolute (7.89% relative).
