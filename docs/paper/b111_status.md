# B111: full LSTM post-correction/scenario comparison

Preparing the complete executable Colab workflow, before export or new fits.
Uses twenty selected B109-audited LSTM checkpoints without retraining them.
Sixteen held-file replays -> eight HGB15 OOF post fits -> four outer replays
over eight scenarios ->32 common-row F MSE cells -> independent audit.
Two post policies: frozen_D10 and mixed_post, both retained without selection.

Registration PREREGISTRATION_B111.md; code b111_full_compare/. Delivery and
observer: research/colab_b111_operator.py,watch_colab_b111.py. Private payload,
predictions,models and audit evidence data/colab_b111. Live state is in
data/.colab_connection/b111_progress.json and b111_watch.json after launch.
Local GPU stays paused; no local fit or Claude call. Commit/push before export.
