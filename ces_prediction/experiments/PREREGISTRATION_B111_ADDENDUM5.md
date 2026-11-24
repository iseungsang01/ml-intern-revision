# B111 addendum5: standalone upload/run/download without a live MCP connection

2026-11-24, before any B111 scientific child. During the resumed session,
the first diagnostic execution timed out without output, and the browser
proxy disconnected again before the prepared bootstrap could be inserted.
Native Computer Use is unavailable. Preserve these failed connector attempts;
they are not training/evaluation outcomes. Do not require a persistent proxy
to complete the already registered experiment.

research/prepare_b111_standalone.py takes the prepared owned bootstrap,
preserves its original local version, and generates a standalone one-cell
COLAB_B111.ipynb with GPU/T4 metadata under ignored data/colab_b111. The
author can open it in Colab, run its single cell, and choose exactly the
original ces_b111_bundle.zip. All T4,archive SHA,source and no-launch checks
from addendum4 remain. The original run111.py executes unchanged. The cell
waits for the actual wrapper exit, prints actual progress, and downloads the
normal results archive after successful completion. It adds only a small
delivery receipt to that zip: owned attempt ID, original archive SHA,
manifest SHA,actual hardware,returncode and operational source hashes. Existing
producer files/result_hashes are not edited. A failed process never restarts.

research/watch_b111_download.py watches only the exact expected new local
Downloads/b111_results.zip, not unrelated files or browsing data. Accept only
a complete valid zip newer than preparation, with the matching owned receipt,
archive/manifest hashes,T4 and zero exit code. Preserve the existing connection
state, copy the receipt-bound archive into the private B111 output directory,
record its SHA, and invoke the existing independent CPU audit. No fits or
kernel execution occur in this local observer. There is no need to enable
the disabled MCP-based watcher concurrently. Save all failures; never infer
that a user has launched merely because the notebook was prepared.

This changes delivery/observation only. Same twenty checkpoint weights,eight
post fits,OOF exclusions,32 common-row cells,bootstrap and statements. No CPU
scientific fallback,local fit,or local GPU work. Sources/registration are
committed/pushed before generating or using this standalone route.
