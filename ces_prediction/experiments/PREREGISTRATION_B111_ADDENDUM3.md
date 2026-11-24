# B111 addendum3: built-in Colab file upload after runtime reconnection

2026-11-24, before any B111 scientific inference or fit. The author reconnected
Colab and explicitly requested Computer Use. The old partial workspace was
absent in a fresh CPU runtime; hardware and torch checks recorded that state.
The visible settings had T4 selected; applying Save starts T4 allocation.
Do not confuse a selected accelerator with an allocated runtime. No model
checkpoint is lost: all twenty remain in the local audited selection.

Replace repeated base64 cell writes with Colab's built-in Files upload UI
(or its standard files.upload widget) for the exact existing private archive
data/colab_b111/ces_b111_bundle.zip. The already authorized destination/data
scope is unchanged. Do not upload any other file or use third-party hosting.
Use the native file picker and only the exact local archive. This is one fresh
delivery after loss of the original partial file, not a scientific fit retry.
No kernel reset/delete, credentials,security setting change or new installation.

research/launch_uploaded_b111.py checks actual T4/CUDA, uploaded file length
and original SHA256, no launch marker or scientific started.json, and a fresh
extraction directory. Only then launch the unchanged frozen run111.py with
the exact B111 CES pins. Record actual hardware/package metadata and separate
operational helper/addendum hashes, preserving the old transport state.
Register a short poll cell, then the existing watcher owns download/audit.
All scientific inputs,model weights,post-fit parameters,source hashes,phase
ordering,bootstrap and statements remain unchanged. No local fit/GPU work.
Commit/push the addendum/helper before uploading/launching through this route.
