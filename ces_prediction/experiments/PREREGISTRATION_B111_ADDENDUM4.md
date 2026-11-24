# B111 addendum4: one-cell upload/launch with read-only automatic attachment

2026-11-24, author explicitly resumed after the temporary pause. No B111
scientific child has launched. The previous browser MCP connection expired;
reopening established notebook read/edit access, but its first diagnostic
execution again timed out without an execution count or output. Native
Computer Use currently reports an unavailable pipe. Do not infer the selected
accelerator from this lack of a kernel response or repeatedly change settings.

research/prepare_b111_manual.py generates a small local GPU-metadata notebook
and places one owned bootstrap cell into the already connected scratchpad.
The author can run that cell and choose only the existing private archive
data/colab_b111/ces_b111_bundle.zip through Colab's built-in files.upload UI.
This delivers exactly the bytes/hash of the original B111 export, with the
same already-authorized Colab destination. It is not a new fit trial or a
different data source. No credentials or other local files are requested.

The bootstrap checks actual CUDA/T4 before upload, original archive length
and SHA before extraction, every manifest file hash before launch, and absence
of a launch marker or started.json. It launches only the frozen run111.py
with the original CES pins and observed-exit wrapper. It emits an aggregate
launch marker with a newly generated attempt ID and original archive SHA.
Never re-run the bootstrap after that marker exists. No CPU fallback or new
scientific code/fit/threshold is authorized by this addendum.

research/wait_b111_manual.py reads notebook state only while waiting. It
accepts only the exact locally generated owned source hash, attempt ID,
fixed remote root, original archive SHA and valid positive PID from the
bootstrap output. It never uploads,executes inference,fits or retries a
scientific process. Once acknowledged, it preserves old transport state,
creates one short read-only poll cell and writes the normal B111 connection
state. It then invokes the unchanged existing observer to retrieve/audit
results. Repeated checks before a launch do not run kernel code. Preserve
all attempts; a partial attachment failure stops rather than relaunching.

Generated notebook/attempt metadata stay under ignored data/colab_b111 and
data/.colab_connection. Source/registration must be committed/pushed before
cell preparation or using this route. The original archive, twenty selected
models,eight post fits,OOF exclusion,scenario readout and independent audit
remain unchanged. Local scientific fitting and GPU work remain stopped.
