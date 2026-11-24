# B109 monitoring and independent retrieved-state audit

Before any B109 Colab fit. Add only local CPU orchestration,not local training:
research/watch_colab_b109.py polls the already registered remote controller,
reports phase/failure/five-minute progress locally and retrieves completed
checkpoints through the existing MCP operator. It never dispatches/retries
scientific jobs. research/audit_colab_b109.py verifies the downloaded archive,
all nine exits,weights/config/history/VAL arithmetic/patience and original
eleven retained paths,then writes a selected-path registry for future evaluation.
No TEST scoring or post-correction fitting. Set CUDA_VISIBLE_DEVICES=-1 for
the local artifact auditor. Commit/push before remote launch. B109 numerical
sources and the exported training archive remain unchanged.
