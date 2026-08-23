"""Shot-ladder measurement of the trained CES VQC on real IonQ Forte hardware.

Question this answers
---------------------
A sampled quantum model estimates <Z> with error ~1/sqrt(shots), so the task's effect size
sets a *shot floor*. But shot noise is not the only error: gate and readout error add a floor
that more shots cannot remove. Which one dominates decides whether a QPU could *ever* resolve
this task, not merely whether it did at one shot count.

So we measure, on identical inputs and identical trained parameters, the deviation of the
hardware <Z_0> from the exact noiseless value at several shot counts:

    deviation ~ 1/sqrt(shots)  -> shot-limited; more shots (more money) would work
    deviation -> plateau       -> gate-error floor; no shot count ever resolves it

Billing reality (measured 2026-08-23 by free dry-run; see docs/ionq_qpu_experiment log)
--------------------------------------------------------------------------------------
IonQ direct bills ``cost_model: 2QGE_operations``, but this circuit is far below the floor,
so every job costs a flat MINIMUM set only by the shot tier:

    shots <= 400  ->  $25.79   (debiasing off)
    shots >= 500  ->  $168.20  (debiasing forced on, 32 variants; cannot be disabled)

Hence the ladder stays at 100/200/400: three points, one constant error-mitigation setting,
and 7x more jobs per dollar than the 500+ tier. ``--mitigated`` adds 2000-shot debiased jobs
as a separate best-case arm -- do not mix those into the ladder fit.

Safety
------
Default target is the FREE cloud simulator. Hardware needs ``--hardware`` and ``--yes``.
Every real submission is priced by a free dry-run first and checked against ``--budget``; the
run stops rather than exceed it. Every completed job is verified to have landed on the
requested QPU (``backend`` field) -- earlier jobs on this account were silently downgraded to
the simulator. State is written after every job so an interruption never loses paid results.

Run from the repo root::

    py ces_prediction/experiments/quantum/ionq_hw_ladder.py                   # free sim check
    py ces_prediction/experiments/quantum/ionq_hw_ladder.py --hardware        # price, then stop
    py ces_prediction/experiments/quantum/ionq_hw_ladder.py --hardware --yes  # spend
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(ROOT / "ces_prediction"))

API = "https://api.ionq.co/v0.4"
LOW_TIER_MAX_SHOTS = 400
PRICE_LOW, PRICE_HIGH = 25.79, 168.20


def load_api_key():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8-sig").splitlines():
            if line.strip().startswith("IONQ_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    key = os.getenv("IONQ_API_KEY")
    if not key:
        raise SystemExit("IONQ_API_KEY not found in .env or environment")
    return key


class IonQ:
    def __init__(self, key):
        self.h = {"Authorization": "apiKey " + key, "Content-Type": "application/json"}

    def call(self, method, path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(API + path, data=data, headers=self.h, method=method)
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return r.status, json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            return e.code, {"_error": e.read().decode()[:300]}

    def quota_used(self):
        _, p = self.call("GET", "/projects")
        return float(p["projects"][0]["quotaUsage"])

    def backend_status(self, target):
        _, b = self.call("GET", "/backends/" + target)
        return b.get("status"), b.get("degraded")

    def submit(self, circuit, n_qubits, shots, backend, dry_run, name):
        body = {"type": "ionq.circuit.v1", "name": name, "shots": shots, "backend": backend,
                "input": {"qubits": n_qubits, "gateset": "qis", "circuit": circuit}}
        if dry_run:
            body["dry_run"] = True
        st, job = self.call("POST", "/jobs", body)
        if not (isinstance(job, dict) and job.get("id")):
            raise RuntimeError("submit failed: HTTP %s %s" % (st, job))
        return job["id"]

    def wait(self, jid, poll=10, timeout_s=7200):
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            _, j = self.call("GET", "/jobs/" + jid)
            if j.get("status") in ("completed", "failed", "canceled"):
                return j
            time.sleep(poll)
        raise TimeoutError("job %s did not finish in %ds" % (jid, timeout_s))

    def cost(self, jid):
        _, c = self.call("GET", "/jobs/%s/cost" % jid)
        if not isinstance(c, dict):
            return None
        est = c.get("cost") or c.get("estimated_cost") or {}
        return est.get("value")

    def probabilities(self, job):
        url = ((job.get("results") or {}).get("probabilities") or {}).get("url")
        if not url:
            return None
        _, d = self.call("GET", url.replace("/v0.4", "", 1))
        return d if isinstance(d, dict) else None


def build_circuit(angles, weights):
    """Translate the PennyLane VQC into IonQ's qis gateset.

    Per layer: AngleEmbedding(rotation='Y') then StronglyEntanglingLayers(one layer).
    PennyLane's Rot(phi, theta, omega) == RZ(phi) RY(theta) RZ(omega); the single-layer
    entangler is a CNOT ring with range 1.
    """
    n_layers, n_qubits, _ = weights.shape
    circ = []
    for layer in range(n_layers):
        for q in range(n_qubits):
            circ.append({"gate": "ry", "target": q, "rotation": float(angles[q])})
        for q in range(n_qubits):
            phi, theta, omega = (float(x) for x in weights[layer, q])
            circ.append({"gate": "rz", "target": q, "rotation": phi})
            circ.append({"gate": "ry", "target": q, "rotation": theta})
            circ.append({"gate": "rz", "target": q, "rotation": omega})
        for q in range(n_qubits):
            circ.append({"gate": "cnot", "control": q, "target": (q + 1) % n_qubits})
    return circ


def expval_z0(probs, n_qubits, little_endian=True):
    """<Z_0> from IonQ's probability dict, keyed by the integer basis state."""
    tot = ev = 0.0
    for k, p in probs.items():
        idx = int(k)
        bit = idx & 1 if little_endian else (idx >> (n_qubits - 1)) & 1
        ev += p * (1.0 - 2.0 * bit)
        tot += p
    return ev / tot if tot > 0 else float("nan")


def exact_z0(angles, weights):
    """Noiseless reference from PennyLane's statevector simulator."""
    import pennylane as qml
    n_layers, n_qubits, _ = weights.shape
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(inp, w):
        for layer in range(n_layers):
            qml.AngleEmbedding(inp, wires=range(n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(w[layer:layer + 1], wires=range(n_qubits))
        return qml.expval(qml.PauliZ(0))

    return float(circuit(torch.as_tensor(angles), torch.as_tensor(weights)))


def load_operating_points(n_samples, seed, pin_indices=()):
    """Real val-set inputs projected onto the trained PCA basis -> embedding angles.

    Uses the same split manifest, stats and seed as quantum_vqc.py, so the operating points
    are ones the circuit was actually scored on rather than synthetic angles.

    Selection is a seeded *permutation* prefix, not ``rng.choice(size=k)``: choice draws a
    different set for every k, so growing --samples would discard already-paid measurements.
    A permutation prefix is stable under growth. ``pin_indices`` (val indices already measured
    on hardware) are placed first so a resumed run reuses them instead of re-buying them.
    """
    from evaluate import _load_stats
    from quantum_vqc import apply_pca, fit_pca, load_split_tensors

    ckpt = torch.load(HERE / "quantum_vqc_weights.pt", map_location="cpu", weights_only=False)
    data_dir = Path(os.getenv("CES_DATA_DIR", ROOT / "data"))
    split_dir = Path(os.getenv("CES_SPLIT_DIR", ROOT / "data" / "splits"))
    metrics = json.loads((ROOT / "ces_prediction" / "metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads((split_dir / "split_manifest.json").read_text(encoding="utf-8"))

    _, train, val, _, _ = load_split_tensors(
        data_dir, int(ckpt["window_size"]), _load_stats(metrics), manifest,
        int(os.getenv("QVQC_MAX_TRAIN", "2000")),
        int(os.getenv("CES_MAX_VAL_SAMPLES", "4000")), seed)

    mean, comps, scale = fit_pca(train["x"], int(ckpt["n_qubits"]))
    drift = float((comps.abs() - ckpt["pca_comps"].abs()).abs().max())
    z = apply_pca(val["x"], mean, comps, scale).numpy()

    rng = np.random.default_rng(seed)
    pinned = [int(i) for i in pin_indices if 0 <= int(i) < len(z)]
    seen = set(pinned)
    order = [int(i) for i in rng.permutation(len(z)) if int(i) not in seen]
    idx = np.array((pinned + order)[:min(n_samples, len(z))], dtype=int)
    return z[idx], idx, ckpt, drift


def main():
    # Jobs queue for minutes each; unbuffered output makes a backgrounded run followable.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--hardware", action="store_true")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--samples", type=int, default=12)
    ap.add_argument("--ladder", type=str, default="100,200,400")
    ap.add_argument("--mitigated", type=int, default=0)
    ap.add_argument("--budget", type=float, default=1264.0)
    ap.add_argument("--submit-only", dest="submit_only", action="store_true",
                    help="queue every remaining job and exit; collect later with --collect")
    ap.add_argument("--collect", action="store_true",
                    help="reclaim finished jobs from IonQ history into the result file, submit nothing")
    ap.add_argument("--backend", type=str, default="qpu.forte-1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=str(HERE / "ionq_hw_ladder_result.json"))
    args = ap.parse_args()

    ladder = [int(s) for s in args.ladder.split(",") if s.strip()]
    for s in ladder:
        if s > LOW_TIER_MAX_SHOTS:
            raise SystemExit("ladder point %d leaves the $25.79 tier (max %d); use --mitigated"
                             % (s, LOW_TIER_MAX_SHOTS))

    api = IonQ(load_api_key())
    target = args.backend if args.hardware else "simulator"
    out_path = Path(args.out)

    # Resume: any (val_index, shots) already measured on hardware is reused, never re-bought.
    prior = {}
    if out_path.exists():
        for r in json.loads(out_path.read_text(encoding="utf-8")).get("records", []):
            if r.get("on_hardware") and r.get("status") == "completed":
                prior[(int(r["val_index"]), int(r["shots"]))] = r
    if prior:
        print("Resuming: %d paid measurement(s) already on disk ($%.2f worth)"
              % (len(prior), sum(float(r.get("cost_usd") or 0) for r in prior.values())))

    print("Loading real val operating points ...")
    angles, idx, ckpt, drift = load_operating_points(
        args.samples + args.mitigated, args.seed, sorted({k[0] for k in prior}))
    weights = ckpt["weights"].numpy()
    n_qubits = int(ckpt["n_qubits"])
    print("  %d samples, %d qubits, %d layers, target %s (window=%s)"
          % (len(angles), n_qubits, ckpt["n_layers"], ckpt["target"], ckpt["window_size"]))
    print("  PCA basis drift vs checkpoint: %.3e" % drift)

    print("Computing exact noiseless <Z_0> ...")
    exact = np.array([exact_z0(a, weights) for a in angles])
    print("  <Z_0> range [%.4f, %.4f]" % (exact.min(), exact.max()))

    jobs = [(i, s, False) for s in ladder for i in range(args.samples)]
    jobs += [(args.samples + i, 2000, True) for i in range(args.mitigated)]

    # A job that completed on the QPU but whose result never reached disk (process killed
    # mid-run) is already paid for. Reclaim it from IonQ's history before spending again.
    if args.hardware:
        want = {"v%d_%dsh" % (int(idx[i]), s): (int(idx[i]), s) for i, s, _ in jobs}
        _, hist = api.call("GET", "/jobs?limit=100")
        for j in hist.get("jobs", []):
            name = str(j.get("name") or "")
            if (j.get("dry_run") or j.get("status") != "completed"
                    or name not in want or want[name] in prior):
                continue
            full = api.call("GET", "/jobs/" + j["id"])[1]
            if full.get("backend") != args.backend:
                continue
            probs = api.probabilities(full)
            if not probs:
                continue
            vi, sh = want[name]
            i = next(k for k, s2, _ in jobs if int(idx[k]) == vi and s2 == sh)
            prior[(vi, sh)] = {
                "sample": int(i), "val_index": vi, "shots": sh, "mitigated": False,
                "job_id": j["id"], "backend": full.get("backend"),
                "requested_backend": args.backend, "on_hardware": True,
                "status": "completed", "exec_ms": full.get("execution_duration_ms"),
                "exact_z0": float(exact[i]),
                "z0_little_endian": expval_z0(probs, n_qubits, True),
                "z0_big_endian": expval_z0(probs, n_qubits, False),
                "cost_usd": api.cost(j["id"]), "recovered_from_history": True}
            print("  reclaimed paid job %s (val %d, %d shots) from IonQ history" % (name, vi, sh))

    todo = [j for j in jobs if (int(idx[j[0]]), j[1]) not in prior]
    est = sum(PRICE_HIGH if m else PRICE_LOW for _, _, m in todo)
    print("\nPlan: %d measurements total, %d already paid for, %d to buy"
          % (len(jobs), len(jobs) - len(todo), len(todo)))
    print("Backend: %s   Estimated: $%.2f   Budget cap: $%.2f" % (target, est, args.budget))

    if args.hardware and not args.collect:
        # --collect spends nothing, so it skips these gates: it must keep working even when
        # the backend has gone unavailable again, which is exactly when results are stranded.
        status, degraded = api.backend_status(args.backend)
        print("Backend status: %s (degraded=%s)" % (status, degraded))
        if status != "available":
            raise SystemExit("criterion 1 FAILED: %s is %s -- doing nothing." % (args.backend, status))
        print("Quota used so far: $%.2f" % api.quota_used())
        if est > args.budget:
            raise SystemExit("estimate $%.2f exceeds --budget $%.2f" % (est, args.budget))
        if not args.yes:
            raise SystemExit("Dry stop: re-run with --yes to spend.")

    records, spent, price_cache = [], 0.0, {}

    def save(recs):
        """Merge into the file; never truncate it.

        A partial run writes only the measurements it has reached so far. Overwriting with
        that list silently deletes paid records from earlier runs that sit later in the job
        order -- this destroyed the same two 400-shot measurements twice today. Records are
        keyed by (val_index, shots), and anything already on disk under a key the current run
        has not produced is carried forward untouched.
        """
        merged = {}
        if out_path.exists():
            for r in json.loads(out_path.read_text(encoding="utf-8")).get("records", []):
                merged[(int(r["val_index"]), int(r["shots"]))] = r
        for r in recs:
            merged[(int(r["val_index"]), int(r["shots"]))] = r
        out_path.write_text(json.dumps({
            "backend": target, "ladder": ladder, "samples": args.samples,
            "mitigated": args.mitigated, "pca_drift": drift,
            "checkpoint": {k: v for k, v in ckpt.items() if not hasattr(v, "shape")},
            "records": [merged[k] for k in sorted(merged)]}, indent=2), encoding="utf-8")

    if args.collect:
        # Reclaim-only: the history sweep above already folded every finished job into
        # `prior`. Write what exists and report what the QPU still owes us.
        for i, shots, _ in jobs:
            k = (int(idx[i]), shots)
            if k in prior:
                r = dict(prior[k]); r["sample"] = int(i); records.append(r)
        save(records)
        print("\nCollected %d of %d measurements ($%.2f accounted). %d still pending on the QPU."
              % (len(records), len(jobs),
                 sum(float(r.get("cost_usd") or 0) for r in records), len(jobs) - len(records)))
        return

    if args.submit_only:
        # Fire-and-forget: IonQ holds submitted jobs server-side, so everything can be queued
        # now and collected from any later session with --collect. This is what lets the
        # machine be shut down mid-experiment; the sequential path below cannot survive that.
        for i, shots, mitigated in todo:
            circ = build_circuit(angles[i], weights)
            if shots not in price_cache:
                pid = api.submit(circ, n_qubits, shots, target, True, "price_%dsh" % shots)
                api.wait(pid)
                price_cache[shots] = api.cost(pid) or 0.0
                print("  priced %d shots at $%.2f (dry-run, free)" % (shots, price_cache[shots]))
            if spent + price_cache[shots] > args.budget:
                print("STOP: next job $%.2f would exceed budget (committed $%.2f)"
                      % (price_cache[shots], spent))
                break
            label = "v%d_%dsh%s" % (int(idx[i]), shots, "_dbz" if mitigated else "")
            jid = api.submit(circ, n_qubits, shots, target, False, label)
            spent += price_cache[shots]
            print("  queued %-16s %s  (committed $%.2f)" % (label, jid, spent))
        for i, shots, _ in jobs:
            k = (int(idx[i]), shots)
            if k in prior:
                r = dict(prior[k]); r["sample"] = int(i); records.append(r)
        save(records)
        print("\nQueued %d job(s), committing $%.2f. Nothing else needs to stay running --"
              % (len(todo), spent))
        print("collect them later with the same command plus --collect instead of --submit-only.")
        return

    for n, (i, shots, mitigated) in enumerate(jobs, 1):
        # Label by val index, not by position: the position depends on --samples, so a
        # position-named job cannot be matched back to its input after the plan changes.
        label = "v%d_%dsh%s" % (int(idx[i]), shots, "_dbz" if mitigated else "")
        key = (int(idx[i]), shots)
        if key in prior:
            rec = dict(prior[key])
            rec["sample"] = int(i)
            records.append(rec)
            print("[%2d/%2d] %-16s reused (val %d, $%.2f already paid)"
                  % (n, len(jobs), label, key[0], float(rec.get("cost_usd") or 0)))
            continue
        circ = build_circuit(angles[i], weights)

        if args.hardware:
            # Price depends only on the shot count: gate counts are identical across samples
            # (only the rotation angles differ), so one dry-run per shot tier is enough. Each
            # dry-run queues like a real job, so caching halves the wall-clock of a long run.
            if shots not in price_cache:
                price_id = api.submit(circ, n_qubits, shots, target, True, "price_%dsh" % shots)
                api.wait(price_id)
                price_cache[shots] = api.cost(price_id) or 0.0
                print("  priced %d shots at $%.2f (dry-run, free)" % (shots, price_cache[shots]))
            price = price_cache[shots]
            if spent + price > args.budget:
                print("STOP: next job $%.2f would exceed budget (spent $%.2f)" % (price, spent))
                break

        jid = api.submit(circ, n_qubits, shots, target, False, label)
        job = api.wait(jid)
        got_backend = job.get("backend")
        probs = api.probabilities(job)
        z_le = expval_z0(probs, n_qubits, True) if probs else float("nan")
        z_be = expval_z0(probs, n_qubits, False) if probs else float("nan")
        actual = (api.cost(jid) or 0.0) if args.hardware else 0.0
        spent += actual

        records.append({
            "sample": int(i), "val_index": int(idx[i]), "shots": shots, "mitigated": mitigated,
            "job_id": jid, "backend": got_backend, "requested_backend": target,
            "on_hardware": bool(args.hardware and got_backend == target),
            "status": job.get("status"), "exec_ms": job.get("execution_duration_ms"),
            "exact_z0": float(exact[i]), "z0_little_endian": z_le, "z0_big_endian": z_be,
            "cost_usd": actual, "spent_cumulative": spent})
        save(records)

        print("[%2d/%2d] %-16s backend=%-14s <Z0> le=%+.4f be=%+.4f exact=%+.4f  $%.2f (tot $%.2f)"
              % (n, len(jobs), label, got_backend, z_le, z_be, exact[i], actual, spent))
        if args.hardware and got_backend != target:
            print("  WARNING: job did NOT land on %s -- stopping." % target)
            break

    print("\nWrote %s   (%d jobs, $%.2f spent)" % (out_path, len(records), spent))


if __name__ == "__main__":
    main()
