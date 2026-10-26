"""The ONE on-arrival slot with code behind it: reading the delivered raw files.

Delivery (2026-10-26): 36 CSV files, `s{shot}_{BES|ECEI|MC}.csv`, one per shot and
diagnostic, placed under `data/raw_mus/` (gitignored, 7.4 GB). Each has a `time` column
(KSTAR discharge seconds) followed by the same channel names as the 10 ms shot CSVs
(BES_0105..BES_0307, ECEI_GT1103..GT1403, MC1T03/MC1T16). No metadata file came with
the delivery (no stated sampling rate, no anti-aliasing filter description), so the
sampling rate is MEASURED here from the row count and the time span, and it is
verified against the delivered time column sample by sample before anything downstream
sees it (the contract below). The measured rates are written to the pre-registration's
on-arrival slot by `--build`, which also records every deviation it finds
(`delivery_manifest.json`).

Delivered time columns are quantised (6 decimals = 1 us for most files, 5 decimals =
10 us for #31873 and #31921), so several consecutive samples can share a printed
timestamp; the stream itself is uniform. The loader therefore returns the reconstructed
uniform clock `t0 + i / fs` and refuses the file if any delivered timestamp sits more
than one print quantum away from that clock.

Contract:
    load_raw(shot: int, channel: str) -> ((t, x), fs)
        t   float64 seconds, monotonically increasing, same clock as the shot CSVs'
            `time` column (KSTAR discharge time)
        x   float64 samples
        fs  float, the channel's sampling rate in Hz -- measured from the delivered
            row count / span, rounded to the nearest 1 kHz, and verified against the
            delivered time column (abort on disagreement)

Do NOT resample, filter, or detrend here: the extractor owns every numerical choice
(PREREGISTRATION_B6.md sec. 3.1 implementation constants), and the quality gate
(sec. 1.4) needs the stream exactly as delivered.

A per-file npz cache (`data/raw_mus/.cache/s{shot}_{diag}.npz`, float32 samples plus
the verified clock parameters) makes repeated loads take seconds instead of minutes;
it is keyed on the CSV's size and mtime and rebuilt when either changes.
"""

from pathlib import Path
import json
import os
import sys

import numpy as np

RAW_READY = True

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
RAW_DIR = Path(os.getenv("CES_RAW_DIR", REPO_ROOT / "data" / "raw_mus"))
CACHE_DIR = RAW_DIR / ".cache"
DIAGS = ("BES", "ECEI", "MC")


def diag_of(channel):
    if channel.startswith("BES_"):
        return "BES"
    if channel.startswith("ECEI_"):
        return "ECEI"
    if channel.startswith("MC"):
        return "MC"
    raise KeyError(f"unknown raw channel {channel}")


def csv_path(shot, diag):
    return RAW_DIR / f"s{shot}_{diag}.csv"


def _print_quantum(path):
    """Resolution of the printed `time` strings (1e-decimals), from the first data row."""
    with open(path, "r", encoding="ascii", errors="replace") as f:
        f.readline()
        first = f.readline().split(",")[0].strip()
    dec = len(first.split(".")[1]) if "." in first else 0
    return 10.0 ** (-dec)


def _verify_clock(t_delivered, quantum):
    """Measured fs (nearest 1 kHz) and the clock check against the delivered column."""
    n = len(t_delivered)
    t0 = float(t_delivered[0])
    span = float(t_delivered[-1]) - t0
    if n < 2 or span <= 0:
        raise SystemExit("FATAL: raw file has no usable time span")
    fs_est = (n - 1) / span
    fs = float(round(fs_est / 1000.0) * 1000.0)
    t_rec = t0 + np.arange(n, dtype=np.float64) / fs
    dev = np.abs(t_delivered - t_rec)
    max_dev = float(dev.max())
    # one print quantum of rounding plus float slack; anything larger means the stream
    # is not uniform at the measured rate (a gap, a rate change, a dropped block)
    tol = quantum + 1e-9 + 1e-6 * span * 1e-6
    if abs(fs_est - fs) / fs > 0.01:
        raise SystemExit(f"FATAL: measured rate {fs_est:.1f} Hz is not within 1% of a "
                         f"1 kHz multiple -- refuse to guess the clock")
    if max_dev > tol:
        raise SystemExit(f"FATAL: delivered time column deviates {max_dev:.3e} s from a "
                         f"uniform {fs:.0f} Hz clock (tolerance {tol:.3e}) -- not uniform")
    return fs, t0, n, max_dev, float(fs_est)


def _cache_path(shot, diag):
    return CACHE_DIR / f"s{shot}_{diag}.npz"


def _csv_signature(path):
    st = path.stat()
    return f"{st.st_size}:{int(st.st_mtime)}"


def build_table(shot, diag, quiet=False):
    """Parse one delivered CSV, verify its clock, cache it, return the table dict."""
    import pandas as pd
    path = csv_path(shot, diag)
    if not path.exists():
        raise SystemExit(f"FATAL: {path} missing -- real raw data required.")
    quantum = _print_quantum(path)
    df = pd.read_csv(path, dtype=np.float64)
    if df.columns[0] != "time":
        raise SystemExit(f"FATAL: {path} first column is {df.columns[0]!r}, not time")
    t = df["time"].to_numpy(np.float64)
    fs, t0, n, max_dev, fs_est = _verify_clock(t, quantum)
    channels = [c for c in df.columns if c != "time"]
    x = df[channels].to_numpy(np.float32)
    n_nan = int(np.isnan(x).sum())
    meta = {"shot": int(shot), "diag": diag, "fs_hz": fs, "fs_est_hz": fs_est, "t0": t0,
            "t1": float(t0 + (n - 1) / fs), "n": int(n), "channels": channels,
            "print_quantum_s": quantum, "max_clock_dev_s": max_dev, "n_nan": n_nan,
            "csv": path.name, "csv_signature": _csv_signature(path)}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(_cache_path(shot, diag), x=x, meta=json.dumps(meta))
    if not quiet:
        print(f"[raw] s{shot} {diag:4s} fs={fs/1e3:7.1f} kHz  n={n:>9d}  "
              f"t={t0:.3f}..{meta['t1']:.3f}  quantum={quantum:g}  "
              f"max_dev={max_dev:.2e}  nan={n_nan}", flush=True)
    return meta, x


def _load_table(shot, diag):
    path = csv_path(shot, diag)
    cache = _cache_path(shot, diag)
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        if path.exists() and meta.get("csv_signature") != _csv_signature(path):
            return build_table(shot, diag, quiet=True)
        return meta, z["x"]
    return build_table(shot, diag, quiet=True)


def load_raw(shot, channel):
    diag = diag_of(channel)
    meta, x = _load_table(shot, diag)
    if channel not in meta["channels"]:
        raise KeyError(f"s{shot}/{diag} has no channel {channel}: {meta['channels']}")
    col = meta["channels"].index(channel)
    fs = float(meta["fs_hz"])
    t = meta["t0"] + np.arange(meta["n"], dtype=np.float64) / fs
    return (t, x[:, col].astype(np.float64)), fs


def coverage(shot, diag):
    """(t0, t1, fs) of one delivered stream, from the cache (parses the CSV if needed)."""
    meta, _ = _load_table(shot, diag)
    return meta["t0"], meta["t1"], meta["fs_hz"]


def main():
    """`py raw_loader.py --build`: parse every delivered file, verify every clock, write the
    npz cache and `data/raw_mus/delivery_manifest.json` (the on-arrival record)."""
    sys.path.insert(0, str(HERE.parents[0] / "hires_shots"))
    from folds import TEST, POOL, COMPANIONS  # noqa: E402
    import pandas as pd  # noqa: E402

    if "--build" not in sys.argv:
        print(__doc__)
        return
    rows = []
    for shot in (*TEST, *POOL, *COMPANIONS):
        grid = pd.read_csv(REPO_ROOT / "data" / f"s{shot}.csv", usecols=["time", "CES_TI", "CES_VT"])
        lab = grid[np.isfinite(grid["CES_TI"]) | np.isfinite(grid["CES_VT"])]["time"].to_numpy()
        for diag in DIAGS:
            meta, _ = build_table(shot, diag)
            inside = ((lab >= meta["t0"]) & (lab <= meta["t1"])).mean() if len(lab) else 0.0
            meta["labelled_rows_covered_frac"] = float(inside)
            rows.append(meta)
    dst = RAW_DIR / "delivery_manifest.json"
    dst.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"[raw] wrote {dst}")
    print("\nshot   diag   fs_kHz        t0 ..      t1   span_s   nan  labelled_cov")
    for m in rows:
        print(f"{m['shot']}  {m['diag']:4s}  {m['fs_hz']/1e3:7.1f}  {m['t0']:8.3f} .. "
              f"{m['t1']:7.3f}  {m['t1']-m['t0']:6.3f}  {m['n_nan']:4d}  "
              f"{m['labelled_rows_covered_frac']:.3f}")


if __name__ == "__main__":
    main()
