"""Download small level-1 arrays of FAIR-MAST for the fast-signal exploration.

Source: https://s3.echo.stfc.ac.uk/mast/level1/shots/{shot}.zarr (anonymous, CC BY-SA 4.0).
Groups, each optional per shot:
  ayc  core Thomson scattering (te, te_error, ne, radius, time); atm is read when ayc is absent
       (the earlier system, superseded by ayc)
  aye  edge Thomson scattering (te, te_error, ne, r, time, time_ when present)
  ane  CO2 interferometer line-integrated density (density, time)
  asm  saddle-coil locked-mode signal and H-mode rating (out_signal, out_time, hm_rating, hm_time)
One npz per shot in data/mast/extra/ with keys "{group}_{array}"; a shot with none of the groups
gets a .miss file.  Only the shots listed in coupling_shots.txt are read.  Prints counts and byte
totals only, no measured value.  Resumable.  Stops when free disk space falls below 20 GB.

Run from the repo root:  py ces_prediction/experiments/mast/fetch_extra.py [--workers 16]
"""
import http.client
import json
import shutil
import socket
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numcodecs
import numpy as np

HOST = "s3.echo.stfc.ac.uk"
BASE = "/mast/level1/shots/"
OUT = Path("data/mast/extra")
HERE = Path(__file__).parent
MIN_FREE_GB = 20.0
OPTIONAL = {"time_"}                 # kept when present, the group is not dropped without it
TS = ("te", "te_error", "ne", "radius", "time")
GROUPS = {
    "ayc": {k: f"ayc/{k}" for k in TS},
    "atm": {k: f"atm/{k}" for k in TS},
    "aye": {"te": "aye/te", "te_error": "aye/te_error", "ne": "aye/ne", "radius": "aye/r",
            "time": "aye/time", "time_": "aye/time_"},
    "ane": {"density": "ane/density", "time": "ane/time"},
    "asm": {"out_signal": "asm/out_signal", "out_time": "asm/out_time",
            "hm_rating": "asm/hm_rating", "hm_time": "asm/hm_time"},
}

_orig_getaddrinfo = socket.getaddrinfo
socket.getaddrinfo = lambda h, p, f=0, *a, **k: _orig_getaddrinfo(h, p, socket.AF_INET, *a, **k)
_local = threading.local()


class Missing(Exception):
    pass


def get(path, tries=5):
    err = None
    for k in range(tries):
        try:
            c = getattr(_local, "conn", None)
            if c is None:
                c = _local.conn = http.client.HTTPSConnection(HOST, timeout=60)
            c.request("GET", path, headers={"User-Agent": "python-client"})
            r = c.getresponse()
            body = r.read()
            if r.status in (403, 404):
                raise Missing(str(r.status))
            if r.status != 200:
                raise OSError(f"status {r.status}")
            return body
        except Missing:
            raise
        except Exception as e:  # reset, timeout: reconnect and retry
            err = e
            try:
                _local.conn.close()
            except Exception:
                pass
            _local.conn = None
            time.sleep(2 * (k + 1))
    raise err


def read_array(shot, name):
    """A single-chunk zarr v2 array of any rank as float32, and the bytes transferred."""
    za = json.loads(get(f"{BASE}{shot}.zarr/{name}/.zarray"))
    shape, chunks = za["shape"], za["chunks"]
    if not shape or any(c < s for c, s in zip(chunks, shape)):
        raise Missing("scalar or multi-chunk array")
    sep = za.get("dimension_separator", ".")
    raw = get(f"{BASE}{shot}.zarr/{name}/{sep.join('0' for _ in shape)}")
    n = len(raw)
    if za.get("compressor"):
        raw = numcodecs.get_codec(za["compressor"]).decode(raw)
    for f in reversed(za.get("filters") or []):
        raw = numcodecs.get_codec(f).decode(raw)
    a = np.frombuffer(raw, dtype=np.dtype(za["dtype"])).reshape(chunks, order=za.get("order", "C"))
    return a[tuple(slice(0, s) for s in shape)].astype(np.float32), n


def fetch(shot):
    dst, miss = OUT / f"{shot}.npz", OUT / f"{shot}.miss"
    if dst.exists() or miss.exists():
        return "skip", 0, ()
    d, total, got = {}, 0, []
    for g, names in GROUPS.items():
        if g == "atm" and "ayc" in got:
            continue
        part = {}
        try:
            for key, name in names.items():
                try:
                    part[key], n = read_array(shot, name)
                    total += n
                except Missing:
                    if key not in OPTIONAL:
                        raise
        except Missing:
            continue
        got.append(g)
        for key, a in part.items():
            d[f"{g}_{key}"] = a
    if not got:
        miss.write_text("none of the groups present")
        return "miss", total, ()
    tmp = OUT / f"{shot}.tmp.npz"
    np.savez_compressed(tmp, **d)
    tmp.replace(dst)
    return "ok", total, tuple(got)


def main():
    args = sys.argv[1:]
    workers = int(args[args.index("--workers") + 1]) if "--workers" in args else 16
    shots = [int(x) for x in (HERE / "coupling_shots.txt").read_text().split()]
    if "--only" in args:
        shots = [int(x) for x in args[args.index("--only") + 1].split(",")]
    OUT.mkdir(parents=True, exist_ok=True)
    t0, total = time.time(), 0
    cnt = {"ok": 0, "skip": 0, "miss": 0, "err": 0}
    grp = {g: 0 for g in GROUPS}
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(fetch, s): s for s in shots}
        for i, fu in enumerate(as_completed(futs), 1):
            try:
                st, n, got = fu.result()
            except Exception as e:
                st, n, got = "err", 0, ()
                print(f"shot {futs[fu]} ERR {repr(e)[:100]}", flush=True)
            cnt[st] += 1
            total += n
            for g in got:
                grp[g] += 1
            if i % 200 == 0 or i == len(shots):
                free = shutil.disk_usage(OUT).free / 1e9
                print(f"{i}/{len(shots)} {cnt} {grp} {total / 1e6:.1f} MB {time.time() - t0:.0f} s "
                      f"free {free:.0f} GB", flush=True)
                if free < MIN_FREE_GB:
                    print("stopping: free disk below 20 GB", flush=True)
                    ex.shutdown(wait=False, cancel_futures=True)
                    break
    print("done", cnt, grp, f"{total / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
