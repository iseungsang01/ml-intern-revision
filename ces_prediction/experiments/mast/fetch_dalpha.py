"""Download the mid-plane D-alpha trace of FAIR-MAST level 1 for the coupling study.

Source: https://s3.echo.stfc.ac.uk/mast/level1/shots/{shot}.zarr (anonymous, CC BY-SA 4.0),
arrays xim/da_hm10_t ("Dalpha - mid-plane wide-angle view from port HM10", volts) and xim/time.
One npz per shot in data/mast/dalpha/. Only the shots listed in coupling_shots.txt are read.
Prints counts and byte totals only, no measured value. Resumable: a shot with an npz or a
recorded miss is skipped. Stops when free disk space falls below 20 GB.

Run from the repo root:  py ces_prediction/experiments/mast/fetch_dalpha.py [--workers 8]
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
OUT = Path("data/mast/dalpha")
HERE = Path(__file__).parent
NAMES = {"dalpha": "xim/da_hm10_t", "time": "xim/time"}
MIN_FREE_GB = 20.0

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


def read_1d(shot, name):
    za = json.loads(get(f"{BASE}{shot}.zarr/{name}/.zarray"))
    if len(za["shape"]) != 1 or za["chunks"][0] < za["shape"][0]:
        raise Missing("not a single-chunk 1-D array")
    raw = get(f"{BASE}{shot}.zarr/{name}/0")
    n = len(raw)
    if za["compressor"]:
        raw = numcodecs.get_codec(za["compressor"]).decode(raw)
    a = np.frombuffer(raw, dtype=np.dtype(za["dtype"]))[: za["shape"][0]]
    return a.astype(np.float32), n


def fetch(shot):
    dst, miss = OUT / f"{shot}.npz", OUT / f"{shot}.miss"
    if dst.exists() or miss.exists():
        return "skip", 0
    try:
        d, total = {}, 0
        for key, name in NAMES.items():
            d[key], n = read_1d(shot, name)
            total += n
    except Missing as e:
        miss.write_text(f"absent ({e})")
        return "miss", 0
    if d["dalpha"].shape != d["time"].shape:
        miss.write_text("time and signal lengths differ")
        return "miss", total
    tmp = OUT / f"{shot}.tmp.npz"
    np.savez_compressed(tmp, **d)
    tmp.replace(dst)
    return "ok", total


def main():
    args = sys.argv[1:]
    workers = int(args[args.index("--workers") + 1]) if "--workers" in args else 8
    shots = [int(x) for x in (HERE / "coupling_shots.txt").read_text().split()]
    OUT.mkdir(parents=True, exist_ok=True)
    t0, total, cnt = time.time(), 0, {"ok": 0, "skip": 0, "miss": 0, "err": 0}
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(fetch, s): s for s in shots}
        for i, fu in enumerate(as_completed(futs), 1):
            try:
                st, n = fu.result()
            except Exception as e:
                st, n = "err", 0
                print(f"shot {futs[fu]} ERR {repr(e)[:100]}", flush=True)
            cnt[st] += 1
            total += n
            if i % 200 == 0 or i == len(shots):
                free = shutil.disk_usage(OUT).free / 1e9
                print(f"{i}/{len(shots)} {cnt} {total / 1e6:.1f} MB {time.time() - t0:.0f} s free {free:.0f} GB",
                      flush=True)
                if free < MIN_FREE_GB:
                    print("stopping: free disk below 20 GB", flush=True)
                    ex.shutdown(wait=False, cancel_futures=True)
                    break
    print("done", cnt, f"{total / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
