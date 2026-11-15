"""Download FAIR-MAST level-1 charge-exchange arrays and beam power, one npz per shot.

Source: https://s3.echo.stfc.ac.uk/mast/level1/shots/{shot}.zarr (anonymous, CC BY-SA 4.0).
Kept per shot: act/{time, majorradius, ss_temperature(+_error), ss_velocity(+_error),
ss_chisq, ss_cx_emissivity, ss_cx2bgratio} and anb/time plus the anb/*sum_power traces (beam power).
Spectra are not downloaded. The script prints byte counts only, no measured values.
Resumable: a shot with an existing npz (or a recorded permanent miss) is skipped.

Run from the repo root:
  py ces_prediction/experiments/mast/fetch_mast.py gate_shots.txt [all_shots.txt] [--workers 6]
"""
import itertools
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numcodecs
import numpy as np

BASE = "https://s3.echo.stfc.ac.uk/mast/level1/shots/"
OUT = Path("data/mast/act")
HERE = Path(__file__).parent
ACT = ["time", "majorradius", "ss_temperature", "ss_temperature_error", "ss_velocity",
       "ss_velocity_error", "ss_chisq", "ss_cx_emissivity", "ss_cx2bgratio"]
ACT_REQUIRED = ACT[:6]
MAX_ANB_SAMPLES = 2_000_000


def get(url, tries=4):
    err = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "python-client"})
            return urllib.request.urlopen(req, timeout=120).read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                raise
            err = e
        except Exception as e:  # timeouts, resets
            err = e
        time.sleep(3 * (k + 1))
    raise err


def read_array(shot, md, name):
    za = md[name + "/.zarray"]
    shape, chunks = za["shape"], za["chunks"]
    dtype = np.dtype(za["dtype"])
    codec = numcodecs.get_codec(za["compressor"]) if za["compressor"] else None
    filters = [numcodecs.get_codec(f) for f in (za.get("filters") or [])]
    sep = za.get("dimension_separator", ".")
    fill = za.get("fill_value")
    if dtype.kind == "f":
        out = np.full(shape, np.nan if fill in (None, "NaN") else fill, dtype=dtype)
    else:
        out = np.full(shape, fill or 0, dtype=dtype)
    n = 0
    grid = [range(-(-s // c)) for s, c in zip(shape, chunks)]
    for idx in itertools.product(*grid):
        key = sep.join(map(str, idx)) if idx else "0"
        try:
            raw = get(f"{BASE}{shot}.zarr/{name}/{key}")
        except urllib.error.HTTPError:
            continue  # missing chunk = fill value
        n += len(raw)
        if codec:
            raw = codec.decode(raw)
        for f in reversed(filters):
            raw = f.decode(raw)
        block = np.frombuffer(raw, dtype=dtype).reshape(chunks, order=za.get("order", "C"))
        sl = tuple(slice(i * c, min((i + 1) * c, s)) for i, c, s in zip(idx, chunks, shape))
        out[sl] = block[tuple(slice(0, s.stop - s.start) for s in sl)]
    return out, n


def fetch(shot):
    dst = OUT / f"{shot}.npz"
    miss = OUT / f"{shot}.miss"
    if dst.exists() or miss.exists():
        return shot, "skip", 0
    try:
        raw = get(f"{BASE}{shot}.zarr/.zmetadata")
    except urllib.error.HTTPError as e:
        miss.write_text(f"no store ({e.code})")
        return shot, "miss", 0
    md = json.loads(raw)["metadata"]
    nbytes = len(raw)
    if any(f"act/{k}/.zarray" not in md for k in ACT_REQUIRED):
        miss.write_text("act arrays absent")
        return shot, "miss", nbytes
    d = {}
    for k in ACT:
        if f"act/{k}/.zarray" in md:
            d[f"act_{k}"], n = read_array(shot, md, f"act/{k}")
            nbytes += n
    for key in md:
        if key.startswith("anb/") and key.endswith("/.zarray"):
            za = md[key]
            keep = key in ("anb/time/.zarray",) or key.endswith("sum_power/.zarray")
            if keep and len(za["shape"]) == 1 and za["shape"][0] <= MAX_ANB_SAMPLES:
                name = key[: -len("/.zarray")]
                d[name.replace("/", "_")], n = read_array(shot, md, name)
                nbytes += n
    d["groups"] = np.array(sorted({k.split("/")[0] for k in md if "/" in k}))
    tmp = OUT / f"{shot}.tmp.npz"
    np.savez_compressed(tmp, **d)
    tmp.replace(dst)
    return shot, "ok", nbytes


def main():
    args = sys.argv[1:]
    workers = 6
    if "--workers" in args:
        i = args.index("--workers")
        workers = int(args[i + 1])
        del args[i:i + 2]
    shots = []
    for f in args:
        shots += [int(x) for x in (HERE / f).read_text().split()]
    OUT.mkdir(parents=True, exist_ok=True)
    t0, total, cnt = time.time(), 0, {"ok": 0, "skip": 0, "miss": 0, "err": 0}
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(fetch, s): s for s in shots}
        for i, fu in enumerate(as_completed(futs), 1):
            try:
                shot, st, n = fu.result()
            except Exception as e:
                shot, st, n = futs[fu], "err", 0
                print(f"shot {shot} ERR {repr(e)[:120]}", flush=True)
            cnt[st] += 1
            total += n
            if i % 25 == 0 or i == len(shots):
                print(f"{i}/{len(shots)} {cnt} {total / 1e6:.1f} MB {time.time() - t0:.0f} s", flush=True)
    print("done", cnt, f"{total / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
