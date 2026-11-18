"""B.68 (PREREGISTRATION_B68.md section 3): the auxiliary head and loss, installed into seq/train_seq.py at run time.

Change to the network.  SeqCESLSTMv2Aux is seq_v2 (model_seq_v2.SeqCESLSTMv2, same defaults) plus one more output
head `head_aux` of the same form as its T_i head (Linear(160 -> 64), GELU, Linear(64 -> 1)) on the same hidden
state (the LayerNorm output of the T_i LSTM).  The head is created after seq_v2's own layers, so under the same
CES_INIT_SEED the seq_v2 part starts from exactly the weights plain seq_v2 starts from.  `forward` is seq_v2's
(main heads only); `forward_with_aux` returns (main, aux).  `state_dict()` leaves the auxiliary head out and
`load_state_dict` takes a plain seq_v2 dict (strict on every seq_v2 key), so the in-loop best-state snapshot, its
restore and the saved checkpoint of train_seq are plain seq_v2: eval_seq loads it as v2 with strict loading.

Loss.  masked MSE of seq_v2 (both main heads) + 0.1 x the T_i positivity penalty (train_seq, unchanged) + AUX_WEIGHT
x the masked MSE of the auxiliary head on the auxiliary target (standardized with the run's target statistics;
rows without a target are masked).  AUX_WEIGHT = 0.5 is fixed by the registration.  The value returned to
train_seq.main (and so the early stopping, the checkpoint choice and the learning-rate plateau) is the masked MSE of
the main heads only, exactly as in seq_v2.

Hooks (install_train), all module globals of train_seq looked up inside main() at call time:
  SEQ_MODELS       rebound to a copy whose "v2" entry is SeqCESLSTMv2Aux (the shared registry is not touched)
  load_grid_files  wrapped: returns the grid unchanged and remembers which array is which discharge
  build_blocks     wrapped: the original blocks plus "aux" / "aux_mask", aligned row by row by time (checked)
  masked_pass      replaced by masked_pass_aux (batched_aux = train_seq.batched plus the two auxiliary arrays)
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import target_b68 as T  # noqa: E402  (puts seq/ and ces_prediction/ on sys.path)

from model_seq_v2 import SeqCESLSTMv2  # noqa: E402

AUX_WEIGHT = 0.5
AUX_PREFIX = "head_aux."
ENV_REQUIRED = {"CES_SEQ_MODEL": "v2", "CES_SEQ_LOSS": "mse", "CES_SEQ_SELECT": "mse", "CES_DROP_STUCK_TARGETS": "1",
                "CES_SEQ_FIXED_EPOCHS": "0"}


class SeqCESLSTMv2Aux(SeqCESLSTMv2):
    def __init__(self, **kw):
        super().__init__(**kw)
        hidden, head = self.lstm_ti.hidden_size, self.head_ti[0].out_features
        self.head_aux = nn.Sequential(nn.Linear(hidden, head), nn.GELU(), nn.Linear(head, 1))
        self.n_params_aux = sum(p.numel() for p in self.head_aux.parameters())

    def forward_with_aux(self, x, lengths=None):
        """(main (B, L, 2), aux (B, L)): seq_v2's forward with the auxiliary head on the T_i hidden state."""
        vt_end = self.n_base + self.n_extra_vt
        x_ti = x[..., :self.n_base]
        if self.n_extra_ti:
            x_ti = torch.cat([x_ti, x[..., vt_end:vt_end + self.n_extra_ti]], dim=-1)
        h_ti = self._run(self.lstm_ti, self.norm_ti, x_ti, lengths)
        h_vt = self._run(self.lstm_vt, self.norm_vt, x[..., self.n_fast:vt_end], lengths)
        return torch.cat([self.head_ti(h_ti), self.head_vt(h_vt)], dim=-1), self.head_aux(h_ti)[..., 0]

    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        prefix = kwargs.get("prefix", args[1] if len(args) > 1 else "")
        for k in [k for k in sd if k.startswith(prefix + AUX_PREFIX)]:
            del sd[k]
        return sd

    def full_state_dict(self):
        return super().state_dict()

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if any(k.startswith(AUX_PREFIX) for k in state_dict):
            raise RuntimeError("SeqCESLSTMv2Aux.load_state_dict takes a plain seq_v2 state dict")
        res = super().load_state_dict(state_dict, strict=False, assign=assign)
        missing = [k for k in res.missing_keys if not k.startswith(AUX_PREFIX)]
        if strict and (missing or res.unexpected_keys):
            raise RuntimeError(f"seq_v2 keys missing {missing}, unexpected {res.unexpected_keys}")
        return res


def check_env(env=None):
    env = os.environ if env is None else env
    bad = {k: env.get(k) for k, v in ENV_REQUIRED.items() if env.get(k) != v}
    if float(env.get("CES_TI_SPIKE_CUT_EV", "0") or 0) != T.SPIKE_CUT_EV:
        bad["CES_TI_SPIKE_CUT_EV"] = env.get("CES_TI_SPIKE_CUT_EV")
    if bad:
        raise SystemExit(f"FATAL: B.68 trains seq_v2 under the env of run_b39.pooled_env only; differs: {bad}")


# ------------------------------------------------------------------------------ blocks
def attach_aux(blocks, arr, name, stats, entry, counts=None):
    """The blocks of one discharge (seq_data.build_blocks of `arr`) plus aux (L,) float32 = the cached target of
    each block row standardized with stats["target"] of T_i (0 where there is none) and aux_mask (L,) float32.
    Each block is located in `arr` by its exact float32 times; its T_i mask must equal the cache's T_i."""
    T.verify_against_grid(entry, arr, name)
    t = arr[:, T.TIME_COL]
    if len(t) > 1 and np.any(np.diff(t.astype(np.float64)) <= 0):
        raise SystemExit(f"FATAL: {name}: grid times are not strictly increasing")
    mean = float(np.asarray(stats["target"]["mean"], dtype=np.float64)[0])
    std = float(np.asarray(stats["target"]["std"], dtype=np.float64)[0])
    z = (entry["aux"].astype(np.float64) - mean) / std
    ok = np.isfinite(z)
    out, p = [], 0
    for b in blocks:
        L = len(b["time"])
        a0 = int(np.searchsorted(t, b["time"][0]))
        if a0 < p or a0 + L > len(t) or not np.array_equal(t[a0:a0 + L], b["time"]):
            raise SystemExit(f"FATAL: {name}: a block could not be located in the grid rows")
        if not np.array_equal(b["mask"][:, 0] > 0, np.isfinite(entry["ti"][a0:a0 + L])):
            raise SystemExit(f"FATAL: {name}: the block's T_i mask differs from the cache's T_i")
        p = a0 + L
        o = ok[a0:a0 + L]
        out.append({**b, "aux": np.where(o, z[a0:a0 + L], 0.0).astype(np.float32), "aux_mask": o.astype(np.float32)})
        if counts is not None:
            c = counts.setdefault(name, [0, 0, 0, 0])
            ti = b["mask"][:, 0] > 0
            c[0] += L
            c[1] += int(o.sum())
            c[2] += int(ti.sum())
            c[3] += int((o & ti).sum())
    return out


def batched_aux(blocks, batch_size, device, shuffle, rng):
    """train_seq.batched (same order, same tensors) plus aux (B, L) and aux_mask (B, L)."""
    order = list(range(len(blocks)))
    if shuffle:
        rng.shuffle(order)
        order.sort(key=lambda i: blocks[i]["x"].shape[0] // 256)  # coarse length bucketing
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        L = max(blocks[i]["x"].shape[0] for i in idx)
        B = len(idx)
        x = torch.zeros(B, L, blocks[0]["x"].shape[1])
        y = torch.zeros(B, L, 2)
        m = torch.zeros(B, L, 2)
        a = torch.zeros(B, L)
        am = torch.zeros(B, L)
        lengths = torch.zeros(B, dtype=torch.long)
        for j, i in enumerate(idx):
            n = blocks[i]["x"].shape[0]
            x[j, :n] = torch.from_numpy(blocks[i]["x"])
            y[j, :n] = torch.from_numpy(blocks[i]["y"])
            m[j, :n] = torch.from_numpy(blocks[i]["mask"])
            a[j, :n] = torch.from_numpy(blocks[i]["aux"])
            am[j, :n] = torch.from_numpy(blocks[i]["aux_mask"])
            lengths[j] = n
        yield x.to(device), y.to(device), m.to(device), lengths, a.to(device), am.to(device)


# ------------------------------------------------------------------------------ loss
def batch_terms(model, x, y, m, lengths, a, am, zero_ti, weight):
    """train_seq's mse objective (masked MSE + 0.1 x penalty) + weight x masked MSE of the auxiliary head."""
    out, aux_out = model.forward_with_aux(x, lengths)
    se = ((out - y) ** 2) * m
    mse = se.sum() / m.sum().clamp(min=1.0)
    pen = (torch.relu(zero_ti - out[..., 0]) * m[..., 0]).sum() / m[..., 0].sum().clamp(min=1.0)
    aux_se = ((aux_out - a) ** 2) * am
    aux_mse = aux_se.sum() / am.sum().clamp(min=1.0)
    loss = mse + 0.1 * pen + weight * aux_mse
    return {"loss": loss, "out": out, "aux_out": aux_out, "se": se, "mse": mse, "pen": pen, "aux_se": aux_se,
            "aux_mse": aux_mse}


def make_masked_pass(weight, log):
    """A replacement for train_seq.masked_pass (same signature and return values) that adds the auxiliary term to
    the back-propagated loss.  Returns (masked MSE of the main heads, mean penalty, the same masked MSE)."""
    def masked_pass(model, blocks, batch_size, device, zero_ti, optimizer=None, clip=1.0, rng=None,
                    loss_cfg=None, abs_err=None):
        if (loss_cfg or {}).get("kind", "mse") != "mse" or abs_err is not None:
            raise SystemExit("FATAL: the B.68 pass is defined for CES_SEQ_LOSS=mse and CES_SEQ_SELECT=mse only")
        if not hasattr(model, "forward_with_aux") or getattr(model, "needs_obs", False):
            raise SystemExit(f"FATAL: the B.68 pass needs SeqCESLSTMv2Aux, got {type(model).__name__}")
        training = optimizer is not None
        model.train(training)
        se_sum, n_obs, pen_sum, n_batches, aux_sum, aux_n = 0.0, 0.0, 0.0, 0, 0.0, 0.0
        for x, y, m, lengths, a, am in batched_aux(blocks, batch_size, device, training, rng or random.Random(0)):
            with torch.set_grad_enabled(training):
                t = batch_terms(model, x, y, m, lengths, a, am, zero_ti, weight)
                if training:
                    optimizer.zero_grad()
                    t["loss"].backward()
                    nn.utils.clip_grad_norm_(model.parameters(), clip)
                    optimizer.step()
            se_sum += float(t["se"].detach().sum())
            n_obs += float(m.sum())
            pen_sum += float(t["pen"].detach())
            n_batches += 1
            aux_sum += float(t["aux_se"].detach().sum())
            aux_n += float(am.sum())
        main = se_sum / max(n_obs, 1.0)
        log.append({"phase": "train" if training else "val", "main_masked_mse": main,
                    "aux_masked_mse": aux_sum / max(aux_n, 1.0), "aux_rows": int(aux_n), "label_elements": int(n_obs),
                    "batches": n_batches})
        print(f"[b68] {'train' if training else 'val  '} main_mse={main:.4f} aux_mse={aux_sum / max(aux_n, 1.0):.4f} "
              f"(aux rows {int(aux_n):,}, weight {weight})", flush=True)
        return main, pen_sum / max(n_batches, 1), main
    return masked_pass


# ------------------------------------------------------------------------------ installation
def install_train(weight=AUX_WEIGHT):
    """Install the four hooks into train_seq; returns (train_seq module, state dict filled during main())."""
    import seq_data
    import seq_models
    import train_seq
    for nm in ("build_blocks", "load_grid_files"):
        if getattr(train_seq, nm) is not getattr(seq_data, nm):
            raise SystemExit(f"FATAL: train_seq.{nm} is not seq_data.{nm}; the B.68 hook would miss it")
    if train_seq.SEQ_MODELS is not seq_models.SEQ_MODELS or train_seq.masked_pass.__module__ != "train_seq":
        raise SystemExit("FATAL: train_seq's registry or masked_pass is not the original")
    cache, meta = T.load_cache()
    state = {"weight": weight, "cache_meta": meta, "names_by_id": {}, "counts": {}, "epochs": [], "grid_calls": 0,
             "block_calls": 0}
    orig_load, orig_build = train_seq.load_grid_files, train_seq.build_blocks

    def load_grid_files(data_dir, drop_stuck, *a, **k):
        if bool(drop_stuck) != T.DROP_STUCK:
            raise SystemExit("FATAL: the B.68 target was built held-free; training is not")
        grid, dims = orig_load(data_dir, drop_stuck, *a, **k)
        if int(dims.get("extra_vt", 0) or 0) or int(dims.get("extra_ti", 0) or 0):
            raise SystemExit("FATAL: B.68 uses the base 22-channel layout only")
        state["grid_calls"] += 1
        state["names_by_id"] = {id(arr): name for name, arr in grid.items()}
        return grid, dims

    def build_blocks(arr, dims, stats, *a, **k):
        blks = orig_build(arr, dims, stats, *a, **k)
        name = state["names_by_id"].get(id(arr))
        if name is None or name not in cache:
            raise SystemExit(f"FATAL: build_blocks got an array that is not a cached discharge ({name})")
        state["block_calls"] += 1
        return attach_aux(blks, arr, name, stats, cache[name], state["counts"])

    train_seq.load_grid_files = load_grid_files
    train_seq.build_blocks = build_blocks
    train_seq.SEQ_MODELS = {**seq_models.SEQ_MODELS, "v2": SeqCESLSTMv2Aux}
    train_seq.masked_pass = make_masked_pass(weight, state["epochs"])
    return train_seq, state
