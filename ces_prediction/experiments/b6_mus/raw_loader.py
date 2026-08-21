"""The ONE on-arrival slot with code behind it: reading the delivered raw files.

Everything downstream of this module is already written and unit-tested against
synthetic signals; this is the format-specific boundary that cannot be written until
the delivery format (MDSplus dump / HDF5 / CSV / ...) is known. Fill in `load_raw`,
flip `RAW_READY`, and `mus_features.py` / `raw_positive_controls.py` run unchanged.

Contract:
    load_raw(shot: int, channel: str) -> ((t, x), fs)
        t   float64 seconds, monotonically increasing, same clock as the shot CSVs'
            `time` column (KSTAR discharge time)
        x   float64 samples
        fs  float, the channel's sampling rate in Hz, read from delivery metadata --
            never inferred silently from np.diff(t) alone (verify one against the
            other and abort on disagreement > 1%)

Do NOT resample, filter, or detrend here: the extractor owns every numerical choice
(PREREGISTRATION_B6.md sec. 3.1 implementation constants), and the quality gate
(sec. 1.4) needs the stream exactly as delivered.
"""

RAW_READY = False


def load_raw(shot, channel):
    raise NotImplementedError(
        "raw_loader.load_raw is the PREREGISTRATION_B6.md on-arrival slot: implement it "
        f"for the delivered format before extracting features (asked for s{shot}/{channel})."
    )
