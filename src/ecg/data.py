"""Data loading for the MIT-BIH Arrhythmia Database.

Turns raw ECG recordings into fixed-length heartbeat windows with AAMI class
labels, and builds two kinds of train/test split:

* ``patient``: the standard inter-patient split (de Chazal et al., 2004).
  No patient appears in both train and test, so the score reflects how the
  model works on a new person.
* ``beat``: beats from all records are shuffled together (what many papers,
  and my 2020 thesis, did). Beats from one patient land in both sets, which
  inflates the score. Kept only to measure that leakage.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# AAMI EC57 grouping of MIT-BIH beat symbols into 5 classes.
AAMI_CLASSES = ["N", "S", "V", "F", "Q"]
SYMBOL_TO_CLASS = {
    "N": 0, "L": 0, "R": 0, "e": 0, "j": 0,   # N: normal and bundle branch block
    "A": 1, "a": 1, "J": 1, "S": 1,           # S: supraventricular ectopic
    "V": 2, "E": 2,                           # V: ventricular ectopic
    "F": 3,                                   # F: fusion
    "/": 4, "f": 4, "Q": 4,                   # Q: paced and unclassifiable
}

# de Chazal et al. (2004) inter-patient split. Paced records 102, 104, 107, 217 excluded.
DS1_TRAIN = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122,
             124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
DS2_TEST = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210,
            212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]


@dataclass
class BeatSet:
    """Heartbeat windows plus labels and the record (patient) each came from."""
    X: np.ndarray        # (n_beats, window) float32
    y: np.ndarray        # (n_beats,) int64 class index
    record: np.ndarray   # (n_beats,) int record number


def extract_beats(signal: np.ndarray, peaks: np.ndarray, symbols: list[str],
                  pre: int, post: int) -> tuple[np.ndarray, np.ndarray]:
    """Cut a window around each annotated R-peak and map its symbol to a class.

    Beats whose symbol is not a heartbeat (rhythm or noise markers) and beats too
    close to the start or end of the recording are skipped. Each window is
    z-score normalised on its own, so amplitude differences between patients
    matter less.
    """
    windows, labels = [], []
    for peak, sym in zip(peaks, symbols):
        if sym not in SYMBOL_TO_CLASS:
            continue
        start, end = peak - pre, peak + post
        if start < 0 or end > len(signal):
            continue
        w = signal[start:end].astype(np.float32)
        w = (w - w.mean()) / (w.std() + 1e-6)
        windows.append(w)
        labels.append(SYMBOL_TO_CLASS[sym])
    if not windows:
        return np.zeros((0, pre + post), np.float32), np.zeros(0, np.int64)
    return np.stack(windows), np.array(labels, dtype=np.int64)


def load_record(record: int, data_dir: Path, pre: int, post: int) -> tuple[np.ndarray, np.ndarray]:
    """Read one MIT-BIH record from disk and return its beats from lead MLII."""
    import wfdb  # imported here so tests without wfdb data still run

    path = str(Path(data_dir) / str(record))
    rec = wfdb.rdrecord(path)
    ann = wfdb.rdann(path, "atr")
    # Lead MLII is channel 0 in almost every record, but not all (e.g. 114).
    lead = rec.sig_name.index("MLII") if "MLII" in rec.sig_name else 0
    return extract_beats(rec.p_signal[:, lead], ann.sample, ann.symbol, pre, post)


def download_mitdb(data_dir: Path) -> None:
    """Download the MIT-BIH database from PhysioNet (about 100 MB) if missing."""
    import wfdb

    data_dir = Path(data_dir)
    if (data_dir / "100.hea").exists():
        print(f"MIT-BIH already in {data_dir}")
        return
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MIT-BIH to {data_dir} ...")
    wfdb.dl_database("mitdb", dl_dir=str(data_dir))


def load_records(records: list[int], data_dir: Path, pre: int, post: int) -> BeatSet:
    Xs, ys, rs = [], [], []
    for r in records:
        X, y = load_record(r, data_dir, pre, post)
        Xs.append(X); ys.append(y); rs.append(np.full(len(y), r))
    return BeatSet(np.concatenate(Xs), np.concatenate(ys), np.concatenate(rs))


def make_synthetic(records: list[int], beats_per_record: int, window: int, seed: int) -> BeatSet:
    """Fake ECG-like beats with patient-specific shapes, for tests and dry runs.

    Each record gets its own baseline wave and mostly two beat classes, like
    real recordings, so patient leakage shows up the same way it does on MIT-BIH.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, window)
    centres, widths = [0.50, 0.42, 0.58, 0.45, 0.55], [0.03, 0.06, 0.08, 0.05, 0.10]
    share = np.array([0.80, 0.08, 0.07, 0.02, 0.03])
    Xs, ys, rs = [], [], []
    for r in records:
        signature = 0.5 * np.sin(2 * np.pi * rng.uniform(1, 4) * t + rng.uniform(0, 6))
        main = rng.choice(5, size=2, replace=False, p=share)
        p = share.copy(); p[main] *= 10; p /= p.sum()
        for _ in range(beats_per_record):
            c = int(rng.choice(5, p=p))
            beat = np.exp(-((t - centres[c]) ** 2) / (2 * widths[c] ** 2)) + signature
            beat = beat + rng.normal(0, 0.35, window)
            beat = (beat - beat.mean()) / (beat.std() + 1e-6)
            Xs.append(beat.astype(np.float32)); ys.append(c); rs.append(r)
    return BeatSet(np.stack(Xs), np.array(ys, np.int64), np.array(rs))


def split_by_record(data: BeatSet, test_records: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Index arrays for a patient-wise split: test = beats from ``test_records``."""
    is_test = np.isin(data.record, test_records)
    return np.where(~is_test)[0], np.where(is_test)[0]


def split_by_beat(data: BeatSet, test_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Index arrays for a stratified random beat-level split (leaky on purpose)."""
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(data.y))
    tr, te = train_test_split(idx, test_size=test_fraction, stratify=data.y, random_state=seed)
    return np.sort(tr), np.sort(te)


def validation_records(train_records: list[int], n_val: int, seed: int) -> list[int]:
    """Hold out a few whole training records for early stopping (no patient overlap)."""
    rng = np.random.default_rng(seed)
    return sorted(int(r) for r in rng.choice(train_records, size=n_val, replace=False))
