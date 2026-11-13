"""Train and evaluate one ECG model from the command line.

Examples
--------
python -m ecg.train --config configs/default.yaml
python -m ecg.train --config configs/default.yaml --split beat --seed 1
python -m ecg.train --config configs/default.yaml --synthetic        # no download, quick check
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

from . import data as D
from .metrics import bootstrap_ci, summary
from .model import ECGNet, count_parameters


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def class_weights(y: np.ndarray, n_classes: int, mode: str) -> torch.Tensor | None:
    """Loss weights so rare classes count more. 'sqrt_inverse' is gentler than 'inverse'."""
    if mode == "none":
        return None
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / (n_classes * counts)
    if mode == "sqrt_inverse":
        w = np.sqrt(w)
    elif mode != "inverse":
        raise ValueError(f"unknown class_weighting: {mode}")
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def build_splits(cfg: dict, synthetic: bool, split: str, seed: int):
    """Return (data, train_idx, val_idx, test_idx)."""
    pre, post = cfg["data"]["pre"], cfg["data"]["post"]
    records = D.DS1_TRAIN + D.DS2_TEST
    if synthetic:
        data = D.make_synthetic(records, cfg["data"]["synthetic_beats_per_record"], pre + post, seed=0)
    else:
        data_dir = Path(cfg["data"]["dir"])
        D.download_mitdb(data_dir)
        data = D.load_records(records, data_dir, pre, post)

    if split == "patient":
        train_idx, test_idx = D.split_by_record(data, D.DS2_TEST)
        val_recs = D.validation_records(D.DS1_TRAIN, cfg["data"]["n_val_records"], seed)
        is_val = np.isin(data.record[train_idx], val_recs)
        val_idx, train_idx = train_idx[is_val], train_idx[~is_val]
    elif split == "beat":
        train_idx, test_idx = D.split_by_beat(data, 0.2, seed)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(train_idx)
        n_val = len(perm) // 10
        val_idx, train_idx = np.sort(perm[:n_val]), np.sort(perm[n_val:])
    else:
        raise ValueError(f"unknown split: {split}")
    return data, train_idx, val_idx, test_idx


@torch.no_grad()
def predict(model: nn.Module, X: torch.Tensor, batch: int, device: str) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        out.append(model(X[i:i + batch].to(device)).argmax(1).cpu())
    return torch.cat(out).numpy()


def run(cfg: dict, split: str, seed: int, synthetic: bool, out_root: Path) -> dict:
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tcfg = cfg["train"]
    data, tr, va, te = build_splits(cfg, synthetic, split, seed)

    overlap = set(data.record[tr]) & set(data.record[te])
    print(f"split={split} seed={seed} device={device} | train {len(tr)} val {len(va)} test {len(te)} beats"
          f" | records in both train and test: {len(overlap)}")

    X = torch.tensor(data.X); y = torch.tensor(data.y)
    model = ECGNet(n_classes=len(D.AAMI_CLASSES), channels=tuple(cfg["model"]["channels"]),
                   kernel=cfg["model"]["kernel"], dropout=cfg["model"]["dropout"]).to(device)
    w = class_weights(data.y[tr], len(D.AAMI_CLASSES), tcfg["class_weighting"])
    loss_fn = nn.CrossEntropyLoss(weight=None if w is None else w.to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    use_amp = tcfg["amp"] and device == "cuda" and torch.cuda.is_bf16_supported()

    g = torch.Generator().manual_seed(seed)
    tr_t = torch.tensor(tr)
    best_f1, best_state, bad_epochs, history = -1.0, None, 0, []
    t0 = time.perf_counter()
    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        perm = tr_t[torch.randperm(len(tr_t), generator=g)]
        total, n = 0.0, 0
        for i in range(0, len(perm), tcfg["batch_size"]):
            b = perm[i:i + tcfg["batch_size"]]
            xb, yb = X[b].to(device), y[b].to(device)
            opt.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            opt.step()
            total += loss.item() * len(b); n += len(b)
        val_f1 = summary(data.y[va], predict(model, X[va], 1024, device))["macro_f1"]
        history.append({"epoch": epoch, "train_loss": total / n, "val_macro_f1": val_f1})
        print(f"  epoch {epoch:2d}  train loss {total / n:.4f}  val macro-F1 {val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1, bad_epochs = val_f1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= tcfg["patience"]:
                print(f"  early stop: no val improvement for {tcfg['patience']} epochs")
                break
    train_seconds = time.perf_counter() - t0

    model.load_state_dict(best_state)
    y_pred = predict(model, X[te], 1024, device)
    res = summary(data.y[te], y_pred)
    lo, hi = bootstrap_ci(data.y[te], y_pred, n_boot=cfg["eval"]["n_bootstrap"], seed=seed)
    res.update({
        "macro_f1_ci95": [lo, hi], "split": split, "seed": seed, "synthetic": synthetic,
        "class_weighting": tcfg["class_weighting"], "git_commit": git_commit(),
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "device": device, "parameters": count_parameters(model), "epochs_run": len(history),
        "train_seconds": round(train_seconds, 1), "best_val_macro_f1": best_f1,
        "records_in_train_and_test": len(overlap), "history": history,
    })

    run_dir = out_root / f"{res['timestamp'].replace(':', '')}_{split}_{tcfg['class_weighting']}_s{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps(res, indent=2))
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg))
    torch.save(best_state, run_dir / "model.pt")
    save_plots(res, run_dir)
    append_run_row(out_root / "runs.csv", res)

    print(f"TEST accuracy {res['accuracy']:.4f} | macro-F1 {res['macro_f1']:.4f} "
          f"(95% CI {lo:.3f}-{hi:.3f}) | saved to {run_dir}")
    for name, m in res["per_class"].items():
        print(f"  {name}: recall {m['recall']:.3f}  precision {m['precision']:.3f}  support {m['support']}")
    return res


def save_plots(res: dict, run_dir: Path) -> None:
    """Learning curve and a row-normalised confusion matrix, for the README."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = res["history"]
    fig, ax1 = plt.subplots(figsize=(5, 3.2))
    ax1.plot([e["epoch"] for e in h], [e["train_loss"] for e in h], marker="o", label="train loss")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("train loss")
    ax2 = ax1.twinx()
    ax2.plot([e["epoch"] for e in h], [e["val_macro_f1"] for e in h], marker="s", color="tab:orange",
             label="val macro-F1")
    ax2.set_ylabel("validation macro-F1")
    fig.legend(loc="upper center", ncol=2, fontsize=8)
    fig.tight_layout(); fig.savefig(run_dir / "learning_curve.png", dpi=150); plt.close(fig)

    cm = np.array(res["confusion_matrix"], dtype=float)
    cm_norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(D.AAMI_CLASSES)), D.AAMI_CLASSES)
    ax.set_yticks(range(len(D.AAMI_CLASSES)), D.AAMI_CLASSES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if cm_norm[i, j] > 0.5 else "black")
    ax.set_title(f"{res['split']} split, recall per row", fontsize=9)
    fig.colorbar(im, fraction=0.046); fig.tight_layout()
    fig.savefig(run_dir / "confusion_matrix.png", dpi=150); plt.close(fig)


def append_run_row(path: Path, res: dict) -> None:
    """One row per run, so every number can be traced to a commit, config and seed."""
    row = {k: res[k] for k in ["timestamp", "git_commit", "split", "class_weighting", "seed",
                               "synthetic", "accuracy", "macro_f1"]}
    row["macro_f1_ci_low"], row["macro_f1_ci_high"] = res["macro_f1_ci95"]
    for name, m in res["per_class"].items():
        row[f"recall_{name}"] = round(m["recall"], 4)
    new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train an ECG beat classifier on MIT-BIH.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--split", choices=["patient", "beat"], default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--weighting", choices=["none", "inverse", "sqrt_inverse"], default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--synthetic", action="store_true", help="use fake data (no download)")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    cfg = load_config(a.config)
    if a.weighting:
        cfg["train"]["class_weighting"] = a.weighting
    if a.epochs:
        cfg["train"]["epochs"] = a.epochs
    run(cfg, a.split or cfg["split"], cfg["seed"] if a.seed is None else a.seed, a.synthetic, Path(a.out))


if __name__ == "__main__":
    main()
