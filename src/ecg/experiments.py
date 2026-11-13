"""Run the full comparison and write a results table for the README.

Runs every combination of split (beat vs patient) and class weighting
(none vs sqrt_inverse) over several seeds, then reports mean and standard
deviation, so no single lucky run drives a conclusion.

python -m ecg.experiments --config configs/default.yaml --seeds 0 1 2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .data import AAMI_CLASSES
from .train import load_config, run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    rows = []
    for split in ["beat", "patient"]:
        for weighting in ["none", "sqrt_inverse"]:
            accs, f1s, recalls = [], [], []
            for seed in a.seeds:
                cfg = load_config(a.config)
                cfg["train"]["class_weighting"] = weighting
                if a.epochs:
                    cfg["train"]["epochs"] = a.epochs
                res = run(cfg, split, seed, a.synthetic, Path(a.out))
                accs.append(res["accuracy"]); f1s.append(res["macro_f1"])
                recalls.append([res["per_class"][c]["recall"] for c in AAMI_CLASSES])
            r = np.array(recalls)
            rows.append((split, weighting, np.mean(accs), np.std(accs), np.mean(f1s), np.std(f1s), r.mean(0)))

    lines = [
        f"Mean ± standard deviation over seeds {a.seeds}. Test set: "
        + ("synthetic data." if a.synthetic else "MIT-BIH (DS2 records for the patient split)."),
        "",
        "| Split | Class weighting | Accuracy | Macro-F1 | " + " | ".join(f"Recall {c}" for c in AAMI_CLASSES) + " |",
        "| --- | --- | --- | --- | " + " | ".join("---" for _ in AAMI_CLASSES) + " |",
    ]
    for split, w, am, asd, fm, fsd, rec in rows:
        lines.append(f"| {split} | {w} | {am:.3f} ± {asd:.3f} | {fm:.3f} ± {fsd:.3f} | "
                     + " | ".join(f"{x:.2f}" for x in rec) + " |")
    table = "\n".join(lines)
    out = Path(a.out) / "summary.md"
    out.write_text(table + "\n")
    print("\n" + table + f"\n\nSaved to {out}")


if __name__ == "__main__":
    main()
