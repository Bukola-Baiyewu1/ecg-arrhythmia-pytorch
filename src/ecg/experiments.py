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
    ap.add_argument("--splits", nargs="+", default=["beat", "patient"], choices=["beat", "patient"])
    ap.add_argument("--rr", nargs="+", default=["off", "on"], choices=["off", "on"],
                    help="run without and/or with rhythm (RR) features")
    ap.add_argument("--weightings", nargs="+", default=["none", "sqrt_inverse"])
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    rows = []
    for split in a.splits:
        for rr in a.rr:
            for weighting in a.weightings:
                accs, f1s, f4s, recalls = [], [], [], []
                for seed in a.seeds:
                    cfg = load_config(a.config)
                    cfg["train"]["class_weighting"] = weighting
                    cfg["model"]["use_rr"] = rr == "on"
                    if a.epochs:
                        cfg["train"]["epochs"] = a.epochs
                    res = run(cfg, split, seed, a.synthetic, Path(a.out))
                    accs.append(res["accuracy"]); f1s.append(res["macro_f1"]); f4s.append(res["macro_f1_nsvf"])
                    recalls.append([res["per_class"][c]["recall"] for c in AAMI_CLASSES])
                r = np.array(recalls)
                feat = "beat + RR" if rr == "on" else "beat only"
                rows.append((split, feat, weighting, np.mean(accs), np.std(accs), np.mean(f1s), np.std(f1s),
                             np.mean(f4s), np.std(f4s), r.mean(0)))

    lines = [
        f"Mean ± standard deviation over seeds {a.seeds}. Test set: "
        + ("synthetic data." if a.synthetic else "MIT-BIH (DS2 records for the patient split)."),
        "",
        "| Split | Input | Class weighting | Accuracy | Macro-F1 (5 classes) | Macro-F1 (N,S,V,F) | "
        + " | ".join(f"Recall {c}" for c in AAMI_CLASSES) + " |",
        "| --- | --- | --- | --- | --- | --- | " + " | ".join("---" for _ in AAMI_CLASSES) + " |",
    ]
    for split, feat, w, am, asd, fm, fsd, f4m, f4sd, rec in rows:
        lines.append(f"| {split} | {feat} | {w} | {am:.3f} ± {asd:.3f} | {fm:.3f} ± {fsd:.3f} | {f4m:.3f} ± {f4sd:.3f} | "
                     + " | ".join(f"{x:.2f}" for x in rec) + " |")
    table = "\n".join(lines)
    out = Path(a.out) / "summary_v2.md"
    out.write_text(table + "\n")
    print("\n" + table + f"\n\nSaved to {out}")


if __name__ == "__main__":
    main()
