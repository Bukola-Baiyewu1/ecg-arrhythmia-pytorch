# ECG Arrhythmia Classification — an honest inter-patient benchmark

A PyTorch 1D residual CNN that classifies heartbeats from the
[MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/) into the five
AAMI classes (N, S, V, F, Q). The project's focus is **evaluation done right**: it measures
how much a common data-split mistake inflates the results (v1), then fixes the weakness the
honest evaluation exposes by adding heart-rhythm features (v2).

## Why this project

Many ECG papers, including my own 2020 master's thesis, split the data **by beat**. Beats
from the same patient then appear in both the training and the test set, and the model can
partly recognise the *patient* instead of the *arrhythmia*. This project compares that
split against the standard **inter-patient** split (de Chazal et al., 2004: train on 22
records, test on 22 different records) under identical settings, so the size of the leakage
can be measured.

## Results

Mean ± standard deviation over 3 seeds, NVIDIA T4. Test set for the patient split: the 22
DS2 records, none of which appear in training.

### v1 — beat shape only: how much does a leaky split inflate the score?

| Split | Class weighting | Accuracy | Macro-F1 | Recall N | Recall S | Recall V | Recall F | Recall Q |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| beat (leaky) | none | 0.988 ± 0.001 | 0.735 ± 0.005 | 1.00 | 0.82 | 0.97 | 0.77 | 0.00 |
| beat (leaky) | sqrt_inverse | 0.986 ± 0.003 | 0.726 ± 0.018 | 0.99 | 0.84 | 0.96 | 0.87 | 0.00 |
| patient | none | 0.924 ± 0.004 | 0.353 ± 0.006 | 0.98 | 0.01 | 0.82 | 0.00 | 0.00 |
| patient | sqrt_inverse | 0.837 ± 0.053 | 0.348 ± 0.009 | 0.89 | 0.07 | 0.72 | 0.01 | 0.00 |

**What v1 shows:** splitting by beat lets the same patients appear in training and test, and
inflates macro-F1 from 0.35 to 0.74 — more than double. On genuinely new patients the
shape-only model almost never detects supraventricular (S) beats (recall 0.01) or fusion
beats (F), and class weighting barely helps. S beats look like normal beats; what identifies
them is that they arrive early, which a single-beat window cannot see. Q stays at 0 in every
setting because almost no Q beats remain once paced records are excluded, so v2 also reports
macro-F1 over N, S, V and F, as is common in the literature.

### v2 — beat shape + rhythm (RR-interval) features

| Split | Input | Class weighting | Accuracy | Macro-F1 (5 classes) | Macro-F1 (N,S,V,F) | Recall N | Recall S | Recall V | Recall F | Recall Q |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| patient | beat + RR | none | 0.924 ± 0.013 | 0.363 ± 0.014 | 0.453 ± 0.018 | 0.97 | 0.04 | 0.88 | 0.00 | 0.00 |
| patient | beat + RR | sqrt_inverse | 0.834 ± 0.067 | 0.371 ± 0.023 | 0.464 ± 0.028 | 0.86 | 0.32 | 0.91 | 0.03 | 0.00 |

**What v2 shows:** _(3 sentences after your run: S recall and macro-F1 (N,S,V,F) with vs
without RR features on the patient split, and what still fails.)_

Plots from the best patient-split run: ![Confusion matrix](results/confusion_matrix.png)
![Learning curve](results/learning_curve.png).

## Method

| Part | Choice | Why |
| --- | --- | --- |
| Input | Lead MLII, 200 samples around each R-peak (0.25 s before, 0.31 s after), z-scored per beat | Fixed-length windows; per-beat scaling reduces amplitude differences between patients |
| Rhythm (v2) | 4 RR features per beat: pre-RR and post-RR relative to the patient's local average of the last 10 intervals, their ratio, and the local average in seconds; standardised with training-set statistics only | Captures prematurity relative to each patient's own heart rate, the main clue for S beats |
| Labels | AAMI EC57 mapping of MIT-BIH symbols to N, S, V, F, Q | Standard grouping used in the literature |
| Split | Inter-patient DS1 → DS2; 4 whole training records held out for validation | No patient in more than one set |
| Model | 3 residual Conv1d blocks (32/64/128 channels, BatchNorm), global average pooling, dropout 0.2; in v2 the RR features pass through a small MLP and are joined with the beat features | Residual connections fix the underfitting of deep plain CNNs |
| Imbalance | Class-weighted cross-entropy (square-root inverse frequency) | N is about 90% of beats; plain inverse weights are too extreme |
| Training | AdamW, lr 1e-3, batch 256, gradient clipping 1.0, bfloat16 autocast on GPU, early stopping on validation macro-F1 (patience 5) | Stable and fast |
| Metrics | Macro-F1 over 5 classes and over N, S, V, F; per-class recall and precision; confusion matrix; 95% bootstrap CI | Accuracy hides failures on rare classes |

## Reproduce

```bash
git clone https://github.com/Bukola-Baiyewu1/ecg-arrhythmia-pytorch.git
cd ecg-arrhythmia-pytorch
pip install -e ".[dev]"
pytest -q                                                   # 15 tests, synthetic data, no download
python -m ecg.train --config configs/default.yaml           # one v2 run; downloads MIT-BIH (~100 MB) once
python -m ecg.experiments --rr off --seeds 0 1 2            # v1: beat shape only (both splits)
python -m ecg.experiments --rr on --seeds 0 1 2             # v2: beat shape + RR features
```

Every run writes `metrics.json`, the exact `config.yaml`, the model weights and two plots to
its own folder in `results/`, and appends a row to `results/runs_v2.csv` with the timestamp,
git commit, split, features, seed and metrics, so every number can be traced back to the code that
produced it.

## Project layout

```
configs/default.yaml     every setting for a run
src/ecg/data.py          download, beat extraction, AAMI labels, patient and beat splits
src/ecg/model.py         residual 1D CNN
src/ecg/metrics.py       per-class metrics, macro-F1, bootstrap CI
src/ecg/train.py         command-line training and evaluation
src/ecg/experiments.py   split × weighting × seed comparison
tests/                   data, model and end-to-end tests (run in CI)
```

## Limitations

- v2 uses only four RR features; longer rhythm context (e.g. a sequence model over several beats) could help further.
- F (fusion) beats are very rare in the test records, so their recall is unstable.
- MIT-BIH has 47 patients, so results on a new population may differ.
- Not a medical device; for research and learning only.

## Reference

de Chazal, P., O'Dwyer, M. and Reilly, R. B. (2004). Automatic classification of heartbeats
using ECG morphology and heartbeat interval features. *IEEE Transactions on Biomedical
Engineering*, 51(7), 1196–1206.
