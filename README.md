# ECG Arrhythmia Classification — an honest inter-patient benchmark

A PyTorch 1D residual CNN that classifies single heartbeats from the
[MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/) into the five
AAMI classes (N, S, V, F, Q). The project's focus is **evaluation done right**: it measures
how much a common data-split mistake inflates the results.

## Why this project

Many ECG papers, including my own 2020 master's thesis, split the data **by beat**. Beats
from the same patient then appear in both the training and the test set, and the model can
partly recognise the *patient* instead of the *arrhythmia*. This project compares that
split against the standard **inter-patient** split (de Chazal et al., 2004: train on 22
records, test on 22 different records) under identical settings, so the size of the leakage
can be measured.

## Results

> Paste the table from `results/summary.md` here after you run the experiments
> (see "Reproduce"). Mean ± standard deviation over 3 seeds.

**What the numbers show:** _(write 3 sentences after your run: how much higher the beat
split scored, which classes collapse on new patients, and whether class weighting helped
rare-class recall)._

Plots from the best patient-split run: `results/<run>/confusion_matrix.png` and
`results/<run>/learning_curve.png`.

## Method

| Part | Choice | Why |
| --- | --- | --- |
| Input | Lead MLII, 200 samples around each R-peak (0.25 s before, 0.31 s after), z-scored per beat | Fixed-length windows; per-beat scaling reduces amplitude differences between patients |
| Labels | AAMI EC57 mapping of MIT-BIH symbols to N, S, V, F, Q | Standard grouping used in the literature |
| Split | Inter-patient DS1 → DS2; 4 whole training records held out for validation | No patient in more than one set |
| Model | 3 residual Conv1d blocks (32/64/128 channels, BatchNorm), global average pooling, dropout 0.2 | Residual connections fix the underfitting of deep plain CNNs |
| Imbalance | Class-weighted cross-entropy (square-root inverse frequency) | N is about 90% of beats; plain inverse weights are too extreme |
| Training | AdamW, lr 1e-3, batch 256, gradient clipping 1.0, bfloat16 autocast on GPU, early stopping on validation macro-F1 (patience 5) | Stable and fast |
| Metrics | Macro-F1 (headline), per-class recall and precision, confusion matrix, 95% bootstrap CI | Accuracy hides failures on rare classes |

## Reproduce

```bash
git clone https://github.com/Bukola-Baiyewu1/ecg-arrhythmia-pytorch.git
cd ecg-arrhythmia-pytorch
pip install -e ".[dev]"
pytest -q                                                   # 12 tests, synthetic data, no download
python -m ecg.train --config configs/default.yaml           # one run; downloads MIT-BIH (~100 MB) once
python -m ecg.experiments --config configs/default.yaml --seeds 0 1 2   # full comparison table
```

Every run writes `metrics.json`, the exact `config.yaml`, the model weights and two plots to
its own folder in `results/`, and appends a row to `results/runs.csv` with the timestamp,
git commit, split, seed and metrics, so every number can be traced back to the code that
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

- Single-beat input: rhythm context (RR intervals) is not used, which limits the S class.
- MIT-BIH has 47 patients, so results on a new population may differ.
- Not a medical device; for research and learning only.

## Reference

de Chazal, P., O'Dwyer, M. and Reilly, R. B. (2004). Automatic classification of heartbeats
using ECG morphology and heartbeat interval features. *IEEE Transactions on Biomedical
Engineering*, 51(7), 1196–1206.
