# Step-by-step guide: finish and publish this project

You'll end up with a public GitHub repo that has passing tests, real results on MIT-BIH, plots,
and a README that explains them. Plan for about 3–4 hours spread over 1–2 days.

The code is already written and tested. Your job is to run it, understand it, and publish
your own real results. **Don't put this project on your CV until Step 6 is done.**

---

## Step 1 — Create the empty repo on GitHub (5 min)

1. Go to github.com and sign in.
2. Click **+** (top right) → **New repository**.
3. Repository name: `ecg-arrhythmia-pytorch`. Description: `ECG heartbeat classification on MIT-BIH with an honest inter-patient split (PyTorch)`.
4. Choose **Public**. Leave "Add a README", ".gitignore" and "license" all **unticked** (the project already has them).
5. Click **Create repository**.

**Success looks like:** an empty repo page that says "Quick setup".

**Troubleshooting:** "name already exists" → you made it before; open that repo instead, or pick `ecg-arrhythmia-classifier`.

---

## Step 2 — Upload the code (10 min)

1. Unzip `ecg-arrhythmia-pytorch.zip` on your computer (double-click it).
2. On the empty repo page, click the link **uploading an existing file**.
3. Open the unzipped `ecg-arrhythmia-pytorch` folder. Select **everything inside it** (not the folder itself) and drag it into the browser.
   - On a Mac, the `.github` and `.gitignore` items are hidden. In Finder press **Cmd + Shift + .** (full stop) to show them, then select all again.
4. Commit message: `Initial project: data, model, training, tests, CI`. Click **Commit changes**.
5. Open the **Actions** tab. A run called "tests" starts.

**Success looks like:** the repo shows `src`, `tests`, `configs`, `.github`, `README.md`, and the Actions run turns **green** after 2–4 minutes.

**Troubleshooting**
- No `.github` folder in the repo → hidden files weren't selected; repeat step 3 with them visible.
- Actions run is **red** → click it, open the failed step, copy the last 20 lines and send them to me.
- Files landed inside an extra `ecg-arrhythmia-pytorch/` folder → you dragged the folder, not its contents. Delete and re-upload the contents.

---

## Step 3 — Run the real experiments in Google Colab (45–90 min, mostly waiting)

1. Go to colab.research.google.com → **New notebook**.
2. **Runtime → Change runtime type → T4 GPU → Save**.
3. In the first cell, paste and run (Shift + Enter). Replace nothing; it's your repo:

```python
!git clone https://github.com/Bukola-Baiyewu1/ecg-arrhythmia-pytorch.git
%cd ecg-arrhythmia-pytorch
!pip install -q -e ".[dev]"
!pytest -q
```

**What it does:** downloads your code, installs it, and runs the 12 tests.
**Success looks like:** the last line says `12 passed`.

4. In a new cell, run one real training run first:

```python
!python -m ecg.train --config configs/default.yaml
```

**What it does:** downloads MIT-BIH from PhysioNet (about 100 MB, a few minutes, only once), cuts about 100,000 heartbeats, trains on 18 patients, validates on 4 other patients, and tests on 22 completely different patients.
**Success looks like:** epoch lines with a falling train loss, then `TEST accuracy … | macro-F1 …` and a recall line per class. The first line also says `records in both train and test: 0`. That 0 is the whole point of the project.

5. Now the full comparison (12 runs: 2 splits × 2 weightings × 3 seeds):

```python
!python -m ecg.experiments --config configs/default.yaml --seeds 0 1 2
```

**Success looks like:** a table at the end, also saved to `results/summary.md`.

**What to expect, so you're not surprised:** the **beat** split will score clearly higher than the **patient** split. On new patients, the S class (supraventricular) and F class (fusion) usually score low, because single beats look very similar to normal beats without rhythm context. Low numbers there are normal and expected, not a bug. Your README explains them honestly.

**Troubleshooting**
- `No module named ecg` → the `%cd` line didn't run; run cell 1 again.
- Download error or timeout → run the cell again; it resumes. If PhysioNet is down, try later.
- `CUDA out of memory` → in `configs/default.yaml` change `batch_size: 256` to `128` (Colab: open the file from the folder icon on the left, edit, Ctrl+S).
- Colab disconnects during the long run → reconnect, rerun cell 1, then run the experiments with fewer seeds: `--seeds 0 1`.

---

## Step 4 — Save your results (10 min)

Run this in a new cell:

```python
!zip -r results.zip results -x "*.pt"
from google.colab import files
files.download("results.zip")
```

**Success looks like:** `results.zip` downloads to your computer, with `summary.md`, `runs.csv` and one folder per run (each with `metrics.json`, `config.yaml`, `confusion_matrix.png`, `learning_curve.png`).

---

## Step 5 — Put the results in the README (30 min)

1. Unzip `results.zip`. Open `summary.md` and copy the table.
2. In your GitHub repo, open `README.md` → click the **pencil** icon (Edit).
3. Under **Results**, replace the grey placeholder lines with the table.
4. Replace the "What the numbers show" placeholder with **three sentences, using your own numbers**:
   - how much higher the beat split's macro-F1 was than the patient split's;
   - which classes dropped most on new patients;
   - whether class weighting raised rare-class recall, and what it cost.
5. Upload the two plots from your best **patient / sqrt_inverse** run: in the repo, open the `results` folder → **Add file → Upload files** → drop `confusion_matrix.png` and `learning_curve.png`. Then add these two lines under the Results table in the README:

```markdown
![Confusion matrix](results/confusion_matrix.png)
![Learning curve](results/learning_curve.png)
```

6. Commit with the message `Add MIT-BIH results`.

**Success looks like:** the repo's front page shows your table, two plots and three honest sentences.

---

## Step 6 — Make sure you can explain it (45 min)

Answer these out loud, 5–6 sentences each, using **your** numbers. If you can't answer one, open the file named in brackets and read it, or ask me.

1. Why did the beat split score higher, and why is the patient split the honest one? (`src/ecg/data.py`, top docstring)
2. Why is macro-F1 your headline metric, not accuracy? (`src/ecg/metrics.py`)
3. Why square-root inverse class weights instead of plain inverse? (`class_weights` in `src/ecg/train.py`)
4. Why residual connections and BatchNorm, compared with your 2020 thesis model? (`src/ecg/model.py`)
5. How would someone reproduce your exact number? (`runs.csv`: commit, config, seed)
6. What would you do next to improve the S class? (Answer idea: add RR-interval features, i.e. rhythm context, alongside the beat shape.)

---

## Step 7 — Only now, add it to your CV and LinkedIn

When steps 1–6 are done, send me your results table and your latest CV, and I'll write the
CV entry with your real numbers. It will follow this shape:

> **ECG arrhythmia classification — inter-patient benchmark (2026)** · github.com/Bukola-Baiyewu1/ecg-arrhythmia-pytorch
> Rebuilt my thesis classifier in PyTorch with a residual 1D CNN and the standard inter-patient split on MIT-BIH, and measured that a beat-level split inflates macro-F1 from [X] to [Y]. Class-weighted training raised rare-class recall [from A to B]. Reproducible runs (config, seed, commit logged), 12 tests and CI.
