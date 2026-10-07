# Lab 01 — Baselines and system cost

**Machine:** Apple M3 (arm64), 8 GB RAM, macOS 15.7
**Environment:** Python 3.11.15 in `.venv` (created with `uv`), packages from `../requirements.txt`. Full list: [`results/versions.txt`](results/versions.txt).

## 1. Environment

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python lab01/src/print_versions.py > lab01/results/versions.txt
```

I picked Python 3.11 because the pinned versions (torch 2.3.1, numpy 1.26.4) don't ship wheels for the system Python 3.14. I changed two lines in `requirements.txt`:

- `pyarrow==16.1.0` → `pyarrow==15.0.2`, because `mlflow==2.14.1` requires `pyarrow<16` and the original pins can't be installed together.
- Added `setuptools<81`, because mlflow 2.14 imports `pkg_resources`, which newer setuptools releases no longer include.

Layout:

```
lab01/
  src/print_versions.py   # Python + every installed package version
  src/baseline.py         # Task 2: data split, models, accuracy
  src/measure.py          # Tasks 3–4: system cost + deployment fit
  results/                # all generated outputs
  report.md
```

## 2. Baseline accuracy

Breast Cancer Wisconsin (569 samples, 30 features). 70/30 stratified split, `random_state=42`.

| Model | Test accuracy |
|---|---|
| LogisticRegression(max_iter=1000) | **0.9415** |
| RandomForestClassifier(n_estimators=100) | 0.9357 |

Source: [`results/baseline_accuracy.csv`](results/baseline_accuracy.csv). On this unscaled data lbfgs reaches `max_iter=1000` and raises a `ConvergenceWarning`. I kept the required hyperparameters anyway. Adding a `StandardScaler` would fix the warning and speed up training.

## 3. System cost

Method (`src/measure.py`):
- **Training time:** 1 warm-up fit, then the median of 5 fits of a fresh `clone()`.
- **Inference:** `predict` on one sample, 1 warm-up, then the median of 100 calls (`time.perf_counter`).
- **Size:** `joblib.dump` to `results/model.joblib`, file size read from disk. Each model is also copied to `model_<Name>.joblib`; `model.joblib` keeps the last model measured (RandomForest).
- **Peak memory:** `memory_profiler.memory_usage(..., max_usage=True)` around `fit` and `predict`. This is the RSS of the whole process. Baseline RSS before measuring comes from `psutil.Process().memory_info().rss`.

| Model | Train time (ms) | Inference, 1 sample (ms) | Size (bytes) | Size (KB) | RSS before (MB) | Peak RSS train (MB) | Peak RSS inference (MB) |
|---|---|---|---|---|---|---|---|
| LogisticRegression | 144.28 | 0.036 | 1,103 | 1.08 | 115.2 | 115.6 | 115.5 |
| RandomForest | 99.63 | 1.491 | 290,905 | 284.09 | 117.3 | 117.5 | 117.5 |

Source: [`results/system_cost.csv`](results/system_cost.csv).

What the numbers show:
- **Memory is dominated by the runtime, not the model.** Python, NumPy and scikit-learn take about 115 MB before any model exists. Training or prediction adds under 0.5 MB.
- **Inference:** LR is about 40× faster. It computes one dot product over 30 features, while RF walks 100 trees.
- **Size:** the RF file is about 260× larger. It stores 100 trees of nodes, while LR stores 30 coefficients and an intercept.
- **Training time is noisy.** Across three runs LR took 77–144 ms. LR is not cheaper to train here because lbfgs runs all 1000 iterations on unscaled features.
- **One outlier run:** the first run, made under memory pressure, showed inference RSS below the baseline (61 MB vs 114 MB) because macOS compresses process pages. The two later runs agreed within about 2 MB, and the table above uses the final run.

## 4. Deployment budget

For each budget, memory is checked against the inference-time peak RSS, since training happens off-device. A target fits only if all three limits pass.

| Target | Memory budget | Latency ≤ | Size ≤ | LogisticRegression | RandomForest |
|---|---|---|---|---|---|
| Cloud | ≥ 1 GB | 100 ms | 500 MB | ✅ 115.5 MB / 0.036 ms / 1.08 KB | ✅ 117.5 MB / 1.49 ms / 284 KB |
| Edge | 256–1024 MB | 50 ms | 50 MB | ✅ | ✅ |
| Mobile | 64–256 MB | 20 ms | 10 MB | ✅ (115.5 MB leaves ~140 MB headroom) | ✅ (117.5 MB leaves ~138 MB headroom) |
| TinyML | ≤ 256 KB | 10 ms | 100 KB | ❌ memory 115.5 MB ≫ 256 KB | ❌ memory 117.5 MB ≫ 256 KB **and** size 284 KB > 100 KB |

Source: [`results/deployment_fit.csv`](results/deployment_fit.csv).

**Conclusions:**
- **Both models fit Cloud, Edge and Mobile.**
  - Latency is far below every budget: 1.5 ms worst case against a 20 ms Mobile limit.
  - Both files are far below the 10 MB Mobile size limit.
  - Mobile memory is the tightest limit. About 115 MB uses roughly 45% of a 256 MB device and would not fit a 64 MB device. Almost all of that is the Python runtime.
- **Neither model fits TinyML as deployed.** A Python and scikit-learn process needs about 450× the 256 KB budget.
  - **LogisticRegression could still reach TinyML if exported.** Its parameters are 31 floats (124 bytes), its latency is 0.036 ms, and its file is already within the size limit. Exported to plain C (for example with `m2cgen`), it would fit a microcontroller easily.
  - **RandomForest can't be exported the same way.** Its 284 KB file is already over the 100 KB size limit. To get under it, it would need fewer or shallower trees, at some accuracy cost.
- **Recommendation:** LogisticRegression is both more accurate (0.9415 vs 0.9357) and cheaper on every inference metric. It is the better model for every target, and it is the only one with a realistic path to TinyML.

## Reproduce

```bash
cd lab01
../.venv/bin/python src/print_versions.py > results/versions.txt
../.venv/bin/python src/baseline.py
../.venv/bin/python src/measure.py
```
