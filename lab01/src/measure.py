"""Tasks 3-4: system cost of both baselines and fit against deployment budgets."""
import csv
import shutil
import statistics
import time

import joblib
import psutil
from memory_profiler import memory_usage
from sklearn.base import clone

from baseline import RESULTS, load_split, make_models

# Upper limits per target: (memory KB, latency ms, model size KB). Cloud memory is >= 1 GB, so no cap.
BUDGETS = {
    "Cloud": (None, 100, 500 * 1024),
    "Edge": (1024 * 1024, 50, 50 * 1024),
    "Mobile": (256 * 1024, 20, 10 * 1024),
    "TinyML": (256, 10, 100),
}


def timed_ms(fn):
    start = time.perf_counter()
    fn()
    return (time.perf_counter() - start) * 1000


def main():
    X_train, X_test, y_train, y_test = load_split()
    sample = X_test[:1]
    rows = []
    for name, proto in make_models().items():
        clone(proto).fit(X_train, y_train)  # warm-up
        train_ms = statistics.median(timed_ms(lambda: clone(proto).fit(X_train, y_train)) for _ in range(5))

        model = clone(proto).fit(X_train, y_train)
        model.predict(sample)  # warm-up
        infer_ms = statistics.median(timed_ms(lambda: model.predict(sample)) for _ in range(100))

        path = RESULTS / "model.joblib"
        joblib.dump(model, path)
        size_bytes = path.stat().st_size
        shutil.copy(path, RESULTS / f"model_{name}.joblib")  # model.joblib ends up holding the last model

        # Whole-process RSS in MiB: includes the Python + NumPy + scikit-learn runtime.
        rss_before = psutil.Process().memory_info().rss / 2**20
        train_peak = memory_usage((clone(proto).fit, (X_train, y_train)), max_usage=True, interval=0.001)
        infer_peak = memory_usage((model.predict, (sample,)), max_usage=True, interval=0.001)

        rows.append({
            "model": name,
            "train_time_ms": round(train_ms, 2),
            "inference_ms": round(infer_ms, 4),
            "size_bytes": size_bytes,
            "size_kb": round(size_bytes / 1024, 2),
            "rss_before_mb": round(rss_before, 1),
            "train_peak_mb": round(train_peak, 1),
            "inference_peak_mb": round(infer_peak, 1),
        })

    with open(RESULTS / "system_cost.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    # Deployment fit uses inference-time numbers: training happens off-device.
    with open(RESULTS / "deployment_fit.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "target", "memory_ok", "latency_ok", "size_ok", "fits"])
        for r in rows:
            for target, (mem_kb, lat_ms, size_kb) in BUDGETS.items():
                checks = (
                    mem_kb is None or r["inference_peak_mb"] * 1024 <= mem_kb,
                    r["inference_ms"] <= lat_ms,
                    r["size_kb"] <= size_kb,
                )
                writer.writerow([r["model"], target, *checks, all(checks)])

    for r in rows:
        print(r)
    print((RESULTS / "deployment_fit.csv").read_text())


if __name__ == "__main__":  # memory_profiler spawns a subprocess that re-imports this file
    main()
