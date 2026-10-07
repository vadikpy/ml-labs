"""Task 2: LogisticRegression and RandomForest baselines on Breast Cancer Wisconsin."""
import csv
from pathlib import Path

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

RESULTS = Path(__file__).resolve().parents[1] / "results"


def load_split():
    X, y = load_breast_cancer(return_X_y=True)
    return train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)


def make_models():
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    }


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_split()
    with open(RESULTS / "baseline_accuracy.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "test_accuracy"])
        for name, model in make_models().items():
            acc = model.fit(X_train, y_train).score(X_test, y_test)
            writer.writerow([name, f"{acc:.4f}"])
            print(f"{name}: {acc:.4f}")
