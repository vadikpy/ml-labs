"""Lab 2: a two-hidden-layer neural network from scratch (NumPy only)."""
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

SEED = 42
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


# ---------- 1. Dataset ----------
def load_data():
    X, y = load_digits(return_X_y=True)
    X = X / 16.0
    Y = np.eye(10)[y]  # one-hot
    return train_test_split(X, Y, test_size=0.3, stratify=y, random_state=SEED)


# ---------- 2. Network design ----------
def init_params(W, rng):
    """64 -> W -> ReLU -> W -> ReLU -> 10 -> softmax, weights ~ N(0, 0.01)."""
    return {
        "W1": rng.normal(0, 0.01, (64, W)), "b1": np.zeros(W),
        "W2": rng.normal(0, 0.01, (W, W)), "b2": np.zeros(W),
        "W3": rng.normal(0, 0.01, (W, 10)), "b3": np.zeros(10),
    }


# ---------- 3. Forward pass ----------
def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))  # shift for numerical stability
    return e / e.sum(axis=1, keepdims=True)


def forward(p, X):
    z1 = X @ p["W1"] + p["b1"]
    a1 = np.maximum(z1, 0)
    z2 = a1 @ p["W2"] + p["b2"]
    a2 = np.maximum(z2, 0)
    z3 = a2 @ p["W3"] + p["b3"]
    probs = softmax(z3)
    return probs, (X, z1, a1, z2, a2)


# ---------- 4. Cross-entropy loss ----------
def cross_entropy(probs, Y):
    return -np.mean(np.sum(Y * np.log(probs + 1e-12), axis=1))


# ---------- 5. Backward pass (chain rule) ----------
def backward(p, probs, cache, Y):
    X, z1, a1, z2, a2 = cache
    n = X.shape[0]
    dz3 = (probs - Y) / n                 # dL/dz3 for softmax + cross-entropy
    g = {"W3": a2.T @ dz3, "b3": dz3.sum(0)}
    dz2 = (dz3 @ p["W3"].T) * (z2 > 0)    # back through W3, then ReLU'
    g["W2"], g["b2"] = a1.T @ dz2, dz2.sum(0)
    dz1 = (dz2 @ p["W2"].T) * (z1 > 0)    # back through W2, then ReLU'
    g["W1"], g["b1"] = X.T @ dz1, dz1.sum(0)
    return g


def loss_at(p, X, Y):
    return cross_entropy(forward(p, X)[0], Y)


def gradient_check(p, X, Y, rng, n_checks=10, eps=1e-5, min_grad=1e-5):
    probs, cache = forward(p, X)
    g = backward(p, probs, cache, Y)
    # float64 round-off in the loss (~1e-16) / (2 * eps) gives ~1e-11 absolute error
    # in the numerical slope, so gradients much below 1e-5 (dead / near-dead ReLU
    # paths) cannot reach 1e-6 relative error. Sample only weights the check can resolve.
    candidates = {k: np.argwhere(np.abs(g[k]) > min_grad) for k in ("W1", "W2", "W3")}
    names = [k for k, c in candidates.items() if len(c)]
    errors = []
    for _ in range(n_checks):
        name = rng.choice(names)
        idx = tuple(int(i) for i in candidates[name][rng.integers(len(candidates[name]))])
        orig = p[name][idx]
        p[name][idx] = orig + eps
        lp = loss_at(p, X, Y)
        p[name][idx] = orig - eps
        lm = loss_at(p, X, Y)
        p[name][idx] = orig
        num = (lp - lm) / (2 * eps)
        ana = g[name][idx]
        rel = abs(ana - num) / max(abs(ana) + abs(num), 1e-12)
        errors.append(rel)
        print(f"  {name}{idx}: analytic={ana:+.6e} numeric={num:+.6e} rel_err={rel:.2e}")
    passed = sum(e < 1e-6 for e in errors)
    print(f"  {passed}/{n_checks} below 1e-6 -> {'PASS' if passed >= 9 else 'FAIL'}")
    return passed >= 9


# ---------- 6. Training loop ----------
def accuracy(p, X, Y):
    return np.mean(forward(p, X)[0].argmax(1) == Y.argmax(1))


def train(Xtr, Ytr, Xte, Yte, W, lr, epochs=50, batch=32, log=True):
    rng = np.random.default_rng(SEED)
    p = init_params(W, rng)
    losses = []
    for ep in range(epochs):
        order = rng.permutation(len(Xtr))
        for i in range(0, len(Xtr), batch):
            b = order[i:i + batch]
            probs, cache = forward(p, Xtr[b])
            g = backward(p, probs, cache, Ytr[b])
            for k in p:
                p[k] -= lr * g[k]
        losses.append(loss_at(p, Xtr, Ytr))
        if log:
            print(f"  W={W:<3} lr={lr:<6} epoch {ep + 1:2d}/{epochs} train_loss={losses[-1]:.4f}")
    return p, losses, accuracy(p, Xte, Yte)


# ---------- 7. Timing ----------
def time_passes(p, X, Y, runs=100):
    fwd, bwd = [], []
    for _ in range(runs):
        t0 = time.perf_counter()
        probs, cache = forward(p, X)
        t1 = time.perf_counter()
        backward(p, probs, cache, Y)
        t2 = time.perf_counter()
        fwd.append(t1 - t0)
        bwd.append(t2 - t1)
    return np.median(fwd), np.median(bwd)


def main():
    os.makedirs(RESULTS, exist_ok=True)
    Xtr, Xte, Ytr, Yte = load_data()
    print(f"[1] train {Xtr.shape} {Ytr.shape}, test {Xte.shape} {Yte.shape}")

    rng = np.random.default_rng(SEED)
    for W in (16, 64):
        p = init_params(W, rng)
        print(f"[2] W={W}: " + ", ".join(f"{k}{v.shape}" for k, v in p.items()))
        probs, _ = forward(p, Xtr[:32])
        assert probs.shape == (32, 10)
        assert np.allclose(probs.sum(1), 1, atol=1e-6)
        print(f"[3] output shape {probs.shape}, max |row_sum - 1| = {np.abs(probs.sum(1) - 1).max():.1e}")
        print(f"[4] untrained loss = {loss_at(p, Xtr, Ytr):.4f} (ln 10 = {np.log(10):.4f})")
        # At N(0, 0.01) init most gradients sit right at the 1e-5 resolution limit;
        # check on a partly trained network so every layer has resolvable gradients.
        p, _, _ = train(Xtr, Ytr, Xte, Yte, W, lr=0.1, epochs=30, log=False)
        print(f"[5] gradient check, W={W} (after 30 epochs at lr=0.1, batch of 64):")
        assert gradient_check(p, Xtr[:64], Ytr[:64], rng)

    print("[6] training")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    summary = []
    for ax, W in zip(axes, (16, 64)):
        for lr in (0.001, 0.01, 0.1):
            p, losses, acc = train(Xtr, Ytr, Xte, Yte, W, lr)
            summary.append((W, lr, losses[-1], acc))
            ax.plot(range(1, 51), losses, label=f"lr={lr} (test acc {acc:.3f})")
        ax.set(title=f"Hidden width W={W}", xlabel="epoch", ylabel="train cross-entropy")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "loss_curves.png"), dpi=150)

    print("\n W    lr      final_loss  test_acc")
    for W, lr, loss, acc in summary:
        print(f" {W:<4} {lr:<7} {loss:.4f}      {acc:.4f}")

    # 16 and 64 are the lab widths; 256 and 1024 show the trend once matmuls dominate.
    print("\n[7] timing, median of 100 runs (MFLOP = matmul work only)")
    for n in (32, len(Xtr)):
        for W in (16, 64, 256, 1024):
            p = init_params(W, np.random.default_rng(SEED))
            f, b = time_passes(p, Xtr[:n], Ytr[:n])
            ff = 2 * n * (64 * W + W * W + W * 10)           # X@W1, a1@W2, a2@W3
            fb = 2 * n * (64 * W + 2 * W * W + 2 * W * 10)   # dW1, dW2, da1, dW3, da2
            print(f"  batch={n:<5} W={W:<5} forward {f * 1e6:8.1f} us ({ff / 1e6:6.1f} MFLOP)"
                  f"   backward {b * 1e6:8.1f} us ({fb / 1e6:6.1f} MFLOP)   bwd/fwd {b / f:.2f}x")

if __name__ == "__main__":
    main()
