import copy, json, statistics, time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.flop_counter import FlopCounterMode
from torchinfo import summary
from torchvision import datasets, transforms
import thop

HERE = Path(__file__).parent
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
torch.manual_seed(0)
torch.set_num_threads(4)


def mlp():
    return nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(),
                         nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10))


def cnn():
    return nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Flatten(), nn.Linear(32 * 7 * 7, 10))


class Attention(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.proj = nn.Linear(49, d)
        self.pos = nn.Parameter(torch.zeros(1, 16, d))
        layer = nn.TransformerEncoderLayer(d, nhead=4, dim_feedforward=128, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.head = nn.Linear(d, 10)

    def forward(self, x):
        p = x.unfold(2, 7, 7).unfold(3, 7, 7)
        p = p.reshape(x.shape[0], 16, 49)
        h = self.enc(self.proj(p) + self.pos)
        return self.head(h.mean(1))


MODELS = {"MLP": mlp, "CNN": cnn, "Attention": Attention}


def lin(i, o, n=1): return 2 * n * i * o
def conv(k, ci, co, h, w): return 2 * k * k * ci * co * h * w


def attn_layer(seq=16, d=64, ff=128):
    return {"QKV proj": 2 * seq * d * 3 * d, "scores QK^T": 2 * seq**2 * d, "softmax·V": 2 * seq**2 * d,
            "out proj": lin(d, d, seq), "FFN": lin(d, ff, seq) + lin(ff, d, seq)}


HAND = {
    "MLP": {"Linear 784→256": lin(784, 256), "Linear 256→128": lin(256, 128), "Linear 128→10": lin(128, 10)},
    "CNN": {"Conv 1→16 @28²": conv(3, 1, 16, 28, 28), "Conv 16→32 @14²": conv(3, 16, 32, 14, 14),
            "Linear 1568→10": lin(1568, 10)},
    "Attention": {"patch proj 49→64 ×16": lin(49, 64, 16),
                  **{f"enc{i} {k}": v for i in (1, 2) for k, v in attn_layer().items()},
                  "Linear 64→10": lin(64, 10)},
}

tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
full = datasets.FashionMNIST(HERE / "data", train=True, download=True, transform=tf)
test = datasets.FashionMNIST(HERE / "data", train=False, download=True, transform=tf)
train, val = random_split(full, [55_000, 5_000], generator=torch.Generator().manual_seed(0))
loaders = {n: DataLoader(d, batch_size=128, shuffle=(n == "train")) for n, d in
           [("train", train), ("val", val), ("test", test)]}
print(f"train {len(train)}  val {len(val)}  test {len(test)}")


@torch.no_grad()
def accuracy(model, loader):
    model.eval()
    return sum((model(x).argmax(1) == y).sum().item() for x, y in loader) / len(loader.dataset)


def latency_ms(model, batch, runs):
    x = torch.randn(batch, 1, 28, 28)
    with torch.inference_mode():
        for _ in range(10):
            model(x)
        ts = []
        for _ in range(runs):
            t = time.perf_counter(); model(x); ts.append(time.perf_counter() - t)
    return statistics.median(ts) * 1e3


def measured_flops(model):
    model.train()
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        model(torch.randn(1, 1, 28, 28))
    return fc.get_total_flops()


results = {}
for name, make in MODELS.items():
    print(f"\n================ {name} ================")
    model = make()
    info = summary(model, input_size=(1, 1, 28, 28), verbose=0,
                   col_names=("output_size", "num_params", "mult_adds"))
    print(info)
    hand = sum(HAND[name].values())
    for k, v in HAND[name].items():
        print(f"  hand  {k:28s} {v:>12,}")
    m_eval = copy.deepcopy(model).eval()
    thop_macs, _ = thop.profile(copy.deepcopy(model).eval(), inputs=(torch.randn(1, 1, 28, 28),), verbose=False)
    torch.backends.mha.set_fastpath_enabled(False)
    thop_nofast, _ = thop.profile(copy.deepcopy(model), inputs=(torch.randn(1, 1, 28, 28),), verbose=False)
    torch.backends.mha.set_fastpath_enabled(True)
    torch_flops = measured_flops(copy.deepcopy(model))
    print(f"  hand FLOPs total     {hand:>12,}")
    print(f"  thop MACs  (eval)    {int(thop_macs):>12,}  -> x2 = {int(2 * thop_macs):,} FLOPs")
    print(f"  thop MACs  no fast   {int(thop_nofast):>12,}  -> x2 = {int(2 * thop_nofast):,} FLOPs")
    print(f"  FlopCounterMode      {torch_flops:>12,}")

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn, val_curve = nn.CrossEntropyLoss(), []
    t0 = time.time()
    for ep in range(5):
        model.train()
        for x, y in loaders["train"]:
            opt.zero_grad(); loss = loss_fn(model(x), y); loss.backward(); opt.step()
        val_curve.append(accuracy(model, loaders["val"]))
        print(f"  epoch {ep + 1}  loss {loss.item():.3f}  val acc {val_curve[-1]:.4f}")
    model.eval()
    results[name] = {
        "params": info.total_params, "hand_flops": hand, "thop_macs": int(thop_macs), "thop_macs_nofastpath": int(thop_nofast),
        "flopcounter_flops": torch_flops, "hand_breakdown": HAND[name],
        "lat_b1_ms": latency_ms(model, 1, 100), "lat_b128_ms": latency_ms(model, 128, 20),
        "val_curve": val_curve, "test_acc": accuracy(model, loaders["test"]),
        "train_s": time.time() - t0,
    }
    r = results[name]
    print(f"  test acc {r['test_acc']:.4f}  latency b1 {r['lat_b1_ms']:.3f} ms  b128 {r['lat_b128_ms']:.2f} ms")

(OUT / "results.json").write_text(json.dumps(results, indent=2))

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, key, xl in zip(axes, ["params", "lat_b1_ms", "lat_b128_ms"],
                       ["parameters", "latency batch 1 (ms, median of 100)", "latency batch 128 (ms, median of 20)"]):
    for n, r in results.items():
        ax.scatter(r[key], r["test_acc"] * 100, s=80)
        ax.annotate(n, (r[key], r["test_acc"] * 100), textcoords="offset points", xytext=(6, 6))
    ax.set_xlabel(xl); ax.set_ylabel("test accuracy (%)"); ax.grid(alpha=.3)
axes[0].set_title("Accuracy vs parameters")
axes[1].set_title("Accuracy vs latency (batch 1)")
axes[2].set_title("Accuracy vs latency (batch 128)")
fig.tight_layout(); fig.savefig(OUT / "tradeoff.png", dpi=130)

print("\n| model | params | hand FLOPs | thop MACs ×2 | FlopCounterMode | b1 ms | b128 ms | test acc |")
for n, r in results.items():
    print(f"| {n} | {r['params']:,} | {r['hand_flops']:,} | {2 * r['thop_macs']:,} | {r['flopcounter_flops']:,} "
          f"| {r['lat_b1_ms']:.3f} | {r['lat_b128_ms']:.2f} | {r['test_acc'] * 100:.2f}% |")
