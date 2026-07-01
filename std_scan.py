"""
Weight-std scan: how deep-network trainability depends on the initialisation
scale, and its relation to the spontaneous-symmetry-breaking (SSB) transition.

For a fixed depth L we sweep the weight-init std and train a deep classifier on
MNIST / FashionMNIST.  We plot test accuracy against the rescaled coupling

    sigma_w = std * sqrt(N)      (U(1),  critical std = 1/sqrt(N))
    sigma_w = std * sqrt(N/k)    (O(k),  critical std = sqrt(k/N))

so the SSB transition always sits at sigma_w = 1.  On the same axis we overlay
the order parameter (mean per-block activation radius in the last layers of a
randomly initialised network), which turns on at sigma_w = 1.

The equivariant model only becomes trainable once the symmetry is spontaneously
broken (sigma_w > 1); the non-equivariant baseline with the same activation
stays near chance at every std.

Examples
--------
    # U(1) on FashionMNIST (default)
    python std_scan.py --group u1 --dataset fashion_mnist --tag fmnist_u1

    # O(4) on MNIST
    python std_scan.py --group ok --k 4 --dataset mnist --tag mnist_ok4

    # quick smoke test
    python std_scan.py --num-layers 20 --epochs 1 --seeds 1 --n-stds 4 \
                       --n-order-repeats 20
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from equivariant_layers import U1Block, RadialTanh, OkBlock, OkRadialTanh


# ── Model ───────────────────────────────────────────────────────────

class Classifier(nn.Module):
    """Linear(in -> N) -> equivariant Block(L layers) -> Linear(N -> out).

    The in/out projections are standard nn.Linear (with bias) and break the
    symmetry at the boundaries; the deep interior block is equivariant.
    """

    def __init__(self, in_features, features, out_features, num_hidden_layers,
                 std, group="u1", k=2, equivariant=True):
        super().__init__()
        self.linear_in = nn.Linear(in_features, features, bias=True)
        if group == "u1":
            self.block = U1Block(
                num_hidden_layers=num_hidden_layers, features=features,
                bias=False, std=std, U1_linear_layers=equivariant,
                activation=RadialTanh,
            )
        elif group == "ok":
            self.block = OkBlock(
                k=k, num_hidden_layers=num_hidden_layers, features=features,
                bias=False, std=std, Ok_linear_layers=equivariant,
                activation=lambda: OkRadialTanh(k),
            )
        else:
            raise ValueError(f"unknown group {group!r} (expected 'u1' or 'ok')")
        self.linear_out = nn.Linear(features, out_features, bias=True)

    def forward(self, x):
        h = x.reshape(x.shape[0], -1)
        return self.linear_out(self.block(self.linear_in(h)))


# ── Data (torchvision, self-downloading) ───────────────────────────

def load_data(device, dataset_name="fashion_mnist"):
    from torchvision import datasets
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    is_fashion = dataset_name.lower() in ("fashion_mnist", "fashion", "fmnist")
    DS = datasets.FashionMNIST if is_fashion else datasets.MNIST
    print(f"Loading {'FashionMNIST' if is_fashion else 'MNIST'} into {root} ...")
    train = DS(root, train=True, download=True)
    test = DS(root, train=False, download=True)

    def to_tensors(ds):
        X = ds.data.float() / 255.0          # (N, 28, 28)
        y = ds.targets.long()
        return X.to(device), y.to(device)

    X_train, y_train = to_tensors(train)
    X_test, y_test = to_tensors(test)
    return X_train, y_train, X_test, y_test


def compute_accuracy(model, X, y, batch_size):
    model.eval()
    correct = 0
    with torch.no_grad():
        for start in range(0, X.shape[0], batch_size):
            logits = model(X[start:start + batch_size])
            correct += (logits.argmax(1) == y[start:start + batch_size]).sum().item()
    return correct / X.shape[0]


# ── Order parameter ────────────────────────────────────────────────

def compute_order_parameter(features, num_hidden_layers, stds, group="u1", k=2,
                            n_repeats=200, batch_size=100, last_layers=30,
                            device="cpu"):
    """Mean per-block activation radius in the last layers of random networks,
    normalised so the fully-saturated state (each block at unit norm) gives 1.

    Averaged over ``n_repeats`` fresh random equivariant blocks per std.
    """
    kk = 2 if group == "u1" else k
    order_means = []
    for std in stds:
        rmeans = []
        for _ in range(n_repeats):
            if group == "u1":
                block = U1Block(num_hidden_layers=num_hidden_layers, features=features,
                                bias=False, std=std, U1_linear_layers=True).to(device)
            else:
                block = OkBlock(k=k, num_hidden_layers=num_hidden_layers, features=features,
                                bias=False, std=std, Ok_linear_layers=True,
                                activation=lambda: OkRadialTanh(k)).to(device)
            x = torch.randn(batch_size, features, device=device)
            with torch.no_grad():
                # return_all stacks [lin_0, act_0, lin_1, act_1, ...]; [1::2] = post-activation
                acts = block(x, return_all=True)[1::2].cpu().numpy()
            # per-block radius: reshape features -> (n_blocks, k), norm over k
            a = acts[-last_layers:]                                    # (L', batch, features)
            a = a.reshape(*a.shape[:2], features // kk, kk)
            rmean = np.mean(np.linalg.norm(a, axis=-1)) / np.sqrt(features / kk)
            rmeans.append(rmean)
        order_means.append(np.mean(rmeans))
    return np.array(order_means)


# ── Training ───────────────────────────────────────────────────────

def train_and_eval(model, X_train, y_train, X_test, y_test,
                   epochs, batch_size, lr, device):
    """Train and return a list of test accuracies, one per epoch."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    N = X_train.shape[0]
    epoch_accs = []
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(N, device=device)
        for start in range(0, N, batch_size):
            idx = perm[start:start + batch_size]
            optimizer.zero_grad()
            loss = criterion(model(X_train[idx]), y_train[idx])
            loss.backward()
            optimizer.step()
        epoch_accs.append(compute_accuracy(model, X_test, y_test, batch_size))
    return epoch_accs


# ── Plotting ───────────────────────────────────────────────────────

def make_plot(df, order_sigma_ws, order_means, epoch, group, k, out_path):
    import pandas as pd  # noqa: F401 (df is a DataFrame)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    df_ep = df[df["epoch"] == epoch]

    if group == "u1":
        series = [("equivariant", "C0", "o", "U(1) equivariant"),
                  ("generic", "C1", "s", "Generic (non-equivariant)")]
        xlabel = r"$\sigma_w = \mathrm{std}\,\sqrt{N}$"
    else:
        series = [("equivariant", "C2", "^", f"O({k}) equivariant"),
                  ("generic", "C1", "s", f"Generic + O({k}) activation")]
        xlabel = r"$\sigma_w = \mathrm{std}\,\sqrt{N/k}$"

    for arch, color, marker, label in series:
        sub = df_ep[df_ep["arch"] == arch]
        grouped = sub.groupby("sigma_w").agg(
            mean=("test_accuracy", "mean"), std=("test_accuracy", "std"),
        ).reset_index()
        ax1.plot(grouped["sigma_w"], grouped["mean"], marker=marker,
                 linewidth=2, color=color, label=label)
        ax1.fill_between(grouped["sigma_w"], grouped["mean"] - grouped["std"],
                         grouped["mean"] + grouped["std"], color=color, alpha=0.2)

    ax1.set_xlabel(xlabel, fontsize=14)
    ax1.set_ylabel("Test accuracy", fontsize=14)
    ax1.axvline(x=1.0, linestyle="--", color="gray", alpha=0.6)
    ax1.tick_params(axis="both", labelsize=12)

    ax2 = ax1.twinx()
    ax2.plot(order_sigma_ws, order_means, color="red", linewidth=2,
             label="Order parameter")
    ax2.set_ylabel("Order parameter", fontsize=14, color="red")
    ax2.tick_params(axis="y", labelcolor="red", labelsize=12)

    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, fontsize=11, loc="center left")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.savefig(out_path.replace(".png", ".pdf"), dpi=200)
    plt.close()


# ── Main ───────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--group", choices=["u1", "ok"], default="u1",
                   help="symmetry group: U(1) [default] or O(k)")
    p.add_argument("--k", type=int, default=4, help="k for O(k) (ignored for u1)")
    p.add_argument("--dataset", default="fashion_mnist",
                   help="'mnist' or 'fashion_mnist'")
    p.add_argument("--features", type=int, default=64, help="hidden width N")
    p.add_argument("--num-layers", type=int, default=100, help="depth L of the block")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n-stds", type=int, default=8, help="std values in the training sweep")
    p.add_argument("--std-min", type=float, default=None,
                   help="min std (default: auto per group)")
    p.add_argument("--std-max", type=float, default=None,
                   help="max std (default: auto per group)")
    p.add_argument("--n-order-stds", type=int, default=50,
                   help="std grid for the (cheap) order-parameter curve")
    p.add_argument("--n-order-repeats", type=int, default=200)
    p.add_argument("--tag", default="", help="suffix for output filenames")
    p.add_argument("--out-dir", default="results")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N, k = args.features, args.k

    # sigma_w normalisation and critical std depend on the layer type.
    if args.group == "u1":
        sw = lambda std: std * np.sqrt(N)
        crit_std = 1.0 / np.sqrt(N)
        std_min = 0.05 if args.std_min is None else args.std_min
        std_max = 0.35 if args.std_max is None else args.std_max
    else:
        sw = lambda std: std * np.sqrt(N / k)
        crit_std = np.sqrt(k / N)
        std_min = 0.10 if args.std_min is None else args.std_min
        std_max = 0.70 if args.std_max is None else args.std_max

    print(f"Device: {device}")
    print(f"group={args.group} N={N}" + (f" k={k}" if args.group == "ok" else "")
          + f"  L={args.num_layers}  critical std={crit_std:.4f} (sigma_w=1)")

    X_train, y_train, X_test, y_test = load_data(device, args.dataset)

    stds = np.linspace(std_min, std_max, args.n_stds)

    # Training sweep over (std, arch, seed).
    archs = ["equivariant", "generic"]
    total = len(stds) * len(archs) * args.seeds
    idx, t0 = 0, time.time()
    rows = []
    for std in stds:
        for arch in archs:
            for seed in range(args.seeds):
                idx += 1
                torch.manual_seed(seed)
                np.random.seed(seed)
                model = Classifier(
                    in_features=28 * 28, features=N, out_features=10,
                    num_hidden_layers=args.num_layers, std=std,
                    group=args.group, k=k, equivariant=(arch == "equivariant"),
                ).to(device)
                accs = train_and_eval(model, X_train, y_train, X_test, y_test,
                                      args.epochs, args.batch_size, args.lr, device)
                for ep, acc in enumerate(accs):
                    rows.append(dict(std=std, sigma_w=sw(std), arch=arch,
                                     seed=seed, epoch=ep + 1, test_accuracy=acc))
                print(f"  [{idx}/{total}] std={std:.3f} (sigma_w={sw(std):.2f}) "
                      f"{arch:12s} seed={seed}: "
                      + " ".join(f"e{e+1}={a:.3f}" for e, a in enumerate(accs)))
    print(f"\nTraining done in {(time.time() - t0)/60:.1f} min")

    # Order parameter (cheap, no training).
    print("Computing order parameter ...")
    order_stds = np.linspace(std_min, std_max, args.n_order_stds)
    order_sigma_ws = np.array([sw(s) for s in order_stds])
    order_means = compute_order_parameter(
        N, args.num_layers, order_stds, group=args.group, k=k,
        n_repeats=args.n_order_repeats, device=device,
    )

    # Save + plot.
    import pandas as pd
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, f"scan{tag}.csv"), index=False)
    pd.DataFrame({"sigma_w": order_sigma_ws, "order_parameter": order_means}).to_csv(
        os.path.join(out_dir, f"order_param{tag}.csv"), index=False)

    for ep in range(1, args.epochs + 1):
        path = os.path.join(out_dir, f"std_scan{tag}_epoch{ep}.png")
        make_plot(df, order_sigma_ws, order_means, ep, args.group, k, path)
    print(f"Saved CSVs and per-epoch plots to {out_dir}/")


if __name__ == "__main__":
    main()
