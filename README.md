# Spontaneous symmetry breaking and Goldstone modes for deep information propagation
https://arxiv.org/abs/2605.14685

We demonstrate that equivariance in internal states allows for an advantage in training very deep networks through the existence of a mode analagous to a Goldstone mode in statistical physics. This code allows reproduction of the main results for MLPs. 

## What's here

| file | purpose |
|------|---------|
| `equivariant_layers.py` | Equivariant layers: `U1Linear`/`RadialTanh`/`U1Block` (U(1)=SO(2)) and `OkLinear`/`OkRadialTanh`/`OkBlock` (O(k)), plus equivariance-check utilities. |
| `std_scan.py`  | Sweeps the weight-init std, trains a deep classifier on MNIST/FashionMNIST, computes the SSB order parameter, and produces the accuracy-vs-`sigma_w` figure. |

## Setup

```bash
pip install -r requirements.txt
```

MNIST / FashionMNIST download automatically (via `torchvision`) into `./data/` on
first run.

## Reproducing main result 
Here we show how to reproduce one of the main results (i.e. Figure 3) by scanning through the variance of the initial weight distribution, demonstrating that training is possible when we are in the spontaneous symmetry broken phase. 

`sigma_w` is the variance rescaled so the SSB transition is at `sigma_w = 1`:

- U(1):  `sigma_w = std * sqrt(N)`   (critical std `= 1/sqrt(N)`)
- O(k):  `sigma_w = std * sqrt(N/k)` (critical std `= sqrt(k/N)`)

```bash
# U(1), FashionMNIST  (N=64, L=100, 5 epochs, 5 seeds)
python std_scan.py --group u1 --dataset fashion_mnist --tag fmnist_u1

# U(1), MNIST
python std_scan.py --group u1 --dataset mnist --tag mnist_u1

# O(4), MNIST
python std_scan.py --group ok --k 4 --dataset mnist --tag mnist_ok4
```

Each run writes to `results/`:

- `scan_<tag>.csv` — per-(std, arch, seed, epoch) test accuracy
- `order_param_<tag>.csv` — order parameter vs `sigma_w`
- `std_scan_<tag>_epoch{1..E}.{png,pdf}` — accuracy (left axis) + order parameter
  (right axis) vs `sigma_w`, one figure per training epoch


Defaults match the paper: `N = 64` hidden features, `L = 100` layers, batch 256,
Adam at `lr = 1e-3`, 5 epochs, 5 seeds. A GPU is used automatically if available.

## Checking equivariance

```python
import torch
from equivariant_layers import U1Block, OkBlock, equivariance_check, ok_equivariance_check

print(equivariance_check(U1Block(features=16, num_hidden_layers=10), features=16))
print(ok_equivariance_check(OkBlock(k=4, features=16, num_hidden_layers=10), k=4, features=16))
# both ~1e-7 in float32
```

## Model

`Classifier = Linear(784 -> N) -> Block(L equivariant layers) -> Linear(N -> 10)`.
The input/output projections are ordinary `nn.Linear` (with bias) and break the
symmetry at the boundaries; the deep interior block is equivariant with no bias.
The non-equivariant baseline replaces the equivariant linear layers with plain
`nn.Linear` at the same init std, keeping the same radial activation.

## Citation
```
@article{iqbal2026spontaneous,
  title={Spontaneous symmetry breaking and Goldstone modes for deep information propagation},
  author={Iqbal, Nabil and Keller, T Anderson and Song, Yue and Miyato, Takeru and Welling, Max},
  journal={arXiv preprint arXiv:2605.14685},
  year={2026}
}
```