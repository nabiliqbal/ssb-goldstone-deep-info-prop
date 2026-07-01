"""
Equivariant feedforward layers for the SSB / Goldstone deep-information-propagation
experiments.

Two families of layers are provided:

  * U(1) = SO(2) equivariance   (``U1Linear`` / ``RadialTanh`` / ``U1Block``)
        Features are grouped into pairs; the group rotates every pair by a common
        angle.  A single linear layer preserves this action iff its weight has the
        block structure  [[w1, w2], [-w2, w1]]  on each 2x2 micro-block.

  * O(k) equivariance           (``OkLinear`` / ``OkRadialTanh`` / ``OkBlock``)
        Features are grouped into blocks of size k; the group acts identically on
        every block.  By Schur's lemma the commutant of the fundamental rep is
        W = w (x) I_k.  (k=2 recovers the O(2) subspace of U(1).)

The activations only rescale the per-block magnitude (tanh(r)/r), leaving the
block direction untouched -- this is what preserves equivariance through depth.

Equivariance can be verified with ``equivariance_check`` / ``ok_equivariance_check``
(both return a relative error that should be ~1e-7 in float32).
"""

from typing import List, Type

import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════════════
# U(1) = SO(2)-equivariant modules
# ═══════════════════════════════════════════════════════════════════


class U1Linear(nn.Module):
    """Linear map preserving a U(1) action that rotates each pair of features.

    The weight is assembled from two blocks (w1, w2) so that every 2x2
    micro-block has the form [[w1, w2], [-w2, w1]], the most general 2x2 matrix
    commuting with SO(2).
    """

    def __init__(self, in_features: int = 4, out_features: int = 4,
                 bias: bool = False, dtype=None, device=None, std: float = 0.5):
        super().__init__()
        assert in_features % 2 == 0 and out_features % 2 == 0, \
            "Number of features must be even for a U(1) action"

        self.in_features = in_features
        self.out_features = out_features

        self.w1 = nn.Parameter(torch.empty(out_features // 2, in_features // 2,
                                            dtype=dtype, device=device))
        self.w2 = nn.Parameter(torch.empty(out_features // 2, in_features // 2,
                                            dtype=dtype, device=device))
        nn.init.normal_(self.w1, mean=0.0, std=std)
        nn.init.normal_(self.w2, mean=0.0, std=std)

        self.bias = nn.Parameter(torch.randn(out_features, dtype=dtype, device=device)) if bias else None

    def _weight(self):
        m, n = self.w1.shape[-2:]
        out = torch.empty(2 * m, 2 * n, device=self.w1.device, dtype=self.w1.dtype)
        out[..., 0::2, 0::2] = self.w1    # top-left  of each 2x2 micro-block
        out[..., 0::2, 1::2] = self.w2    # top-right
        out[..., 1::2, 0::2] = -self.w2   # bottom-left
        out[..., 1::2, 1::2] = self.w1    # bottom-right
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x @ self._weight().T
        if self.bias is not None:
            y = y + self.bias
        return y


class RadialTanh(nn.Module):
    """Applies tanh(r)/r to the magnitude r of each feature pair.

    The two components of every pair are rescaled by the same factor, so the
    pair's angle (the U(1) charge) is preserved exactly.
    """

    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, features, ...) -> (batch, features//2, 2, ...)
        y = x.reshape(x.shape[0], x.shape[1] // 2, 2, *x.shape[2:])
        r = torch.linalg.norm(y, dim=2, keepdim=True)
        scale = torch.tanh(r) / (r + self.eps)
        y = y * scale
        return y.reshape(x.shape)


class U1Block(nn.Module):
    """Stack of ``num_hidden_layers`` (U1Linear -> RadialTanh) layers.

    Set ``U1_linear_layers=False`` for the non-equivariant baseline: the same
    architecture with a plain ``nn.Linear`` (initialised at the same std) in
    place of ``U1Linear``, keeping the RadialTanh activation.
    """

    def __init__(
        self,
        features: int = 4,
        num_hidden_layers: int = 20,
        activation: Type[nn.Module] = RadialTanh,
        std: float = 0.5,
        bias: bool = False,            # bias breaks U(1) equivariance
        U1_linear_layers: bool = True,
    ):
        super().__init__()
        self.num_hidden_layers = num_hidden_layers

        layers = []
        for _ in range(num_hidden_layers):
            if U1_linear_layers:
                layers.append(U1Linear(in_features=features, out_features=features,
                                       bias=bias, std=std))
            else:
                lin = nn.Linear(in_features=features, out_features=features, bias=bias)
                nn.init.normal_(lin.weight, mean=0.0, std=std)
                layers.append(lin)
            layers.append(activation())
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, return_all: bool = False):
        outputs: List[torch.Tensor] = []
        h = x
        for layer in self.layers:
            h = layer(h)
            outputs.append(h)
        if return_all:
            return torch.stack(outputs)
        return h


def rot_mat(angle, features: int = 4):
    """Block-diagonal U(1) rotation matrix rotating every pair by ``angle``."""
    cos, sin = torch.cos(angle), torch.sin(angle)
    small_rot = torch.tensor([[cos, sin], [-sin, cos]])
    return torch.kron(torch.eye(features // 2), small_rot)


def equivariance_check(model, batch_size: int = 16, features: int = 4, rest=()):
    """Relative error ||model(Rx) - R model(x)|| / ||R model(x)|| for random R."""
    x = torch.randn(batch_size, features, *rest)
    rot_mats = torch.stack([rot_mat(torch.randn(1), features=features)
                            for _ in range(batch_size)], dim=0)
    x_rot = torch.einsum('bij,bj...->bi...', rot_mats, x)
    outs_rot = torch.einsum('bij,bj...->bi...', rot_mats, model(x))
    return torch.linalg.norm(model(x_rot) - outs_rot) / torch.linalg.norm(outs_rot)


# ═══════════════════════════════════════════════════════════════════
# O(k)-equivariant modules
#
# Generalise the U(1) = SO(2) modules to O(k) for arbitrary k. The feature
# space is divided into N/k blocks of size k, and O(k) acts identically on each
# block. By Schur's lemma the only linear maps commuting with the fundamental
# rep (irreducible for k >= 2) are W = w (x) I_k.
# ═══════════════════════════════════════════════════════════════════


class OkLinear(nn.Module):
    """O(k)-equivariant linear layer, weight W = w (x) I_k.

    ``w`` has shape (out_features//k, in_features//k). For k=2 this is the
    O(2) subspace of ``U1Linear`` (the w1 part, no w2).
    """

    def __init__(self, k: int, in_features: int, out_features: int,
                 bias: bool = False, dtype=None, device=None, std: float = 0.5):
        super().__init__()
        assert in_features % k == 0, f"in_features={in_features} not divisible by k={k}"
        assert out_features % k == 0, f"out_features={out_features} not divisible by k={k}"

        self.k = k
        self.in_features = in_features
        self.out_features = out_features

        self.w = nn.Parameter(torch.empty(out_features // k, in_features // k,
                                           dtype=dtype, device=device))
        nn.init.normal_(self.w, mean=0.0, std=std)

        self.bias = nn.Parameter(torch.randn(out_features, dtype=dtype, device=device)) if bias else None

    def _weight(self):
        m, n = self.w.shape
        out = torch.zeros(m * self.k, n * self.k, device=self.w.device, dtype=self.w.dtype)
        for d in range(self.k):
            out[d::self.k, d::self.k] = self.w
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x @ self._weight().T
        if self.bias is not None:
            y = y + self.bias
        return y


class OkRadialTanh(nn.Module):
    """O(k)-equivariant activation: tanh(r)/r on the k-dim norm of each block.

    For k=2 this is identical to ``RadialTanh``.
    """

    def __init__(self, k: int = 2, eps: float = 1e-12):
        super().__init__()
        self.k = k
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.reshape(x.shape[0], x.shape[1] // self.k, self.k, *x.shape[2:])
        r = torch.linalg.norm(y, dim=2, keepdim=True)
        scale = torch.tanh(r) / (r + self.eps)
        y = y * scale
        return y.reshape(x.shape)


class OkBlock(nn.Module):
    """Stack of ``num_hidden_layers`` (OkLinear -> OkRadialTanh(k)) layers.

    Set ``Ok_linear_layers=False`` for the non-equivariant baseline (plain
    ``nn.Linear`` at the same std, keeping the OkRadialTanh activation).
    """

    def __init__(
        self,
        k: int = 2,
        features: int = 4,
        num_hidden_layers: int = 20,
        activation=None,
        std: float = 0.5,
        bias: bool = False,
        Ok_linear_layers: bool = True,
    ):
        super().__init__()
        self.num_hidden_layers = num_hidden_layers
        make_act = (lambda: OkRadialTanh(k)) if activation is None else activation

        layers = []
        for _ in range(num_hidden_layers):
            if Ok_linear_layers:
                layers.append(OkLinear(k, in_features=features, out_features=features,
                                       bias=bias, std=std))
            else:
                lin = nn.Linear(in_features=features, out_features=features, bias=bias)
                nn.init.normal_(lin.weight, mean=0.0, std=std)
                layers.append(lin)
            layers.append(make_act())
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, return_all: bool = False):
        outputs: List[torch.Tensor] = []
        h = x
        for layer in self.layers:
            h = layer(h)
            outputs.append(h)
        if return_all:
            return torch.stack(outputs)
        return h


def ok_rot_mat(k: int, features: int):
    """Block-diagonal random Haar O(k) matrix acting identically on each block."""
    A = torch.randn(k, k)
    Q, R = torch.linalg.qr(A)
    Q = Q @ torch.diag(torch.sign(torch.diag(R)))
    return torch.kron(torch.eye(features // k), Q)


def ok_equivariance_check(model, k: int = 2, batch_size: int = 16,
                          features: int = 4, rest=()):
    """Relative error ||model(Rx) - R model(x)|| / ||R model(x)|| for random R in O(k)."""
    x = torch.randn(batch_size, features, *rest)
    rot_mats = torch.stack([ok_rot_mat(k, features) for _ in range(batch_size)], dim=0)
    x_rot = torch.einsum('bij,bj...->bi...', rot_mats, x)
    outs_rot = torch.einsum('bij,bj...->bi...', rot_mats, model(x))
    return torch.linalg.norm(model(x_rot) - outs_rot) / torch.linalg.norm(outs_rot)
