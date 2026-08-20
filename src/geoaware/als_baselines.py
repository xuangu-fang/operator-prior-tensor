"""Standard alternating-least-squares tensor completion, from TensorLy.

These are the baselines a tensor-methods reader expects to see first: CP fitted
by ALS and Tucker fitted by HOOI, both on the observed entries only.  They are
taken from an established library rather than reimplemented, so that a weak
showing cannot be blamed on the comparison being written by the people proposing
the alternative.

Neither knows anything about geometry.  A factor row is one node, and nodes are
just indices, so two nodes on opposite sides of an impermeable wall are as
related as two neighbours.  That is exactly the assumption the paper is about.
"""

from __future__ import annotations

import numpy as np
import tensorly as tl
import torch
from tensorly.decomposition import parafac, tucker


def _prepare(values: torch.Tensor, observed: torch.Tensor):
    """Observed entries in place, zeros elsewhere, plus the mask TensorLy wants."""
    mask = observed.reshape(values.shape).cpu().numpy().astype(int)
    filled = (values.cpu().numpy() * mask).astype(np.float64)
    return tl.tensor(filled), tl.tensor(mask)


def cp_als(values: torch.Tensor, observed: torch.Tensor, rank: int, *,
           n_iter_max: int = 200, seed: int = 0,
           tol: float = 1e-8) -> torch.Tensor:
    """CP by alternating least squares with missing entries.

    ``rank`` is the single CP rank shared by every mode -- the structural
    difference from Tucker, which is why it is quoted separately rather than
    matched entry for entry.
    """
    tensor, mask = _prepare(values, observed)
    weights, factors = parafac(tensor, rank=rank, mask=mask, init="random",
                               n_iter_max=n_iter_max, tol=tol,
                               random_state=seed, normalize_factors=False)
    return torch.as_tensor(np.asarray(tl.cp_to_tensor((weights, factors))),
                           dtype=values.dtype).reshape(values.shape)


def tucker_als(values: torch.Tensor, observed: torch.Tensor,
               ranks: tuple[int, ...], *, n_iter_max: int = 200, seed: int = 0,
               tol: float = 1e-8) -> torch.Tensor:
    """Tucker by higher-order orthogonal iteration with missing entries."""
    tensor, mask = _prepare(values, observed)
    core, factors = tucker(tensor, rank=list(ranks), mask=mask, init="random",
                           n_iter_max=n_iter_max, tol=tol, random_state=seed)
    return torch.as_tensor(np.asarray(tl.tucker_to_tensor((core, factors))),
                           dtype=values.dtype).reshape(values.shape)
