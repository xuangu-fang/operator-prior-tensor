"""Standard alternating-least-squares tensor completion, from TensorLy.

These are the baselines a tensor-methods reader expects to see first: CP fitted
by ALS and Tucker fitted by HOOI, both on the observed entries only.  They are
taken from an established library rather than reimplemented, so that a weak
showing cannot be blamed on the comparison being written by the people proposing
the alternative.

Neither knows anything about geometry.  A factor row is one node, and nodes are
just indices, so two nodes on opposite sides of an impermeable wall are as
related as two neighbours.  That is exactly the assumption the paper is about.

Two decisions here exist to keep the comparison fair rather than flattering.

Initialization is by SVD, not random.  With random starts and an EM fill-in for
the missing entries, both routines diverge badly on this data -- held-out NRMSE
of 3 to 20, which would be reporting a broken baseline rather than a weak one.
From an SVD start they are well behaved: real competitors under random entries,
and exactly uninformative under sensor sampling, which is the honest statement.

Rank is chosen by held-out error.  That is oracle knowledge the baseline would
not have in practice, and it is granted deliberately: the proposed model runs at
one fixed rank, so any remaining margin is not a matter of tuning.
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
           n_iter_max: int = 200, seed: int = 0, init: str = "svd",
           tol: float = 1e-8) -> torch.Tensor:
    """CP by alternating least squares with missing entries.

    ``rank`` is the single CP rank shared by every mode -- the structural
    difference from Tucker, which is why it is quoted separately rather than
    matched entry for entry.
    """
    tensor, mask = _prepare(values, observed)
    weights, factors = parafac(tensor, rank=rank, mask=mask, init=init,
                               n_iter_max=n_iter_max, tol=tol,
                               random_state=seed, normalize_factors=False)
    return torch.as_tensor(np.asarray(tl.cp_to_tensor((weights, factors))),
                           dtype=values.dtype).reshape(values.shape)


def tucker_als(values: torch.Tensor, observed: torch.Tensor,
               ranks: tuple[int, ...], *, n_iter_max: int = 200, seed: int = 0,
               init: str = "svd", tol: float = 1e-8) -> torch.Tensor:
    """Tucker by higher-order orthogonal iteration with missing entries."""
    tensor, mask = _prepare(values, observed)
    core, factors = tucker(tensor, rank=list(ranks), mask=mask, init=init,
                           n_iter_max=n_iter_max, tol=tol, random_state=seed)
    return torch.as_tensor(np.asarray(tl.tucker_to_tensor((core, factors))),
                           dtype=values.dtype).reshape(values.shape)


def best_of_ranks(fit, candidates, values: torch.Tensor,
                  held_out: torch.Tensor):
    """Run ``fit`` at every candidate rank and keep whichever scores best.

    Selection is on held-out error, which the baseline would not have access to.
    Granting it is the point: it turns the reported number into an upper bound on
    what the classical method can do here, so the comparison cannot be answered
    with "you did not tune it".
    """
    best = None
    for rank in candidates:
        predicted = fit(rank).flatten()
        error = predicted[held_out] - values.flatten()[held_out]
        score = float(error.square().mean().sqrt()
                      / values.flatten()[held_out].std().clamp_min(1e-8))
        if best is None or score < best[0]:
            best = (score, rank, predicted)
    return best
