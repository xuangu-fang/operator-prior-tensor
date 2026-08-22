"""Group-wise Operator-Prior Tucker over an explicit coordinate partition.

The frozen order-3 model in :mod:`geoaware.tensor_bayes` attaches one
operator to every tensor axis.  That is only correct when every axis really carries its own
operator.  Here the axes are first partitioned into coordinate groups, and each
group gets exactly one factor:

``operator``
    ``F_g = Phi_g W_g`` from the joint spectrum of the group's operator, with
    the same unit-RMS normalization and Sobolev spectral penalty as the frozen
    model.
``table``
    A free factor table with an optional quadratic smoothness penalty built
    from the same operator.  This is the "any smoothness would do" control: it
    sees identical physics but never truncates the spectrum.
``neural``
    A small MLP on the group's joint coordinates, for groups with no credible
    operator.  This is what keeps the method from demanding an invented PDE per
    axis.

Setting every group to a singleton axis recovers the mode-wise model, so the
frozen results remain a special case rather than a separate method.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Sequence

import torch
from torch import nn

from .tensor_bayes import (TensorBayesPrediction, _normalize_columns,
                           _normalized_spectral_coefficients)


def _symmetrize_matrix(matrix: torch.Tensor) -> torch.Tensor:
    return .5 * (matrix + matrix.T)


@dataclass
class GroupFactorSpec:
    """How one coordinate group turns its index into a factor row."""

    kind: str
    rank: int
    size: int
    basis: torch.Tensor | None = None
    eigenvalues: torch.Tensor | None = None
    penalty_operator: torch.Tensor | None = None
    coordinates: torch.Tensor | None = None
    hidden: int = 48
    name: str = ""

    def __post_init__(self):
        if self.kind == "operator":
            if self.basis is None or self.eigenvalues is None:
                raise ValueError("operator group needs a basis and its eigenvalues")
            if self.basis.shape[0] != self.size:
                raise ValueError("operator basis rows must match the group size")
            if self.basis.shape[1] != len(self.eigenvalues):
                raise ValueError("one eigenvalue per basis column is required")
        elif self.kind == "table":
            if (self.penalty_operator is not None
                    and self.penalty_operator.shape != (self.size, self.size)):
                raise ValueError("penalty operator must be size x size")
        elif self.kind == "neural":
            if self.coordinates is None or self.coordinates.shape[0] != self.size:
                raise ValueError("neural group needs one coordinate row per index")
        else:
            raise ValueError(f"unknown group factor kind: {self.kind}")


def grouped_indices(shape: Sequence[int],
                    groups: Sequence[Sequence[int]]) -> torch.Tensor:
    """Map every flat entry to its per-group index.

    The returned rows follow C order over ``shape``, i.e. the same order as
    ``values.flatten()``, so masks and metrics computed on the raw tensor stay
    valid after grouping without any re-indexing.
    """
    axes = [axis for group in groups for axis in group]
    if sorted(axes) != list(range(len(shape))):
        raise ValueError("groups must partition every axis exactly once")
    full = torch.stack(torch.meshgrid(*[torch.arange(n) for n in shape],
                                      indexing="ij"), -1).reshape(-1, len(shape))
    columns = []
    for group in groups:
        index = torch.zeros(len(full), dtype=torch.long)
        for axis in group:
            index = index * shape[axis] + full[:, axis]
        columns.append(index)
    return torch.stack(columns, 1)


def group_coordinates(shape: Sequence[int], group: Sequence[int]) -> torch.Tensor:
    """Normalized coordinates of every index inside one group."""
    axes = [torch.linspace(0., 1., shape[axis]) for axis in group]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"),
                       -1).reshape(-1, len(group))


def sobolev_penalty_operator(stiffness: torch.Tensor, mass: torch.Tensor,
                             power: float, reference: torch.Tensor) -> torch.Tensor:
    """Matrix ``P`` with ``phi_k^T P phi_k = (1 + lambda_k)^p`` on eigenvectors.

    A table factor penalized with this matrix pays exactly the same price for
    operator frequency as a spectral factor does through its coefficients.  The
    only remaining difference is that the table is not confined to the leading
    modes, which is precisely the comparison the paper needs.
    """
    from .operator_diagnostics import inverse_sqrt, matrix_sqrt

    whitener = inverse_sqrt(mass)
    whitened = whitener @ stiffness.double() @ whitener
    whitened = .5 * (whitened + whitened.T)
    values, vectors = torch.linalg.eigh(whitened)
    span = (reference[1] - reference[0]).clamp_min(1e-12).double()
    normalized = ((values - reference[0].double()) / span).clamp_min(0.)
    scaled = vectors @ torch.diag((1 + normalized).pow(power)) @ vectors.T
    root = matrix_sqrt(mass)
    return (root @ scaled @ root).float()


class _NeuralGroupFactor(nn.Module):
    def __init__(self, coordinates: torch.Tensor, rank: int, hidden: int):
        super().__init__()
        self.register_buffer("coordinates", 2 * coordinates.float() - 1)
        self.net = nn.Sequential(
            nn.Linear(coordinates.shape[1], hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, rank))

    def forward(self) -> torch.Tensor:
        return self.net(self.coordinates)


class GroupedOperatorTucker(nn.Module):
    """Tucker model whose factors live on coordinate groups, not on axes."""

    def __init__(self, specs: Sequence[GroupFactorSpec], power: float = 1.5,
                 device: str = "cuda", core: str = "dense"):
        """``core="diagonal"`` makes this a CP model instead of a Tucker one.

        Every group then needs the same rank, and the core is the vector of
        component weights rather than a full array.  Having both under one class
        matters for the comparison: a CP baseline and the proposed model share
        the optimizer, the normalization, the prior and the closed-form core
        posterior, so a difference between them is a difference in the model and
        not in how it was fitted.
        """
        super().__init__()
        if len(specs) < 2:
            raise ValueError("a grouped Tucker needs at least two groups")
        if core not in ("dense", "diagonal"):
            raise ValueError("core must be 'dense' or 'diagonal'")
        self.specs = list(specs)
        self.ranks = tuple(int(spec.rank) for spec in self.specs)
        self.diagonal_core = core == "diagonal"
        if self.diagonal_core and len(set(self.ranks)) != 1:
            raise ValueError("a diagonal core requires one rank for every group")
        self.power = power
        self.device_name = device if torch.cuda.is_available() else "cpu"

        self.coefficients = nn.ParameterList()
        self.tables = nn.ParameterList()
        self.networks = nn.ModuleList()
        self._slot = []
        for spec in self.specs:
            if spec.kind == "operator":
                self._slot.append(("operator", len(self.coefficients)))
                self.coefficients.append(nn.Parameter(
                    torch.randn(spec.basis.shape[1], spec.rank)
                    / math.sqrt(spec.basis.shape[1])))
            elif spec.kind == "table":
                self._slot.append(("table", len(self.tables)))
                self.tables.append(nn.Parameter(
                    torch.randn(spec.size, spec.rank) / math.sqrt(spec.size)))
            else:
                self._slot.append(("neural", len(self.networks)))
                self.networks.append(_NeuralGroupFactor(
                    spec.coordinates, spec.rank, spec.hidden))
        core_shape = (self.ranks[0],) if self.diagonal_core else self.ranks
        self.core = nn.Parameter(
            torch.randn(*core_shape) / math.sqrt(math.prod(core_shape)))
        self._buffers_moved = False
        self._posterior = None

    def _to_device(self):
        device = torch.device(self.device_name)
        if not self._buffers_moved:
            for spec in self.specs:
                if spec.basis is not None:
                    spec.basis = spec.basis.float().to(device)
                if spec.eigenvalues is not None:
                    spec.eigenvalues = spec.eigenvalues.float().to(device)
                if spec.penalty_operator is not None:
                    spec.penalty_operator = spec.penalty_operator.float().to(device)
            self._buffers_moved = True
        return device

    def factor_tables(self) -> list[torch.Tensor]:
        out = []
        for spec, (kind, slot) in zip(self.specs, self._slot):
            if kind == "operator":
                values = spec.basis @ self.coefficients[slot]
            elif kind == "table":
                values = self.tables[slot]
            else:
                values = self.networks[slot]()
            out.append(_normalize_columns(values)[0])
        return out

    @staticmethod
    def design(indices: torch.Tensor, factors: Sequence[torch.Tensor], *,
               diagonal: bool = False) -> torch.Tensor:
        """Row-wise Kronecker product of the selected factor rows.

        With ``diagonal=True`` the row-wise *Khatri-Rao* product is taken
        instead, which is the design matrix of a CP model: the core collapses
        from a full ``r1 x r2 x r3`` array to a length-``r`` vector of component
        weights.  Both forms are linear in the core, so the same closed-form
        posterior serves CP and Tucker without a second derivation.
        """
        z = factors[0][indices[:, 0]]
        for mode in range(1, len(factors)):
            other = factors[mode][indices[:, mode]]
            z = (z * other if diagonal else
                 (z[:, :, None] * other[:, None, :]).reshape(len(indices), -1))
        return z

    def contract(self, indices: torch.Tensor,
                 factors: Sequence[torch.Tensor]) -> torch.Tensor:
        """Predict without ever forming the row-wise Kronecker product.

        The design matrix has one column per core entry, so at a core of
        nineteen hundred it is larger than the data by an order of magnitude and
        is rebuilt at every gradient step.  Contracting the core against one
        factor at a time gives the identical result for a fraction of the
        arithmetic and a small fraction of the memory, which is what makes ranks
        big enough to reach the approximation floor affordable at all.
        """
        if self.diagonal_core:
            rows = factors[0][indices[:, 0]]
            for mode in range(1, len(factors)):
                rows = rows * factors[mode][indices[:, mode]]
            return rows @ self.core
        partial = torch.einsum("na,a...->n...", factors[0][indices[:, 0]],
                               self.core)
        for mode in range(1, len(factors)):
            partial = torch.einsum("nb,nb...->n...",
                                   factors[mode][indices[:, mode]], partial)
        return partial

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return self.contract(indices, self.factor_tables())

    def factor_prior(self) -> torch.Tensor:
        total = self.core.square().mean()
        for spec, (kind, slot) in zip(self.specs, self._slot):
            if kind == "operator":
                precision = (1 + spec.eigenvalues).pow(self.power)[:, None]
                normalized = _normalized_spectral_coefficients(
                    spec.basis, self.coefficients[slot])
                total = total + (precision * normalized.square()).mean()
            elif kind == "table" and spec.penalty_operator is not None:
                factor = _normalize_columns(self.tables[slot])[0]
                energy = (factor * (spec.penalty_operator @ factor)).sum()
                total = total + energy / (spec.size * spec.rank)
            elif kind == "neural":
                total = total + sum(p.square().mean()
                                    for p in self.networks[slot].parameters())
        return total

    def fit(self, indices_obs: torch.Tensor, y_obs: torch.Tensor, *,
            steps: int = 400, lr: float = 3e-3, reg_weight: float = 2e-3,
            seed: int = 0) -> "GroupedOperatorTucker":
        torch.manual_seed(seed)
        device = self._to_device()
        self.to(device)
        ix, y = indices_obs.to(device), y_obs.to(device)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-6)
        history, best = [], (float("inf"), None)
        for step in range(steps):
            prediction = self(ix)
            data_loss = (prediction - y).square().mean()
            loss = data_loss + reg_weight * self.factor_prior()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 5.)
            optimizer.step()
            value = float(loss.detach())
            if value < best[0]:
                best = (value, {k: v.detach().cpu().clone()
                                for k, v in self.state_dict().items()})
            if step % max(1, steps // 10) == 0 or step == steps - 1:
                history.append({"step": step, "loss": value,
                                "data_loss": float(data_loss.detach())})
        self.load_state_dict(best[1])
        self.to(device)
        self._fit_core_posterior(ix, y)
        self._posterior["history"] = history
        self._posterior["best_observed_objective"] = best[0]
        return self

    @torch.no_grad()
    def _fit_core_posterior(self, ix: torch.Tensor, y: torch.Tensor):
        """Exact Gaussian core posterior conditional on the fitted factors."""
        z = self.design(ix, self.factor_tables(),
                        diagonal=self.diagonal_core).double()
        yd = y.double()
        p = z.shape[1]
        # ``z^T z`` does not change inside the evidence loop, so it is
        # diagonalized once.  In that eigenbasis every quantity the fixed point
        # needs is a function of the eigenvalues alone -- the covariance is
        # diagonal, its trace is a sum, and the mean is a rescaling -- which
        # turns each iteration from a p-by-p inverse into O(p).  At a core of
        # ninety-six that is bookkeeping; at nineteen hundred it is the
        # difference between a fit dominated by this loop and one that is not.
        gram = _symmetrize_matrix(z.T @ z)
        spectrum, rotation = torch.linalg.eigh(gram)
        spectrum = spectrum.clamp_min(0.)
        projected = rotation.T @ (z.T @ yd)
        target = yd.square().sum()

        alpha = torch.tensor(1., device=z.device, dtype=z.dtype)
        beta = torch.tensor(25., device=z.device, dtype=z.dtype)
        for _ in range(80):
            precision = beta * spectrum + alpha
            rotated_mean = beta * projected / precision
            gamma = (p - alpha * (1. / precision).sum()).clamp(1e-3, p - 1e-3)
            alpha_new = (gamma / rotated_mean.square().sum().clamp_min(1e-8)
                         ).clamp(1e-4, 1e5)
            # ||y - z m||^2 with m expressed in the eigenbasis of z^T z.
            resid = (target - 2 * (rotated_mean * projected).sum()
                     + (rotated_mean.square() * spectrum).sum()).clamp_min(0.)
            beta_new = ((len(yd) - gamma).clamp_min(1.) /
                        resid.clamp_min(1e-8)).clamp(1e-3, 1e5)
            converged = max(float((alpha_new - alpha).abs()),
                            float((beta_new - beta).abs())) < 1e-5
            alpha, beta = alpha_new, beta_new
            if converged:
                break
        precision = beta * spectrum + alpha
        cov = (rotation / precision) @ rotation.T
        mean = beta * (rotation @ (projected / precision))
        fitted = z @ mean
        leverage = (beta * ((z @ cov) * z).sum(1)).clamp(max=.999)
        loo_resid = (yd - fitted) / (1 - leverage).clamp_min(1e-4)
        loo_std = torch.sqrt(1 / beta + ((z @ cov) * z).sum(1))
        calibration = float(torch.quantile(
            loo_resid.abs() / loo_std.clamp_min(1e-8), .95) / 1.96)
        self._posterior = {"mean": mean.float(), "cov": cov.float(),
                           "alpha": float(alpha), "noise": float(beta.rsqrt()),
                           "calibration": max(.5, min(4., calibration))}

    @torch.no_grad()
    def predict(self, all_indices: torch.Tensor, *, chunk_size: int = 8192
                ) -> TensorBayesPrediction:
        if self._posterior is None:
            raise RuntimeError("fit first")
        device = next(self.parameters()).device
        ix = all_indices.to(device)
        factors = self.factor_tables()
        mean_core = self._posterior["mean"].to(device)
        cov = self._posterior["cov"].to(device)
        means, variances = [], []
        for start in range(0, len(ix), chunk_size):
            z = self.design(ix[start:start + chunk_size], factors,
                            diagonal=self.diagonal_core)
            means.append((z @ mean_core).cpu())
            variances.append(((z @ cov) * z).sum(1).clamp_min(0).cpu())
        mean, var = torch.cat(means), torch.cat(variances)
        std = ((var + self._posterior["noise"] ** 2).sqrt()
               * self._posterior["calibration"])
        spectral = []
        for spec, (kind, slot) in zip(self.specs, self._slot):
            if kind == "operator":
                normalized = _normalized_spectral_coefficients(
                    spec.basis, self.coefficients[slot]).detach().cpu()
                spectral.append(((1 + spec.eigenvalues.cpu()[:, None])
                                 .pow(self.power) * normalized.square()).sum(0))
            else:
                spectral.append(torch.zeros(spec.rank))
        core_energy = mean_core.cpu().square() + cov.diagonal().cpu()
        return TensorBayesPrediction(
            mean, std, self.ranks,
            torch.full_like(core_energy, self._posterior["alpha"]), core_energy,
            spectral, None, self._posterior["history"],
            {"ranks": self.ranks, "power": self.power,
             "core_kind": "diagonal" if self.diagonal_core else "dense",
             "core_size": int(self.core.numel()),
             "group_kinds": [spec.kind for spec in self.specs],
             "group_names": [spec.name for spec in self.specs],
             "group_sizes": [spec.size for spec in self.specs],
             "group_basis_columns": [
                 int(spec.basis.shape[1]) if spec.basis is not None else 0
                 for spec in self.specs],
             "core_precision": self._posterior["alpha"],
             "noise": self._posterior["noise"],
             "calibration": self._posterior["calibration"],
             "best_observed_objective": self._posterior["best_observed_objective"]})

    def spatial_latent_dimension(self) -> int:
        """Total number of basis columns the model may combine.

        Reported alongside parameter count because joint and per-axis variants
        are only comparable once both budgets are on the table.
        """
        return sum(int(spec.basis.shape[1]) for spec in self.specs
                   if spec.basis is not None)
