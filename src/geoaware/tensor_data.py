"""Explicit multiway tensor benchmarks for the Paper-A refocus."""

from __future__ import annotations

import math
import torch

from .bases import BasisSpec, basis_on_grid
from .data import FieldDataset


def _neumann_diffusion_operator(
        size: int, contrast: float, phase: float = 0.) -> tuple[torch.Tensor, torch.Tensor]:
    """Finite-volume ``-d/dx(a(x)d/dx)`` with zero-flux boundaries.

    ``contrast`` controls log-diffusivity, not an abstract subspace rotation:
    ``a(x)=exp(contrast*(cos(2 pi x)+.35 sin(3 pi x+phase)))``.  The symmetric
    edge-flux discretization is positive semidefinite and exactly preserves the
    constant Neumann mode.
    """
    if size < 4:
        raise ValueError("diffusion grid needs at least four points")
    if contrast < 0:
        raise ValueError("contrast must be non-negative")
    x = torch.linspace(0., 1., size, dtype=torch.float64)
    midpoint = .5 * (x[:-1] + x[1:])
    diffusivity = torch.exp(contrast * (
        torch.cos(2 * math.pi * midpoint)
        + .35 * torch.sin(3 * math.pi * midpoint + phase)))
    conductance = diffusivity * (size - 1) ** 2
    operator = torch.zeros(size, size, dtype=torch.float64)
    edge = torch.arange(size - 1)
    operator[edge, edge] += conductance
    operator[edge + 1, edge + 1] += conductance
    operator[edge, edge + 1] -= conductance
    operator[edge + 1, edge] -= conductance
    return operator, diffusivity


def _relative_product_projection_residual(
        values: torch.Tensor, bases: list[torch.Tensor]) -> float:
    projected = values.double()
    for mode, basis in enumerate(bases):
        q = torch.linalg.qr(basis.double(), mode="reduced").Q
        projected = torch.tensordot(q @ q.T, projected, dims=([1], [mode]))
        projected = projected.movedim(0, mode)
    return float((values.double() - projected).norm() /
                 values.double().norm().clamp_min(1e-12))


def diffusion_green_tensor(
        shape: tuple[int, int, int] = (18, 24, 24),
        contrast: float = 0., basis_cutoff: int = 8,
        truth_modes: int = 14, reaction: float = .15) -> FieldDataset:
    """Parabolic Green-response tensor under operator misspecification.

    The entries are ``u(t, x_receiver, x_source)`` for

    ``du/dt + (L_a + reaction I)u = 0``,

    where ``L_a`` is a variable-coefficient Neumann diffusion operator.  The
    learner always uses the constant-diffusivity reference eigenspace, while
    truth uses ``contrast``-perturbed eigenpairs.  Unlike the earlier synthetic
    principal-angle benchmark, mismatch here changes both spatial eigenvectors
    and physical decay rates.  The exact oracle product-space projection
    residual is recorded in ``metadata`` rather than assumed from the input.
    """
    nt, nr, ns = shape
    if nr != ns:
        raise ValueError("Green response currently requires receiver/source grids to match")
    if not 2 <= basis_cutoff <= nr:
        raise ValueError("basis_cutoff must lie in [2, spatial grid size]")
    if not basis_cutoff <= truth_modes <= nr:
        raise ValueError("truth_modes must be at least basis_cutoff and at most grid size")

    reference, _ = _neumann_diffusion_operator(nr, 0.)
    physical, diffusivity = _neumann_diffusion_operator(nr, contrast, phase=.37)
    ref_values, ref_vectors = torch.linalg.eigh(reference)
    true_values, true_vectors = torch.linalg.eigh(physical)
    ref_scale = ref_values[1].clamp_min(1e-12)
    true_scale = true_values[1].clamp_min(1e-12)
    ref_rates = ref_values / ref_scale
    true_rates = true_values / true_scale

    # Early times preserve spatial structure but avoid the singular t=0 Green
    # kernel, which would make the task almost pure identity-matrix completion.
    time = torch.linspace(.025, .55, nt, dtype=torch.float64)
    true_decay = torch.exp(-time[:, None] *
                           (reaction + true_rates[:truth_modes][None, :]))
    spectral_weight = (1 + true_rates[:truth_modes]).pow(-.18)
    values = torch.einsum(
        "tq,xq,sq,q->txs", true_decay, true_vectors[:, :truth_modes],
        true_vectors[:, :truth_modes], spectral_weight)
    values = (values - values.mean()) / values.std().clamp_min(1e-12)

    time_basis = torch.exp(-time[:, None] *
                           (reaction + ref_rates[:basis_cutoff][None, :]))
    time_basis = torch.linalg.qr(time_basis, mode="reduced").Q.float()
    spatial_basis = ref_vectors[:, :basis_cutoff].float()
    bases = [time_basis, spatial_basis, spatial_basis.clone()]
    normalized_eigenvalues = (ref_rates[:basis_cutoff] /
                              ref_rates[1].clamp_min(1e-12)).float()
    eigenvalues = [normalized_eigenvalues.clone() for _ in range(3)]
    residual = _relative_product_projection_residual(values.float(), bases)
    metadata = {
        "pde": "du/dt + (-d/dx(a(x)d/dx) + reaction I)u = 0",
        "boundary_condition": "homogeneous Neumann (zero flux)",
        "log_diffusivity_contrast": float(contrast),
        "diffusivity_min": float(diffusivity.min()),
        "diffusivity_max": float(diffusivity.max()),
        "basis_cutoff": int(basis_cutoff),
        "truth_modes": int(truth_modes),
        "reaction": float(reaction),
        "oracle_product_projection_residual": residual,
    }
    specs = tuple(BasisSpec("neumann", max(1, basis_cutoff - 1), name)
                  for name in ("decay-time", "receiver", "source"))
    return FieldDataset(
        f"diffusion_green_c{contrast:.2f}_k{basis_cutoff}", values.float(),
        ("time", "receiver", "source"), specs, (False, False, False),
        "generated:geoaware.tensor_data.diffusion_green_tensor",
        "Variable-coefficient Neumann diffusion Green-response tensor; learner uses a finite reference-operator spectrum.",
        tuple(bases), tuple(eigenvalues), metadata)


def operator_cp_tensor(shape: tuple[int,int,int]=(20,28,36),seed: int=701) -> FieldDataset:
    """Mostly-low-rank tensor with independent mode geometry and mild mismatch."""
    specs=(BasisSpec("neumann",7,"time-interval"),
           BasisSpec("dirichlet",8,"bounded-range"),
           BasisSpec("periodic",7,"angle-circle"))
    bases=[basis_on_grid(n,s)[0] for n,s in zip(shape,specs)]
    g=torch.Generator().manual_seed(seed); rank=4
    coeff=[]
    for b in bases:
        c=torch.randn(b.shape[1],rank,generator=g)
        decay=torch.arange(1,b.shape[1]+1).float().pow(-1.15)[:,None]
        coeff.append(c*decay)
    factors=[]
    for b,c in zip(bases,coeff):
        f=b@c; f/=f.square().mean(0,keepdim=True).sqrt(); factors.append(f)
    amp=torch.tensor([1.15,-.80,.52,.30])
    values=torch.einsum("tr,xr,yr,r->txy",*factors,amp)
    # Mild off-model local interaction: not itself a CP draw from the learner.
    t=torch.linspace(0,1,shape[0])[:,None,None]
    x=torch.linspace(0,1,shape[1])[None,:,None]
    a=2*math.pi*torch.arange(shape[2])[None,None,:]/shape[2]
    residual=.13*torch.exp(-((t-.63)/.12)**2-((x-.74)/.10)**2)*torch.cos(5*a+4*t)
    values=(values+residual); values=(values-values.mean())/values.std()
    return FieldDataset("operator_cp_tensor",values.float(),("time","range","angle"),specs,
                        (False,False,True),"generated:geoaware.tensor_data.operator_cp_tensor",
                        "Rank-4 operator-factor tensor with a mild localized off-model interaction.")


def operator_tucker_tensor(shape: tuple[int,int,int]=(20,28,36),seed: int=1701) -> FieldDataset:
    """Low multilinear-rank Tucker field that is not a low CP-rank generator."""
    specs=(BasisSpec("neumann",7,"time-interval"),
           BasisSpec("dirichlet",8,"bounded-range"),
           BasisSpec("periodic",7,"angle-circle"))
    bases=[basis_on_grid(n,s)[0] for n,s in zip(shape,specs)]
    ranks=(4,5,5); g=torch.Generator().manual_seed(seed); factors=[]
    for b,r in zip(bases,ranks):
        c=torch.randn(b.shape[1],r,generator=g)/torch.arange(1,b.shape[1]+1).float()[:,None].pow(.9)
        q=torch.linalg.qr(b@c,mode="reduced").Q*math.sqrt(b.shape[0]); factors.append(q)
    core=torch.randn(*ranks,generator=g)
    core*=torch.linspace(1,.3,ranks[0])[:,None,None]
    core*=torch.linspace(1,.25,ranks[1])[None,:,None]
    core*=torch.linspace(1,.25,ranks[2])[None,None,:]
    values=torch.einsum("abc,ta,xb,yc->txy",core,*factors)
    values=(values-values.mean())/values.std()
    return FieldDataset("operator_tucker_tensor",values.float(),("time","range","angle"),specs,
                        (False,False,True),"generated:geoaware.tensor_data.operator_tucker_tensor",
                        "Operator-factor Tucker tensor with multilinear rank (4,5,5).")


def operator_mixed_tensor(mismatch: float, shape: tuple[int,int,int]=(20,28,36),
                          seed: int=701) -> FieldDataset:
    """Continuous CP-to-dense-Tucker format-mismatch benchmark.

    ``mismatch=0`` is the mildly off-model CP task; ``mismatch=1`` is an
    independent dense-core Tucker task.  Intermediate values are normalized
    mixtures and provide a controlled approximation-error axis.
    """
    if not 0 <= mismatch <= 1:
        raise ValueError("mismatch must lie in [0, 1]")
    cp = operator_cp_tensor(shape, seed)
    tucker = operator_tucker_tensor(shape, seed + 1000)
    values = (1 - mismatch) * cp.values + mismatch * tucker.values
    values = (values - values.mean()) / values.std().clamp_min(1e-8)
    return FieldDataset(
        f"operator_mixed_{mismatch:.2f}", values.float(), cp.mode_names,
        cp.basis_specs, cp.periodic,
        "generated:geoaware.tensor_data.operator_mixed_tensor",
        f"Normalized CP/Tucker mixture with format mismatch {mismatch:.2f}.")


def operator_nonaligned_tensor(shape: tuple[int, int, int] = (20, 28, 36),
                               seed: int = 2701) -> FieldDataset:
    """Smooth tensor deliberately not generated by the learner eigenbasis.

    This is a controlled approximation-error benchmark, not a realistic PDE
    dataset.  Its temporal and bounded-range factors contain coordinate warps,
    localized envelopes and non-integer phases; its periodic factor also uses
    harmonics above the seven-frequency learner truncation.  A small coupled
    interaction violates the exact separated decoder.  The benchmark therefore
    asks when an operator prior remains useful despite a known basis mismatch.
    """
    specs = (BasisSpec("neumann", 7, "time-interval"),
             BasisSpec("dirichlet", 8, "bounded-range"),
             BasisSpec("periodic", 7, "angle-circle"))
    t = torch.linspace(0, 1, shape[0])
    x = torch.linspace(0, 1, shape[1])
    a = 2 * math.pi * torch.arange(shape[2]) / shape[2]
    g = torch.Generator().manual_seed(seed)
    ranks = (4, 5, 5)

    time = torch.stack([
        torch.exp(-1.8 * t) * torch.cos(math.pi * (1.35 * t + .55 * t.square())),
        torch.exp(-((t - .28) / .16).square()),
        torch.sin(math.pi * (2.65 * t + .35 * t.square())),
        torch.sigmoid(18 * (t - .57)) - .5,
    ], 1)
    space = torch.stack([
        torch.exp(-((x - center) / width).square()) *
        torch.sin(math.pi * frequency * x + phase)
        for center, width, frequency, phase in [
            (.18, .16, 1.45, .2), (.38, .21, 2.35, -.4),
            (.64, .17, 3.55, .7), (.82, .12, 4.40, -.8),
            (.53, .30, 6.30, .1),
        ]
    ], 1)
    angle = torch.stack([
        torch.cos(k * a + phase) + .18 * torch.sin((k + 2) * a - phase)
        for k, phase in [(2, .1), (5, -.4), (8, .7), (9, -.2), (11, .5)]
    ], 1)
    factors = []
    for factor in (time, space, angle):
        factor = factor - factor.mean(0, keepdim=True)
        factors.append(factor / factor.square().mean(0, keepdim=True).sqrt().clamp_min(1e-8))
    core = torch.randn(*ranks, generator=g)
    core *= torch.linspace(1, .35, ranks[0])[:, None, None]
    core *= torch.linspace(1, .3, ranks[1])[None, :, None]
    core *= torch.linspace(1, .3, ranks[2])[None, None, :]
    values = torch.einsum("abc,ta,xb,yc->txy", core, *factors)
    coupled = (.18 * torch.exp(-((t[:, None, None] - .70) / .12).square()) *
               torch.exp(-((x[None, :, None] - .34) / .11).square()) *
               torch.cos(9 * a[None, None, :] + 5 * t[:, None, None] * x[None, :, None]))
    values = values + coupled
    values = (values - values.mean()) / values.std().clamp_min(1e-8)
    return FieldDataset(
        "operator_nonaligned_tensor", values.float(),
        ("time", "range", "angle"), specs, (False, False, True),
        "generated:geoaware.tensor_data.operator_nonaligned_tensor",
        "Smooth warped/high-frequency Tucker field outside the learner basis, plus a coupled residual.")


def operator_basis_mismatch_tensor(
        mismatch: float, shape: tuple[int, int, int] = (20, 28, 36),
        seed: int = 3701) -> FieldDataset:
    """Tucker field with a calibrated continuous operator-space mismatch.

    ``mismatch`` is not an arbitrary interpolation coefficient.  Up to floating
    point error, it is the relative Frobenius error of the best projection of
    the noiseless tensor onto the learner's three-mode operator product space.
    Aligned and orthogonal factor columns are mixed at a common principal angle;
    the Tucker core, multilinear ranks and total signal energy stay fixed.

    This is an oracle diagnostic benchmark for a bias--variance phase diagram,
    rather than a claim that real PDE misspecification is one-dimensional.
    """
    if not 0 <= mismatch <= 1:
        raise ValueError("mismatch must lie in [0, 1]")
    specs = (BasisSpec("neumann", 7, "time-interval"),
             BasisSpec("dirichlet", 8, "bounded-range"),
             BasisSpec("periodic", 7, "angle-circle"))
    ranks = (4, 5, 5)
    basis_pairs = [basis_on_grid(n, spec) for n, spec in zip(shape, specs)]
    learner_bases = [pair[0] for pair in basis_pairs]
    if any(n - torch.linalg.matrix_rank(basis) < rank
           for n, basis, rank in zip(shape, learner_bases, ranks)):
        raise ValueError("shape leaves too few off-basis dimensions for the requested ranks")

    generator = torch.Generator().manual_seed(seed)
    aligned, orthogonal = [], []
    t = torch.linspace(0, 1, shape[0])
    x = torch.linspace(0, 1, shape[1])
    angle = 2 * math.pi * torch.arange(shape[2]) / shape[2]
    smooth_off_candidates = [
        torch.stack([
            torch.exp(-1.8 * t) * torch.cos(math.pi * (1.35 * t + .55 * t.square())),
            torch.exp(-((t - .28) / .16).square()),
            torch.sin(math.pi * (2.65 * t + .35 * t.square())),
            torch.sigmoid(18 * (t - .57)) - .5,
        ], 1),
        torch.stack([
            torch.exp(-((x - center) / width).square()) *
            torch.sin(math.pi * frequency * x + phase)
            for center, width, frequency, phase in [
                (.18, .16, 1.45, .2), (.38, .21, 2.35, -.4),
                (.64, .17, 3.55, .7), (.82, .12, 4.40, -.8),
                (.53, .30, 6.30, .1),
            ]
        ], 1),
        torch.stack([
            torch.cos(k * angle + phase) + .18 * torch.sin((k + 2) * angle - phase)
            for k, phase in [(8, .1), (9, -.4), (10, .7), (11, -.2), (12, .5)]
        ], 1),
    ]
    for mode, (n, basis, eigenvalues, rank, candidates) in enumerate(
            zip(shape, learner_bases, [pair[1] for pair in basis_pairs],
                ranks, smooth_off_candidates)):
        learner_q = torch.linalg.qr(basis, mode="reduced").Q
        coefficients = torch.randn(basis.shape[1], rank, generator=generator)
        coefficients *= (1 + eigenvalues).pow(-.45)[:, None]
        # Excluding the periodic constant keeps every generated tensor centered.
        if mode == 2:
            coefficients[0] = 0
        aligned_q = torch.linalg.qr(basis @ coefficients, mode="reduced").Q
        candidates = candidates - learner_q @ (learner_q.T @ candidates)
        off_q = torch.linalg.qr(candidates, mode="reduced").Q[:, :rank]
        # Reproject once after QR to suppress leakage when a smooth candidate
        # has a large but not exact component in the learner span.
        off_q = off_q - learner_q @ (learner_q.T @ off_q)
        off_q = torch.linalg.qr(off_q, mode="reduced").Q[:, :rank]
        aligned.append(aligned_q * math.sqrt(n))
        orthogonal.append(off_q * math.sqrt(n))

    # If q is the retained amplitude in each of three modes, the squared
    # product-space projection energy is q^6.  This choice makes the requested
    # mismatch exactly sqrt(1 - q^6).
    retained = (1 - mismatch ** 2) ** (1 / 6)
    rejected = math.sqrt(max(0., 1 - retained ** 2))
    factors = [retained * inside + rejected * outside
               for inside, outside in zip(aligned, orthogonal)]
    core = torch.randn(*ranks, generator=generator)
    core *= torch.linspace(1, .35, ranks[0])[:, None, None]
    core *= torch.linspace(1, .30, ranks[1])[None, :, None]
    core *= torch.linspace(1, .30, ranks[2])[None, None, :]
    values = torch.einsum("abc,ta,xb,yc->txy", core, *factors)
    values = values / values.std().clamp_min(1e-8)
    return FieldDataset(
        f"operator_basis_mismatch_{mismatch:.2f}", values.float(),
        ("time", "range", "angle"), specs, (False, False, True),
        "generated:geoaware.tensor_data.operator_basis_mismatch_tensor",
        f"Rank-(4,5,5) Tucker field with oracle relative operator-space "
        f"projection error {mismatch:.2f}.")


def explicit_mode_bases(data: FieldDataset, kind: str="correct", seed: int=919) -> tuple[list[torch.Tensor],list[torch.Tensor]]:
    if data.operator_bases is not None:
        if data.operator_eigenvalues is None:
            raise ValueError("operator_bases require operator_eigenvalues")
        basis = [item.clone() for item in data.operator_bases]
        eig = [item.clone() for item in data.operator_eigenvalues]
        if kind == "correct":
            return basis, eig
        if kind == "permuted":
            shuffled = []
            for mode, values in enumerate(basis):
                generator = torch.Generator().manual_seed(seed + values.shape[0] + mode * 1009)
                shuffled.append(values[torch.randperm(values.shape[0], generator=generator)])
            return shuffled, eig
        if kind == "discrete":
            return [torch.eye(n) for n in data.shape], [torch.zeros(n) for n in data.shape]
        raise ValueError(kind)
    basis=[]; eig=[]
    for n,spec in zip(data.shape,data.basis_specs):
        p,e=basis_on_grid(n,spec)
        if kind=="correct": pass
        elif kind=="permuted":
            gen=torch.Generator().manual_seed(seed+n+len(basis)*1009)
            p=p[torch.randperm(n,generator=gen)]
        elif kind=="discrete":
            p=torch.eye(n); e=torch.zeros(n)
        else: raise ValueError(kind)
        basis.append(p); eig.append(e)
    return basis,eig


def flat_product_features(data: FieldDataset,max_features: int=512,
                          kind: str="correct") -> tuple[torch.Tensor,torch.Tensor]:
    basis,eigs=explicit_mode_bases(data,kind)
    combos=torch.cartesian_prod(*[torch.arange(b.shape[1]) for b in basis])
    joint=sum(eigs[m][combos[:,m]] for m in range(len(basis)))
    keep=torch.argsort(joint)[:min(max_features,len(joint))]; combos=combos[keep]; joint=joint[keep]
    indices=data.flat_indices(); phi=torch.ones(len(indices),len(combos))
    for m,b in enumerate(basis): phi*=b[indices[:,m]][:,combos[:,m]]
    phi/=phi.square().mean(0,keepdim=True).sqrt().clamp_min(1e-8)
    return phi,joint
