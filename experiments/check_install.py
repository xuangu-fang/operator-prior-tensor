#!/usr/bin/env python3
"""Seconds-long check that a fresh install is actually working.

Run this before anything else on a new machine.  It builds two datasets and
measures the one quantity whose value is known in advance: on a domain with no
barrier the geometry-aware and geometry-blind operators are the *same operator*,
so their projection residuals must agree exactly.  If that ratio is not 1.00,
something is wrong with the mesh or the assembly and no later number can be
trusted.

No GPU, no fitting, no downloads.
"""

from __future__ import annotations

import sys

from geoaware.benchmark import build_family
from geoaware.operator_diagnostics import product_projection_residual

EXPECTED = {"open": 1.00, "sealed_4": 11.84}
TOLERANCE = .05


def residual(data, name: str) -> float:
    basis = data.operator_matrices[f"{name}_basis"]
    return product_projection_residual(data.values, [None, None, basis])


def main() -> int:
    failures = []
    for layout, expected in EXPECTED.items():
        data = build_family("plane_barrier", layout, n_scenarios=12, n_time=12)
        aware = residual(data, "geometry_operator")
        blind = residual(data, "blind_operator")
        ratio = blind / aware
        print(f"plane_barrier/{layout:10s} nodes={data.metadata['n_nodes']:5d}  "
              f"aware={aware:.4f}  blind={blind:.4f}  ratio={ratio:5.2f}  "
              f"(expected {expected:.2f})")
        if abs(ratio - expected) > TOLERANCE * max(expected, 1.):
            failures.append(f"{layout}: got {ratio:.2f}, expected {expected:.2f}")

    if failures:
        print("\nFAILED:", "; ".join(failures))
        print("The barrier-free control must be exactly 1.00 -- with no barrier "
              "the two operators are the same object.  Anything else means the "
              "mesh or the assembly is wrong; do not run experiments yet.")
        return 1
    print("\nOK — the geometry pipeline behaves as it should.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
