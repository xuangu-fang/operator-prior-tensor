"""Experiment configuration from YAML, so nothing important is hard-coded.

Every number that changes an experiment's meaning -- ranks, cutoffs, the barrier
contrast, the noise level, the observation protocol -- belongs in a file that
can be committed next to the results it produced, not in a default buried in an
``argparse`` call.  A student running a hundred variants should be editing YAML
and reading it back out of the artifact, not remembering which flags were passed.

Three rules the loader enforces, each of which cost this project a round:

- **Unknown keys are an error.**  A typo in a config that silently does nothing
  is worse than a crash, because the run completes and the number looks real.
- **Command-line flags override the file, and the merged result is recorded.**
  Every artifact stores the configuration it actually ran with, so a result can
  never be attributed to a setting it did not use.
- **Per-family settings inherit from the top level.**  Resolution and cutoff
  genuinely differ between families -- they are set from feature size and from
  the two cutoff rules -- while ranks and step counts genuinely should not.
"""

from __future__ import annotations

import copy
import dataclasses
from pathlib import Path
from typing import Any

import yaml

# The frozen values, with the reasoning recorded next to each in
# docs/HANDOVER_ZH.md.  A config file overrides any of them; anything it does
# not mention keeps the value that produced the reported table.
DEFAULTS: dict[str, Any] = {
    # --- data -------------------------------------------------------------
    "n_scenarios": 12,
    "n_time": 12,
    "truth_modes": 60,
    "contrast": 0.3,
    "reaction": 0.15,
    "time_span": [0.15, 3.0],
    "dynamics": None,          # None keeps each family's own equation
    "basis_cutoff": None,      # None keeps each family's frozen cutoff
    "resolution": None,        # None keeps each family's frozen resolution
    # --- observation ------------------------------------------------------
    "masks": ["spatial_sensors", "random"],
    "ratios": [0.10],
    "seeds": [201, 202, 203, 204, 205],
    "noise": 0.10,
    # --- model ------------------------------------------------------------
    "ranks": [12, 10, 16],
    "power": 1.5,
    "reg": 0.002,
    "hidden": 48,
    "observability_threshold": 0.1,
    # --- optimisation -----------------------------------------------------
    "steps": 1500,
    "learning_rate": 0.003,
    "device": "cuda",
    # --- what to run ------------------------------------------------------
    "families": ["plane_barrier", "plane_domain", "volume_barrier", "sphere",
                 "floorplan"],
    "models": ["geometry_operator", "blind_operator", "flat_chart",
               "neural_tucker", "neural_cp", "permuted"],
    "als_iters": 200,
}

# Settings a family may legitimately override, because they are properties of
# the geometry rather than of the experiment.
PER_FAMILY_KEYS = {"resolution", "basis_cutoff", "dynamics", "models",
                   "ranks", "time_span"}


@dataclasses.dataclass
class Config:
    """A resolved configuration: defaults, then file, then command line."""

    values: dict[str, Any]
    per_family: dict[str, dict[str, Any]]
    source: str

    def __getattr__(self, name: str) -> Any:
        try:
            return self.values[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def for_family(self, family: str) -> "Config":
        """This configuration as it applies to one family."""
        merged = dict(self.values)
        merged.update(self.per_family.get(family, {}))
        return Config(merged, {}, f"{self.source}[{family}]")

    def as_dict(self) -> dict[str, Any]:
        return {"source": self.source, "values": copy.deepcopy(self.values),
                "per_family": copy.deepcopy(self.per_family)}


def load(path: str | Path | None = None, **overrides: Any) -> Config:
    """Defaults, overlaid with a YAML file, overlaid with explicit overrides.

    ``overrides`` whose value is ``None`` are ignored, so an unset command-line
    flag does not silently erase what the file said.
    """
    values = copy.deepcopy(DEFAULTS)
    per_family: dict[str, dict[str, Any]] = {}
    source = "defaults"

    if path is not None:
        document = yaml.safe_load(Path(path).read_text()) or {}
        families = document.pop("per_family", {}) or {}
        unknown = set(document) - set(DEFAULTS)
        if unknown:
            raise ValueError(
                f"{path}: unknown configuration keys {sorted(unknown)}; "
                f"known keys are {sorted(DEFAULTS)}")
        for family, settings in families.items():
            bad = set(settings) - PER_FAMILY_KEYS
            if bad:
                raise ValueError(
                    f"{path}: per_family[{family}] may not set {sorted(bad)}; "
                    f"a family may only override {sorted(PER_FAMILY_KEYS)}")
        values.update(document)
        per_family = families
        source = str(path)

    explicit = {k: v for k, v in overrides.items() if v is not None}
    unknown = set(explicit) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown overrides {sorted(unknown)}")
    values.update(explicit)
    if explicit:
        source += " + command line"
    return Config(values, per_family, source)
