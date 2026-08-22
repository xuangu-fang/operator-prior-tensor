"""Configuration loading, because a silent typo is worse than a crash.

A misspelled key that does nothing produces a completed run and a number that
looks real, attributed to a setting that was never used.  These tests pin the
three behaviours that prevent it.
"""

import pytest
import yaml

from geoaware.config import DEFAULTS, load


def test_defaults_are_the_reported_configuration():
    config = load()
    assert config.ranks == [12, 10, 16]
    assert config.steps == 1500
    assert config.seeds == [201, 202, 203, 204, 205]
    assert config.source == "defaults"


def test_a_file_overrides_defaults_and_says_so(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({"steps": 7, "ranks": [2, 2, 2]}))
    config = load(path)
    assert config.steps == 7 and config.ranks == [2, 2, 2]
    # Untouched keys keep the value that produced the reported table.
    assert config.power == DEFAULTS["power"]
    assert str(path) in config.source


def test_unknown_keys_are_rejected_rather_than_ignored(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({"stpes": 7}))
    with pytest.raises(ValueError, match="unknown configuration keys"):
        load(path)
    with pytest.raises(ValueError, match="unknown overrides"):
        load(None, stpes=7)


def test_a_family_may_override_geometry_but_not_the_protocol(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(
        {"ranks": [4, 4, 4],
         "per_family": {"sphere": {"basis_cutoff": 32, "resolution": 5}}}))
    config = load(path)
    sphere = config.for_family("sphere")
    assert sphere.basis_cutoff == 32 and sphere.resolution == 5
    # Ranks and step counts are properties of the experiment, not the geometry.
    assert sphere.ranks == [4, 4, 4]
    assert config.for_family("plane_barrier").basis_cutoff is None

    path.write_text(yaml.safe_dump({"per_family": {"sphere": {"seeds": [1]}}}))
    with pytest.raises(ValueError, match="may not set"):
        load(path)


def test_an_unset_command_line_flag_does_not_erase_the_file(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({"steps": 7}))
    assert load(path, steps=None).steps == 7
    assert load(path, steps=9).steps == 9


def test_the_shipped_configs_all_load():
    for name in ("configs/main.yaml", "configs/wave.yaml", "configs/quick.yaml"):
        config = load(name)
        assert config.steps > 0
        assert config.families
