"""Static checks for the public, code-only source release."""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTERED = {
    "25_build_window4_v2.py",
    "26_analyze_window4_v2.py",
    "27_submit_window4_v2.py",
    "28_analyze_window5_validity.py",
    "29_audit_validity_window.py",
}
ADDED_IN_1_0_1 = {
    "probe_pairing_check.py",
    "ablations.py",
    "paired72.py",
    "sim_noise_models.py",
    "make_extra_tables.py",
    "check_readout_fix.py",
}
EXPECTED = REGISTERED | ADDED_IN_1_0_1


def test_expected_script_set_is_present() -> None:
    assert {p.name for p in (ROOT / "scripts").glob("*.py")} == EXPECTED


def test_python_sources_parse() -> None:
    for path in (ROOT / "scripts").glob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_builder_has_output_pairing_option() -> None:
    source = (ROOT / "scripts" / "25_build_window4_v2.py").read_text(encoding="utf-8")
    assert "--output-pairing" in source
    assert "def readout_map(" in source


def test_no_known_credentials_or_local_paths() -> None:
    # A basic guard, not a full security audit.
    forbidden = ("crn:v1:", "PHYSIQUE_HOME_PAPAT", "nasac-faculty", "/home/aishwarya", "C:\\Users\\")
    for path in (ROOT / "scripts").glob("*.py"):
        content = path.read_text(encoding="utf-8")
        for secret in forbidden:
            assert secret not in content, f"Found forbidden string in {path.name}"


def test_code_only_release_has_no_datasets() -> None:
    assert not (ROOT / "results").exists()
    assert not (ROOT / "data").exists()
    assert list(ROOT.rglob("*.qpy")) == []


def test_version_is_consistent() -> None:
    version = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
    assert version == "1.0.1"
    assert f"version: {version}" in (ROOT / "CITATION.cff").read_text(encoding="utf-8")


def test_documentation_matches_code_only_scope() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "code-only" in readme
    assert "not" in readme.lower()
    assert (ROOT / "CITATION.cff").is_file()
    assert (ROOT / "LICENSE").is_file()
    assert "REPLACE_WITH" not in (ROOT / "CITATION.cff").read_text(encoding="utf-8")
