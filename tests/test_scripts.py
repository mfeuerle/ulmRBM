"""Smoke tests running every script in ``scripts/``, to ensure the examples and exercises stay up to date.

The scripts are collected automatically, i.e. new scripts are tested without changes to this file.
Each script runs in its own process with a non-interactive plotting backend and has to exit without error.

Some scripts run for several minutes, thus these tests are marked with ``scripts`` and are deselected by
default (see ``addopts`` in ``pyproject.toml``). Run them via ``pytest -m scripts``.
"""

import os
import shutil
import subprocess
from functools import cache
import sys
from pathlib import Path

import pytest

pytest.importorskip("dolfinx")
pytest.importorskip("pyvista")
pytest.importorskip("matplotlib")

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = sorted((ROOT / "scripts").rglob("*.py"))
TIMEOUT = 3600  # seconds per script


def _needs_matlab(script: Path) -> bool:
    """The modified Hilbert transform is assembled by Matlab, see `ulmRBM.fenicsx.problems.hilbert_transform`."""
    return "hilbert" in script.read_text()


@cache
def _matlab_available() -> bool:
    """Whether Matlab actually runs, i.e. is installed and licensed (e.g. a network license might be unreachable)."""
    if shutil.which("matlab") is None:
        return False
    try:
        return subprocess.run(["matlab", "-batch", "exit"], capture_output=True, timeout=300).returncode == 0
    except subprocess.TimeoutExpired:
        return False


@pytest.mark.scripts
@pytest.mark.parametrize("script", SCRIPTS, ids=[str(s.relative_to(ROOT / "scripts")) for s in SCRIPTS])
def test_script(script, tmp_path):
    if _needs_matlab(script) and not _matlab_available():
        pytest.skip("requires a working Matlab installation for the modified Hilbert transform")

    env = dict(os.environ, MPLBACKEND="Agg", PYVISTA_OFF_SCREEN="true",
               PYTHONPATH=os.pathsep.join([str(ROOT), os.environ.get("PYTHONPATH", "")]))
    # run in a temporary directory, as some scripts write output files
    result = subprocess.run([sys.executable, str(script)], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=TIMEOUT)
    assert result.returncode == 0, f"{script.name} failed:\n{result.stderr[-3000:]}"
