# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Include installed-consumer runner policy in the existing script suite."""
from pathlib import Path
import subprocess
import sys


def test_backend_acceptance_runner(tmp_path):
    source = Path(__file__).resolve().parents[2]
    subprocess.run([sys.executable, "-m", "pytest", "-q", "--basetemp",
                    str(tmp_path / "acceptance"),
                    str(source / "tests/packaging/backend_acceptance/test_matrix.py")],
                   check=True, cwd=source)
