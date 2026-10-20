from pathlib import Path
import subprocess
import sys


def test_fast_pipeline(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    master = tmp_path / "master.npz"
    output = tmp_path / "out"
    subprocess.run(
        [sys.executable, str(root / "tools" / "make_smoke_master_bundle.py"), "--output", str(master)],
        check=True,
        cwd=root,
    )
    subprocess.run(
        [
            sys.executable,
            str(root / "run_fast_rebuttal.py"),
            "--master-bundle",
            str(master),
            "--output-dir",
            str(output),
            "--split-seeds",
            "0,1",
        ],
        check=True,
        cwd=root,
    )
    assert (output / "strict_metrics.csv").exists()
    assert (output / "extension_aggregate.csv").exists()
    assert (output / "run_manifest.json").exists()
