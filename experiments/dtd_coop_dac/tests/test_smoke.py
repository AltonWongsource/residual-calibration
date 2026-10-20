from pathlib import Path
import subprocess
import sys


def test_end_to_end_smoke(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    bundle = tmp_path / "bundle.npz"
    out = tmp_path / "out"
    subprocess.run([sys.executable, str(root / "tools" / "make_smoke_bundle.py"), "--output", str(bundle)], check=True, cwd=root)
    subprocess.run([sys.executable, str(root / "run_dac_lape.py"), "--bundle", str(bundle), "--output-dir", str(out)], check=True, cwd=root)
    assert (out / "metrics.csv").exists()
    assert (out / "summary.json").exists()
