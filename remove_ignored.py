from pathlib import PurePosixPath as P
import subprocess

files = subprocess.check_output(["git", "ls-files"]).decode("utf-8", "replace").splitlines()
remove = []

for f in files:
    p = P(f)
    if f in ("data/.gitkeep", "data/ml_dataset.parquet", "data/validated_configs.json"):
        continue

    if (
        p.match("data/*")
        or p.match("**/__pycache__")
        or p.match("**/.pytest_cache")
        or f == "verification_output.txt"
        or p.name.startswith("tmp")
        or p.name.startswith("sweep_")
        or p.name.startswith("successful")
        or p.match("graphify-out*")
        or p.match(".venv*")
        or p.match("sweeps/*")
        or p.match("trade_logs/*")
        or p.name.startswith("CLAUDE")
    ):
        remove.append(f)

if remove:
    subprocess.run(["git", "rm", "--cached", "--ignore-unmatch", "-r"] + remove, check=True)