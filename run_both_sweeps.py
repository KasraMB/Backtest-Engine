"""
Run AnchoredMeanReversion sweep (Phase 2 + 3) → then Account mgmt sweep.
Designed to be launched detached via pythonw.

Subprocess output is forwarded line-by-line to the parent's stdout (which
itself is captured to a file by Start-Process -RedirectStandardOutput).
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
PY       = sys.executable


def _run(script: str) -> int:
    t0 = time.time()
    print(f"\n{'='*60}", flush=True)
    print(f"== STARTING: {script}", flush=True)
    print(f"== {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"{'='*60}\n", flush=True)
    # Use Popen with PIPE so we can forward output explicitly (pythonw can't inherit consoles)
    proc = subprocess.Popen(
        [PY, "-u", script],          # -u: unbuffered output
        cwd=WORK_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
    proc.wait()
    dur = time.time() - t0
    print(f"\n{'='*60}", flush=True)
    print(f"== FINISHED: {script}  exit={proc.returncode}  duration={dur:.0f}s", flush=True)
    print(f"{'='*60}\n", flush=True)
    return proc.returncode


def main():
    # Step 1: AnchoredMeanReversion sweep (resumes from A checkpoint, reruns Phase 2+3)
    rc = _run("tmp_amr_v2_sweep.py")
    if rc != 0:
        print(f"AnchoredMeanReversion sweep failed with exit {rc}, aborting.", flush=True)
        sys.exit(rc)

    # Step 2: Reset account mgmt checkpoint so it picks up fresh AnchoredMeanReversion results
    ckpt = os.path.join(WORK_DIR, "account_mgmt_checkpoint.pkl")
    if os.path.exists(ckpt):
        os.remove(ckpt)
        print(f"Removed stale account_mgmt_checkpoint.pkl", flush=True)

    # Step 3: Account management sweep
    rc = _run("tmp_account_mgmt_sweep.py")
    if rc != 0:
        print(f"Account mgmt sweep failed with exit {rc}", flush=True)
        sys.exit(rc)

    print("\nALL SWEEPS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
