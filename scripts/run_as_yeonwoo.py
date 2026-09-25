"""Run scripts with the server-required GPU process name yeonwoo.

Usage: python -B scripts/run_as_yeonwoo.py scripts/<script>.py [arguments...]
Check: python -B scripts/run_as_yeonwoo.py --check-gpu-name
"""
from __future__ import annotations
import csv
import io
import json
import os
from pathlib import Path
import pwd
import runpy
import subprocess
import sys


def main():
    if pwd.getpwuid(os.geteuid()).pw_name != "yeonwoo":
        raise SystemExit("Run under the yeonwoo account.")
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    # Fail before GPU use if this required package is missing.
    import setproctitle
    os.environ["MHW_PROCESS_NAME"] = "yeonwoo"
    setproctitle.setproctitle("yeonwoo")
    if sys.argv[1] == "--check-gpu-name":
        import torch
        allocation = torch.zeros(1, device="cuda:0")
        torch.cuda.synchronize()
        result = subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ], text=True)
        rows = list(csv.reader(io.StringIO(result)))
        own = [row for row in rows if int(row[0].strip()) == os.getpid()]
        if not own or any(row[1].strip() != "yeonwoo" for row in own):
            raise SystemExit(f"GPU process name check failed: {own!r}")
        print(json.dumps({"status": "passed", "user": "yeonwoo", "pid": os.getpid(),
                          "process_name": own[0][1].strip(),
                          "gpu_memory_mib": own[0][2].strip()}))
        del allocation
        return
    script = Path(sys.argv[1]).resolve(strict=True)
    sys.argv = [str(script), *sys.argv[2:]]
    sys.path.insert(0, str(script.parent))
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
