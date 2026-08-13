"""Run human_detection.py for a manual test scenario, capturing every stdout
line (with a wall-clock timestamp prefix) to logs/<scenario>_<ts>.log.

Usage:
    python tools\\run_test.py <scenario-name> [-- extra args to human_detection.py]

Example:
    python tools\\run_test.py A1
    python tools\\run_test.py C2 -- --web

Only stdout/stderr text is captured - no config or environment values are
read or written by this script, so DB_PASSWORD / CAMERA_RTSP_URL (env-only,
see config.py) never pass through it.
"""

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _SCRIPT_DIR.parent
_LOG_DIR = _ROOT_DIR / "logs"


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)

    scenario = args[0]
    extra_args = args[1:]
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    _LOG_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = _LOG_DIR / f"{scenario}_{timestamp}.log"

    cmd = [sys.executable, str(_ROOT_DIR / "human_detection.py")] + extra_args
    print(f"[run_test] scenario={scenario!r} log={log_path}")
    print(f"[run_test] launching: {' '.join(cmd)}")
    print("[run_test] press 'q' in the video window (or Ctrl+C here) to stop")

    start = time.time()
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(f"# scenario={scenario} started={datetime.now().isoformat()}\n")
        log_file.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(_ROOT_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            for line in proc.stdout:
                elapsed = time.time() - start
                stamped = f"[t+{elapsed:7.2f}s] {line}"
                sys.stdout.write(stamped)
                log_file.write(stamped)
                log_file.flush()
        except KeyboardInterrupt:
            proc.terminate()
        finally:
            proc.wait()
            log_file.write(f"# ended={datetime.now().isoformat()} elapsed={time.time() - start:.2f}s\n")

    print(f"\n[run_test] done. log saved to {log_path}")
    print(f"[run_test] now write your notes, then run: python tools\\analyze_log.py {log_path}")


if __name__ == "__main__":
    main()
