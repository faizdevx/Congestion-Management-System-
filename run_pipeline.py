import subprocess
import sys
from pathlib import Path
import io
import os

# ==============================
# FIX: Force UTF-8 output on Windows
# ==============================
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
os.environ["PYTHONIOENCODING"] = "utf-8"

# ==============================
# CONFIGURATION
# ==============================
SCRIPTS = [
    "scripts/03_aggregate_congestion.py",
    "scripts/05_detect_hotspots.py",
    "scripts/06_candidate_routes.py",
    "scripts/07_assign_routes.py",
    "scripts/08_dispatch_and_simulate.py",
]

# ✅ Use the same Python environment (virtualenv)
PYTHON_EXE = sys.executable

# ==============================
# RUNNER FUNCTION
# ==============================
def run_script(script_path):
    print(f"[RUNNING] {script_path} ...", flush=True)
    result = subprocess.run(
        [PYTHON_EXE, script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    if result.returncode == 0:
        print(f"[✅ DONE] {script_path}\n", flush=True)
        return True
    else:
        # ⚙️ Replace emojis with ASCII-safe fallback
        err_msg = result.stderr.strip()
        print(f"[WARN] STDERR: {err_msg}\n", flush=True)
        print(f"[FAILED] {script_path}\n", flush=True)
        return False


def run_pipeline():
    print("🚀 Starting City Traffic Simulation Pipeline...\n", flush=True)
    success_all = True
    for s in SCRIPTS:
        success = run_script(s)
        if not success:
            success_all = False
    print("\n✅ [DONE] Pipeline Completed!\n" if success_all else "\n❌ Pipeline Completed with Errors!\n", flush=True)
    return success_all


if __name__ == "__main__":
    run_pipeline()
