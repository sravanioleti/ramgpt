from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parent
env = os.environ.copy()
env.setdefault("PYTHONUNBUFFERED", "1")
python = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
if not python.exists():
    python = Path(sys.executable)

api = subprocess.Popen(
    [str(python), "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8000"],
    cwd=ROOT,
    env=env,
)

try:
    deadline = time.time() + 600
    while time.time() < deadline:
        if api.poll() is not None:
            raise RuntimeError("The API failed to start. Check the backend error above.")
        try:
            with urlopen("http://127.0.0.1:8000/api/health", timeout=2) as response:
                if response.status == 200:
                    break
        except OSError:
            time.sleep(2)
    else:
        raise RuntimeError("The API did not become ready within 10 minutes.")
    subprocess.run(
        [str(python), "-m", "streamlit", "run", "frontend.py"],
        cwd=ROOT,
        env=env,
        check=False,
    )
finally:
    if api.poll() is None:
        api.terminate()
        try:
            api.wait(timeout=5)
        except subprocess.TimeoutExpired:
            api.kill()
