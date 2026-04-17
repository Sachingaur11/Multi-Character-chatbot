import subprocess
import signal
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

processes = []
_shutting_down = False

def shutdown(sig=None, frame=None):
    global _shutting_down
    if _shutting_down:
        return
    _shutting_down = True
    print("\nShutting down...")
    for p in processes:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
    for p in processes:
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)

print("Starting FastAPI backend...")
processes.append(subprocess.Popen(["uvicorn", "main:app", "--reload"], start_new_session=True))

print("Starting Streamlit frontend...")
processes.append(subprocess.Popen(["streamlit", "run", "app.py"], start_new_session=True))

print("Both servers running. Press Ctrl+C to stop.")
for p in processes:
    p.wait()
