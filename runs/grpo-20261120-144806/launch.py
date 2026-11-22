import os, signal, subprocess, time, urllib.request
from pathlib import Path
d = Path(__file__).resolve().parent
import fcntl, json
lock = (d / "supervisor.lock").open("a")
try:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    raise SystemExit("This training run already has an active supervisor.")
record = d.parents[1] / "checkpoints" / d.name / "checkpoint_state.json"
if record.exists() and json.loads(record.read_text()).get("complete"):
    from run_checkpoints import RunCheckpoints
    RunCheckpoints(record.parent)  # Finish any post-commit cleanup interrupted on exit.
    print("Training already completed.")
    raise SystemExit(0)
children = []
tag = "SEARCH_R1_RUN=" + d.name
env = dict(os.environ, SEARCH_R1_RUN=d.name, BATCH="40", MICRO="2", LOG_PROB_MICRO="8")
def start(script, args):
    with (d / (script + ".log")).open("ab") as log:
        proc = subprocess.Popen(["bash", str(d / (script + ".sh")), *args], stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)
    children.append(proc)
    return proc
def tagged():
    result=[]
    for p in Path("/proc").iterdir():
        if not p.name.isdigit(): continue
        try:
            if p.stat().st_uid == os.getuid() and tag.encode() in (p / "environ").read_bytes().split(b"\0"):
                result.append(int(p.name))
        except OSError: pass
    return result
def cleanup():
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for p in children:
            if p.poll() is None:
                try: os.killpg(p.pid, sig)
                except ProcessLookupError: pass
        for pid in tagged():
            try: os.kill(pid,sig)
            except ProcessLookupError: pass
        for _ in range(12):
            if not tagged(): break
            time.sleep(1)
for sig in (signal.SIGTERM,signal.SIGINT):
    signal.signal(sig, lambda *_: (_ for _ in ()).throw(SystemExit(143)))
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
try:
    if tagged():
        print("Clearing leftover workers from this same run.", flush=True)
        cleanup()
        if tagged():
            raise RuntimeError("Previous workers still running")
    retrieval=start("retrieval",[])
    for _ in range(300):
        if retrieval.poll() is not None: raise RuntimeError("retrieval exited before training")
        try:
            with opener.open("http://127.0.0.1:8000/docs", timeout=3): break
        except OSError: time.sleep(3)
    else: raise RuntimeError("retrieval startup timed out")
    trainer=start("train",["full"])
    print(f"train pid={trainer.pid}, retrieval pid={retrieval.pid}",flush=True)
    while trainer.poll() is None:
        if retrieval.poll() is not None: raise RuntimeError("retrieval exited during training")
        time.sleep(5)
    print(f"trainer exit={trainer.returncode}",flush=True)
    raise SystemExit(trainer.returncode)
finally:
    cleanup()
    print("cleanup complete",flush=True)
