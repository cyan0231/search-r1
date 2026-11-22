import json
import os
import shutil
from pathlib import Path

def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)

def write_record(path, data):
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(data, f)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)
    sync_dir(path.parent)

class RunCheckpoints:
    """Publish only complete four-rank snapshots in this run's fixed directory."""
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.record = self.root / "checkpoint_state.json"
        self.stage = self.root / ".writing"
        self.state = json.loads(self.record.read_text()) if self.record.exists() else {}
        for step in self.kept():
            self.verify(self.folder(step))
        self.prune()

    def folder(self, step):
        return self.root / f"global_step_{step}"

    def kept(self):
        return {step for step in (self.state.get("latest"), self.state.get("best_step")) if step is not None}

    def prune(self):
        for p in self.root.glob("global_step_*"):
            suffix = p.name[len("global_step_"):]
            if suffix.isdecimal() and int(suffix) not in self.kept():
                shutil.rmtree(p)
        if self.stage.exists():
            shutil.rmtree(self.stage)

    def verify(self, folder):
        required = [folder / "data.pt"]
        for rank in range(4):
            for name in ("model","optim","extra_state"):
                required.append(folder / "actor" / f"{name}_world_size_4_rank_{rank}.pt")
        for p in required:
            if not p.is_file() or p.stat().st_size == 0:
                raise RuntimeError(f"Incomplete checkpoint: {p}")

    def commit(self, step, score, final):
        self.verify(self.stage)
        for base, _, files in os.walk(self.stage, topdown=False):
            for name in files:
                with (Path(base)/name).open("rb") as f:
                    os.fsync(f.fileno())
            sync_dir(base)
        self.stage.rename(self.folder(step))
        sync_dir(self.root)
        state = self.state.copy()
        if score is not None and (state.get("best_em") is None or score > state["best_em"]):
            state.update(best_step=step, best_em=float(score))
        state.update(last_completed_save=step, complete=final,
                     latest=state["best_step"] if final else step)
        write_record(self.record, state)
        self.state = state
        self.prune()
