"""Bounded external execution with a pinned, isolated BOMLIP-CSP snapshot."""
from __future__ import annotations
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
from src.runtime.crystal_candidates import sha256

BOMLIP_COMMIT = "2caf530c8aa2ca14230264bfe69bfbcd6952b41e"
MACE_SHA256 = "165cce4cfec5a34b9c64d4ebf95de15d71106bb584b7291c8470f0749977c46f"


def snapshot(source: Path, destination: Path) -> Path:
    source = source.resolve()
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all"], text=True)
    if head != BOMLIP_COMMIT or dirty:
        raise ValueError(f"BOMLIP requires a clean checkout at {BOMLIP_COMMIT}; source is preserved")
    names = subprocess.check_output(["git", "-C", str(source), "ls-files", "-z"]).decode().split("\0")
    destination.mkdir()
    for name in filter(None, names):
        original = source / name
        if original.is_symlink():
            raise ValueError(f"upstream snapshot contains unsupported symlink: {name}")
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
    return destination


def pin_mace_model(upstream: Path, model: Path) -> dict:
    model = model.resolve()
    if sha256(model) != MACE_SHA256:
        raise ValueError("model SHA256 differs from paper MACE-OFF23-small")
    engine = upstream / "mace-bench/src/batchopt/relaxengine.py"
    text = engine.read_text()
    if text.count('model="small"') != 3 or "import os" not in text:
        raise ValueError("unexpected upstream model-call layout")
    before = sha256(engine)
    engine.write_text(text.replace('model="small"', 'model=os.environ["PXRD_MACE_MODEL"]'))
    return {"model_sha256": MACE_SHA256, "engine_before_sha256": before,
            "engine_after_sha256": sha256(engine), "replacement_count": 3}


def run_logged(command: list[str], *, cwd: Path, env: dict, log: Path, timeout: float):
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    with log.open("x") as stream:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    if code:
        raise RuntimeError(f"external process exited {code}; see {log}")


def record(path: Path, payload: dict):
    path.write_text(json.dumps(payload, indent=2) + "\n")
