"""Small shared helpers: seeding, JSON/JSONL IO, resumable run markers, logging, timing."""
import json
import logging
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_logger(name: str = "rah") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger


def write_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=_json_default), encoding="utf-8")
    os.replace(tmp, path)          # atomic: a crash never leaves a half-written result


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_jsonl(path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=_json_default) + "\n")
    os.replace(tmp, path)


def read_jsonl(path) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def is_done(path) -> bool:
    """A run directory is complete only if its result file exists (written atomically last)."""
    return Path(path).exists()


@contextmanager
def timed(logger, what: str):
    t0 = time.time()
    yield
    logger.info(f"{what} took {time.time() - t0:.1f}s")


def pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def torch_dtype(name: str):
    if name == "bfloat16" and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        return torch.float16
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]
