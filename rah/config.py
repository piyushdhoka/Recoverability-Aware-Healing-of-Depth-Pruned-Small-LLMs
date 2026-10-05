"""Config loading: configs/base.yaml deep-merged with a model file (configs/<name>.yaml)."""
import copy
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO / "configs"


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(name: str) -> dict:
    base = yaml.safe_load((CONFIG_DIR / "base.yaml").read_text(encoding="utf-8"))
    model = yaml.safe_load((CONFIG_DIR / f"{name}.yaml").read_text(encoding="utf-8"))
    cfg = deep_merge(base, model)
    cfg.setdefault("fixture_data", False)
    missing = [k for k in ("name", "model_id", "prune_main", "prune_light") if k not in cfg]
    if missing:
        raise KeyError(f"config '{name}' is missing {missing}")
    return cfg


def paths(cfg: dict) -> dict:
    """All on-disk locations for one model. results/ is small and git-tracked; artifacts/ is not."""
    data_root = REPO / ("data_fixture" if cfg.get("fixture_data") else "data")
    p = {
        "eval": data_root / "eval",
        "pools": data_root / "pools",
        "results": REPO / "results" / cfg["name"],
        "artifacts": REPO / "artifacts" / cfg["name"],
    }
    p["teacher"] = p["artifacts"] / "teacher"
    for v in p.values():
        v.mkdir(parents=True, exist_ok=True)
    return p
