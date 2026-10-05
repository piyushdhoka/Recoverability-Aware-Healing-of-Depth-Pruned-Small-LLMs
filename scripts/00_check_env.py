"""Stage 0: record the software/hardware environment and fail fast on missing pieces."""
import platform

import _bootstrap  # noqa: F401
import torch
import transformers

from rah.pipeline import cli
from rah.utils import write_json


def main():
    args, cfg, p = cli(__doc__)
    import datasets, peft, jsonschema, scipy  # noqa: F401
    from lm_eval.tasks.ifeval.utils import test_instruction_following_strict  # noqa: F401
    from importlib.metadata import version

    env = {
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": torch.__version__, "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_mem_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1) if torch.cuda.is_available() else None,
        "bf16": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        **{k: version(k) for k in ["transformers", "peft", "datasets", "accelerate", "jsonschema", "scipy", "lm_eval"]},
        "model_id": cfg["model_id"], "device_note": cfg.get("device_note"),
    }
    assert transformers.__version__ == "4.56.2", "pin transformers==4.56.2 on both laptops (requirements.txt)"
    write_json(p["results"] / "env.json", env)
    print(env)


if __name__ == "__main__":
    main()
