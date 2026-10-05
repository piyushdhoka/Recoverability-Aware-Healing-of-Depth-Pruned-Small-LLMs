"""Build fixed evaluation items per capability, split into disjoint test (final) and dev (pilot) sets.

Item format: {"id", "cap", "messages", "meta"}; meta holds whatever the scorer needs.
Dataset field names are checked by scripts/check_datasets.py before any real run.
"""
import json
import random

from huggingface_hub import hf_hub_download

from . import prompts as P

BFCL_REPO = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"


def _split(items: list, n_test: int, n_dev: int, seed: int, cap: str) -> tuple[list, list]:
    rng = random.Random(seed)
    items = items[:]
    rng.shuffle(items)
    if len(items) < n_test + n_dev:
        raise ValueError(f"{cap}: only {len(items)} items, need {n_test + n_dev}")
    return items[:n_test], items[n_test:n_test + n_dev]


def build_fmt(n_test, n_dev, seed, subset="Github_easy", max_chars=2000):
    import jsonschema
    from datasets import load_dataset, concatenate_datasets

    dsd = load_dataset("epfl-dlab/JSONSchemaBench", subset)
    ds = concatenate_datasets([dsd[s] for s in dsd])
    items = []
    for i, row in enumerate(ds):
        raw = row["json_schema"]
        if len(raw) > max_chars:
            continue
        try:
            schema = json.loads(raw)
            jsonschema.validators.validator_for(schema).check_schema(schema)
        except Exception:
            continue
        items.append({"id": f"fmt-{i}", "cap": "fmt", "messages": P.fmt_messages(schema), "meta": {"schema": schema}})
    return _split(items, n_test, n_dev, seed, "fmt")


def _read_bfcl(fname):
    path = hf_hub_download(BFCL_REPO, fname, repo_type="dataset")
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def build_tool(n_test, n_dev, seed):
    items = []
    for cat in ("simple", "multiple"):
        qs = _read_bfcl(f"BFCL_v3_{cat}.json")
        ans = {a["id"]: a["ground_truth"] for a in _read_bfcl(f"possible_answer/BFCL_v3_{cat}.json")}
        for q in qs:
            turn = q["question"][0]
            user = [m["content"] for m in turn if m["role"] == "user"][-1]
            items.append({"id": q["id"], "cap": "tool", "messages": P.tool_messages(q["function"], user),
                          "meta": {"functions": q["function"], "ground_truth": ans[q["id"]]}})
    return _split(items, n_test, n_dev, seed, "tool")


def build_inst(n_test, n_dev, seed):
    from datasets import load_dataset

    ds = load_dataset("google/IFEval", split="train")
    items = [{"id": f"inst-{r['key']}", "cap": "inst", "messages": P.plain_messages(r["prompt"]),
              "meta": {"key": r["key"], "prompt": r["prompt"], "instruction_id_list": r["instruction_id_list"],
                       "kwargs": r["kwargs"]}} for r in ds]
    return _split(items, n_test, n_dev, seed, "inst")


def build_math(n_test, n_dev, seed):
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    items = [{"id": f"math-{i}", "cap": "math", "messages": P.math_messages(r["question"]),
              "meta": {"answer": r["answer"].split("####")[-1].strip().replace(",", "")}} for i, r in enumerate(ds)]
    return _split(items, n_test, n_dev, seed, "math")


def load_xstest():
    """Returns [(prompt, unsafe_bool)]. Tries known mirrors; unsafe = XSTest 'contrast' prompts."""
    from datasets import load_dataset

    errors = []
    for repo, split in (("walledai/XSTest", "test"), ("natolambert/xstest-v2-copy", "prompts")):
        try:
            ds = load_dataset(repo, split=split)
        except Exception as e:  # try the next mirror
            errors.append(f"{repo}: {e}")
            continue
        cols = ds.column_names
        out = []
        for r in ds:
            if "label" in cols:
                unsafe = str(r["label"]).lower() == "unsafe"
            elif "type" in cols:
                unsafe = str(r["type"]).startswith("contrast")
            else:
                raise KeyError(f"{repo}: cannot find safety label in {cols}")
            out.append((r["prompt"], unsafe))
        return out
    raise RuntimeError("XSTest unavailable: " + " | ".join(errors))


def build_safe(n_test, n_dev, seed):
    rows = load_xstest()
    items = [{"id": f"safe-{i}", "cap": "safe", "messages": P.plain_messages(p), "meta": {"unsafe": u}}
             for i, (p, u) in enumerate(rows)]
    unsafe = [it for it in items if it["meta"]["unsafe"]]
    safe = [it for it in items if not it["meta"]["unsafe"]]
    ut, ud = _split(unsafe, n_test // 2, n_dev // 2, seed, "safe/unsafe")      # balanced halves
    st, sd = _split(safe, n_test - n_test // 2, n_dev - n_dev // 2, seed, "safe/benign")
    return ut + st, ud + sd


def build_know(n_test, n_dev, seed):
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split="test")
    items = [{"id": f"know-{i}", "cap": "know", "messages": P.know_messages(r["question"], r["choices"]),
              "meta": {"answer": int(r["answer"]), "subject": r["subject"]}}
             for i, r in enumerate(ds) if len(r["choices"]) == 4]
    return _split(items, n_test, n_dev, seed, "know")


BUILDERS = {"fmt": build_fmt, "tool": build_tool, "inst": build_inst, "math": build_math,
            "safe": build_safe, "know": build_know}


# --------------------------------------------------------------------------- fixtures (smoke test)
def fixture_items(cap: str, n: int, offset: int = 0) -> list[dict]:
    """Synthetic items with the same structure as the real ones; used only by the smoke test."""
    out = []
    for i in range(offset, offset + n):
        if cap == "fmt":
            schema = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}
            out.append({"id": f"fmt-{i}", "cap": cap, "messages": P.fmt_messages(schema), "meta": {"schema": schema}})
        elif cap == "tool":
            fn = [{"name": "get_weather", "description": "weather",
                   "parameters": {"type": "dict", "properties": {"city": {"type": "string"}}, "required": ["city"]}}]
            out.append({"id": f"tool-{i}", "cap": cap, "messages": P.tool_messages(fn, "Weather in Pune?"),
                        "meta": {"functions": fn, "ground_truth": [{"get_weather": {"city": ["Pune"]}}]}})
        elif cap == "inst":
            out.append({"id": f"inst-{i}", "cap": cap, "messages": P.plain_messages("Write a haiku. No commas."),
                        "meta": {"key": i, "prompt": "Write a haiku. No commas.",
                                 "instruction_id_list": ["punctuation:no_comma"], "kwargs": [{}]}})
        elif cap == "math":
            out.append({"id": f"math-{i}", "cap": cap, "messages": P.math_messages("What is 2+3?"),
                        "meta": {"answer": "5"}})
        elif cap == "safe":
            out.append({"id": f"safe-{i}", "cap": cap, "messages": P.plain_messages("How do I kill a process?"),
                        "meta": {"unsafe": i % 2 == 0}})
        elif cap == "know":
            out.append({"id": f"know-{i}", "cap": cap,
                        "messages": P.know_messages("2+2=?", ["3", "4", "5", "6"]), "meta": {"answer": 1}})
    return out
