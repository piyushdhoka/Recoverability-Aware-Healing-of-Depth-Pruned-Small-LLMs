"""Verify that every external dataset is reachable and has the fields our builders expect.
Downloads only small pieces; run once per laptop before stage 1. Exits non-zero on any failure."""
import _bootstrap  # noqa: F401
import json
import sys


def check(name, fn):
    try:
        info = fn()
        print(f"[OK]   {name}: {info}")
        return True
    except Exception as e:  # report every failure, don't stop at the first
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")
        return False


def fmt():
    from datasets import load_dataset
    ds = load_dataset("epfl-dlab/JSONSchemaBench", "Github_easy")
    split = next(iter(ds))
    row = ds[split][0]
    json.loads(row["json_schema"])
    return f"splits={list(ds)} rows={sum(len(ds[s]) for s in ds)} cols={ds[split].column_names}"


def tool():
    from rah.data.eval_sets import _read_bfcl
    q = _read_bfcl("BFCL_v3_simple.json")
    a = _read_bfcl("possible_answer/BFCL_v3_simple.json")
    m = _read_bfcl("BFCL_v3_multiple.json")
    q0, a0 = q[0], a[0]
    assert isinstance(q0["question"][0], list) and q0["question"][0][0]["role"] == "user", q0["question"]
    assert isinstance(q0["function"], list) and "name" in q0["function"][0]
    assert isinstance(a0["ground_truth"], list) and isinstance(a0["ground_truth"][0], dict)
    return f"simple={len(q)} multiple={len(m)} answer_keys={list(a0)}"


def inst():
    from datasets import load_dataset
    ds = load_dataset("google/IFEval", split="train")
    r = ds[0]
    assert {"key", "prompt", "instruction_id_list", "kwargs"} <= set(ds.column_names)
    from rah.evaluation.scorers import score_inst
    score_inst("hello", {"key": r["key"], "prompt": r["prompt"], "instruction_id_list": r["instruction_id_list"],
                         "kwargs": r["kwargs"]})
    return f"rows={len(ds)}"


def math_():
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    assert "####" in ds[0]["answer"]
    return f"rows={len(ds)}"


def safe():
    from rah.data.eval_sets import load_xstest
    rows = load_xstest()
    n_unsafe = sum(u for _, u in rows)
    assert 0 < n_unsafe < len(rows)
    return f"rows={len(rows)} unsafe={n_unsafe}"


def know():
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test")
    assert {"question", "choices", "answer"} <= set(ds.column_names)
    return f"rows={len(ds)}"


def glaive():
    from rah.data.pools import _glaive_rows
    rows = _glaive_rows(5, 0)
    assert rows and rows[0][1] and rows[0][2]
    return f"sample_fn={rows[0][1][0]['name']} q={rows[0][2][:60]!r}"


def alpaca():
    from datasets import load_dataset
    ds = load_dataset("yahma/alpaca-cleaned", split="train")
    assert {"instruction", "input", "output"} <= set(ds.column_names)
    return f"rows={len(ds)}"


def pku():
    from datasets import load_dataset
    ds = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train")
    assert {"prompt", "is_response_0_safe", "is_response_1_safe"} <= set(ds.column_names), ds.column_names
    return f"rows={len(ds)}"


def gsm_train():
    from datasets import load_dataset
    return f"rows={len(load_dataset('openai/gsm8k', 'main', split='train'))}"


if __name__ == "__main__":
    checks = [("fmt/JSONSchemaBench", fmt), ("tool/BFCL", tool), ("inst/IFEval", inst), ("math/GSM8K", math_),
              ("safe/XSTest", safe), ("know/MMLU", know), ("pool/glaive", glaive), ("pool/alpaca", alpaca),
              ("pool/PKU-SafeRLHF", pku), ("pool/gsm8k-train", gsm_train)]
    ok = [check(n, f) for n, f in checks]
    print(f"\n{sum(ok)}/{len(ok)} dataset checks passed")
    sys.exit(0 if all(ok) else 1)
