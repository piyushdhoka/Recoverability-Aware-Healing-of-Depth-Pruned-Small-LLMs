"""Every instruction-following constraint we generate must be checkable by the official IFEval checkers."""
import pytest

from rah.data.pools import INST_CONSTRAINTS, passes_qc


@pytest.mark.parametrize("cid,kw,desc", INST_CONSTRAINTS)
def test_constraint_is_valid_ifeval_instruction(cid, kw, desc):
    from lm_eval.tasks.ifeval import instructions_registry

    inst = instructions_registry.INSTRUCTION_DICT[cid](cid)
    inst.build_description(**kw)
    inst.check_following("some response text")          # must not raise


def test_qc_examples():
    assert passes_qc("math", "so #### 12", {"answer": "12"})
    assert not passes_qc("math", "so #### 13", {"answer": "12"})
    assert passes_qc("inst", "all lowercase here", {"instruction_id_list": ["change_case:english_lowercase"],
                                                    "kwargs": [{}], "prompt": "p"})
    fns = [{"name": "f", "parameters": {"type": "object", "properties": {}}}]
    assert passes_qc("tool", '{"name": "f", "arguments": {}}', {"functions": fns})
    assert not passes_qc("tool", '{"name": "g", "arguments": {}}', {"functions": fns})
    assert not passes_qc("general", "   ", {})
