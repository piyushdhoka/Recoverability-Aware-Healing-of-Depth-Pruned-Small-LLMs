import pytest

from rah.evaluation.scorers import (expected_calibration_error, extract_json, extract_number, is_refusal,
                                    score_fmt, score_inst, score_math, score_safe, score_tool)

SCHEMA = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}
FN = [{"name": "get_weather", "parameters": {"type": "dict", "properties": {"city": {}, "unit": {}}}}]
GT = [{"get_weather": {"city": ["Pune"], "unit": ["celsius", ""]}}]


def test_extract_json_handles_fences_and_prose():
    assert extract_json('Sure!\n```json\n{"x": 1}\n```') == {"x": 1}
    assert extract_json('result: {"x": 2} done') == {"x": 2}
    assert extract_json("no json here") is None


def test_fmt():
    assert score_fmt('{"x": 3}', {"schema": SCHEMA})["score"] == 1.0
    assert score_fmt('{"x": "3"}', {"schema": SCHEMA})["score"] == 0.0
    assert score_fmt("nope", {"schema": SCHEMA}) == {"score": 0.0, "parsed": False}


@pytest.mark.parametrize("resp,expected", [
    ('{"name": "get_weather", "arguments": {"city": "Pune"}}', 1.0),          # optional arg omitted
    ('{"name": "get_weather", "arguments": {"city": "pune", "unit": "celsius"}}', 1.0),
    ('{"name": "get_weather", "arguments": {"city": "Mumbai"}}', 0.0),         # wrong value
    ('{"name": "get_time", "arguments": {"city": "Pune"}}', 0.0),              # wrong function
    ('{"name": "get_weather", "arguments": {"city": "Pune", "zip": 1}}', 0.0),  # unexpected arg
    ('[{"get_weather": {"city": "Pune"}}]', 1.0),                              # alternative call format
    ('{"name": "get_weather", "arguments": {}}', 0.0),                         # missing required arg
])
def test_tool(resp, expected):
    assert score_tool(resp, {"functions": FN, "ground_truth": GT})["score"] == expected


@pytest.mark.parametrize("resp", ['{"name": 5, "arguments": {}}', '{"name": null, "arguments": {}}',
                                  '{"name": ["get_weather"], "arguments": {"city": "Pune"}}', '{"x": 1, "y": 2}',
                                  '[]', '"just a string"', '{"name": "get_weather", "arguments": [1, 2]}'])
def test_tool_degenerate_outputs_score_zero_without_crashing(resp):
    assert score_tool(resp, {"functions": FN, "ground_truth": GT})["score"] == 0.0


def test_tool_numeric_string_equivalence():
    fn = [{"name": "f", "parameters": {"properties": {"n": {}}}}]
    assert score_tool('{"name": "f", "arguments": {"n": "5"}}', {"functions": fn, "ground_truth": [{"f": {"n": [5]}}]})["score"] == 1.0


def test_math():
    assert extract_number("so 3+4 = 7\n#### 7") == 7
    assert extract_number("The answer is $1,250.") == 1250
    assert score_math("... #### 42", {"answer": "42"})["score"] == 1.0
    assert score_math("I think 41", {"answer": "42"})["score"] == 0.0


def test_safe():
    assert is_refusal("I'm sorry, but I can't help with that.")
    assert not is_refusal("Use `kill -9 <pid>` to stop the process.")
    assert score_safe("I cannot help with that.", {"unsafe": True})["score"] == 1.0
    assert score_safe("I cannot help with that.", {"unsafe": False})["score"] == 0.0


def test_inst_uses_ifeval_checker():
    meta = {"key": 1, "prompt": "p", "instruction_id_list": ["punctuation:no_comma"], "kwargs": [{}]}
    assert score_inst("no commas here", meta)["score"] == 1.0
    assert score_inst("one, two", meta)["score"] == 0.0


def test_ece():
    assert expected_calibration_error([1.0, 1.0], [1, 1]) == 0.0
    assert abs(expected_calibration_error([0.9, 0.9], [0, 0]) - 0.9) < 1e-9
