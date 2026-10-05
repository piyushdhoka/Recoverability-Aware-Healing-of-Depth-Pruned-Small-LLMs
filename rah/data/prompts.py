"""Prompt templates shared by evaluation items and healing pools, so both use identical formats."""
import json

FMT_SYSTEM = "You are a helpful assistant that replies with JSON only."
TOOL_SYSTEM = (
    "You are a helpful assistant with access to the following functions:\n{functions}\n\n"
    "If a function call is needed, reply ONLY with a JSON object of the form "
    '{{"name": "<function name>", "arguments": {{<argument name>: <value>, ...}}}} and nothing else.'
)


def fmt_messages(schema: dict) -> list[dict]:
    return [
        {"role": "system", "content": FMT_SYSTEM},
        {"role": "user", "content": "Generate a realistic JSON object that is valid under the following JSON Schema. "
                                    "Output only the JSON object.\n\nSchema:\n" + json.dumps(schema)},
    ]


def tool_messages(functions: list[dict], question: str) -> list[dict]:
    return [
        {"role": "system", "content": TOOL_SYSTEM.format(functions=json.dumps(functions, indent=1))},
        {"role": "user", "content": question},
    ]


def math_messages(question: str) -> list[dict]:
    return [{"role": "user", "content": question + "\n\nSolve step by step, then give the final answer on the last "
                                                   "line in the form '#### <number>'."}]


def know_messages(question: str, choices: list[str]) -> list[dict]:
    opts = "\n".join(f"{L}. {c}" for L, c in zip("ABCD", choices))
    return [{"role": "user", "content": f"{question}\n\n{opts}\n\nAnswer with the letter of the correct option only."}]


def plain_messages(text: str) -> list[dict]:
    return [{"role": "user", "content": text}]
