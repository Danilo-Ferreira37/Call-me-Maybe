from pydantic import BaseModel, ValidationError, ConfigDict
from typing import List, Literal, Dict, Tuple
import json
import os
import argparse


class FuncCall(BaseModel):
    """Represents a parsed function‑calling prompt from the JSON input file."""
    prompt: str
    model_config = ConfigDict(extra="forbid")


class VariableType(BaseModel):
    """Represents a parsed variable type from the JSON
    function definition file."""

    type: Literal["number", "string", "boolean", "integer"]
    model_config = ConfigDict(extra="forbid")


class FuncDef(BaseModel):
    """Represents a parsed function defi loadnitioned
    from the JSON input file."""

    name: str
    description: str
    parameters: Dict[str, VariableType]
    returns: VariableType
    model_config = ConfigDict(extra="forbid")


def parse_args() -> argparse:
    parser = argparse.ArgumentParser()
    parser.add_argument("--functions_definition",
                        default="data/input/functions_definition.json")
    parser.add_argument("--input",
                        default="data/input/function_calling_tests.json")
    parser.add_argument("--output",
                        default="data/output/function_calls.json")

    return parser.parse_args()


def parse_funcdef(path: str) -> List[FuncDef]:
    """Parse the JSON file containing function
    definitions and return them as FuncDef objects."""
    try:
        with open(path) as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        print("Error in the json file:", e)
        exit(1)

    if not isinstance(config, list):
        print("Error: The file must have a list on the top")
        exit(1)

    try:
        return [FuncDef.model_validate(item) for item in config]
    except ValidationError:
        print("Error: The input file functions_definition.json"
              " is in a wrong format!")
        exit(1)


def parse_funccall(path: str) -> List[FuncCall]:
    """Parse the JSON file containing function‑calling
    prompts and return them as FuncCall objects."""
    try:
        with open(path) as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        print("Error in the json file:", e)
        exit(1)

    if not isinstance(config, list):
        print("Error: The file must have a list on the top")
        exit(1)

    try:
        return [FuncCall.model_validate(item) for item in config]
    except ValidationError:
        print("Error: The input filefunction_calling_tests.json "
              "is in a wrong format!")
        exit(1)


def read_input_files(file1: str, file2: str) -> Tuple[List[FuncDef],
                                                      List[FuncCall]]:
    """Parse both JSON input files and return their function
    definitions and prompts."""

    return parse_funcdef(file1), parse_funccall(file2)


def get_tokens_vocabulary() -> None:
    """Load the tokenizer vocabulary JSON and write it to data/vocab.json."""
    with open(f"{os.environ['HOME']}/sgoinfre/hf-cache/hub"
              "/models--Qwen--Qwen3-0.6B/snapshots/c1899de2"
              "89a04d12100db370d81485cdf75e47ca/vocab.json") as tokens:
        output = json.load(tokens)
        with open("data/vocab.json", "w") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
