from pydantic import BaseModel, ValidationError, ConfigDict, field_validator
from typing import List, Literal, Dict, Tuple
import json
import os
import argparse
from argparse import Namespace


def load_json_no_duplicates(path: str) -> dict[str, object]:
    """Loads file and verify if a file be in a valid json
      and if json file has duplicates"""
    def detect_duplicates(pairs:
                          list[tuple[str, object]]) -> dict[str, object]:
        seen = {}
        for key, value in pairs:
            if key in seen:
                raise ValueError(f"Error: Duplicate "
                                 f"key '{key}' in JSON file {path}")
            seen[key] = value
        return seen

    with open(path) as f:
        raw = f.read()
    try:
        result: dict[str, object] = json.loads(
                    raw,
                    object_pairs_hook=detect_duplicates
                )
        return result
    except ValueError as e:
        print(e)
        exit(1)


class FuncCall(BaseModel):
    """Represents a parsed function‑calling prompt from the JSON input file."""
    prompt: str
    model_config = ConfigDict(extra="forbid")

    @field_validator("prompt")
    def parse_empty_string(cls, v: str) -> str:
        if not v.strip():
            print("Error: Prompt cannot be empty")
            exit(1)
        return v


class VariableType(BaseModel):
    """Represents a parsed variable type from the JSON
    function definition file."""
    type: Literal["number", "string", "boolean", "integer"]
    model_config = ConfigDict(extra="forbid")


class FuncDef(BaseModel):
    """Represents a parsed function definition
    from the JSON input file."""
    name: str
    description: str
    parameters: Dict[str, VariableType]
    returns: VariableType
    model_config = ConfigDict(extra="forbid")

    @field_validator("name", "description")
    def parse_empty_string(cls, v: str) -> str:
        if not v.strip():
            print("Error: Prompt cannot be empty")
            exit(1)
        return v

    @field_validator("parameters")
    def parse_empty_parameters(cls, p: dict[str, str]) -> dict[str, str]:
        for k, _ in p.items():
            if not k.strip():
                print("Error: Parameter prompt cannot be empty")
                exit(1)
        return p


def parse_args() -> Namespace:
    """Get arguments from terminal"""
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
        config = load_json_no_duplicates(path)
    except json.JSONDecodeError as e:
        print("Error in the json file:", e)
        exit(1)
    except ValueError as e:
        print(e)
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
        config = load_json_no_duplicates(path)
    except json.JSONDecodeError as e:
        print("Error in the json file:", e)
        exit(1)
    except ValueError as e:
        print(e)
        exit(1)

    if not isinstance(config, list):
        print("Error: The file must have a list on the top")
        exit(1)

    try:
        return [FuncCall.model_validate(item) for item in config]
    except ValidationError:
        print("Error: The input file function_calling_tests.json "
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
