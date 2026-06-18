from pydantic import BaseModel, ValidationError, ConfigDict
from typing import List, Literal, Dict, Union
import json
import os


class PromptItem(BaseModel):
    prompt: str
    model_config = ConfigDict(extra="forbid")


class VariableType(BaseModel):
    type: Literal["number", "string", "boolean"]
    model_config = ConfigDict(extra="forbid")


class FuncFormat(BaseModel):
    name: str
    description: str
    parameters: Dict[str, VariableType]
    returns: VariableType
    model_config = ConfigDict(extra="forbid")


def parse_prompt(path: str) -> Union[List[PromptItem], List[FuncFormat]]:
    try:
        with open(path) as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        print(e)
        exit(1)

    if not isinstance(config, list):
        print("Error: The file must have a list on the top")
        exit(1)

    try:
        return [PromptItem.model_validate(item) for item in config]
    except ValidationError:
        pass

    try:
        return [FuncFormat.model_validate(item) for item in config]
    except ValidationError:
        pass

    print("Error: The file does not match any of the supported formats.")
    exit(1)


def get_all_possible_tokens():
    with open(f"{os.environ['HOME']}/sgoinfre/hf-cache/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca/vocab.json") as tokens:
        output = json.load(tokens)
        with open("data/vocab.json", "w") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
