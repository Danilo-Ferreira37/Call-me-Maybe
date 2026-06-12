from pydantic import BaseModel, Field
import json

def json_format_parse(path: str):
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(e)
        exit(1)

class PromptKey():
    pass