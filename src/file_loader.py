import json
from src.models import FunctionDef, Prompt
from pydantic import ValidationError


def load_function_definitions(path: str) -> list[FunctionDef]:
    """ Load and validate function definitions from a JSON file """
    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: file not found: {path}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON in {path} : {e}")
        return []
    if not isinstance(data, list):
        print(f"Error: expected a list in {path}")
        return []
    result: list[FunctionDef] = []
    for item in data:
        try:
            result.append(FunctionDef.model_validate(item))
        except ValidationError as e:
            print(f"Error: skipping invalid entry {e}")
    return result


def load_prompts(path: str) -> list[Prompt]:
    """ Load and validate prompts from a JSON file. """

    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: file not found: {path}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON in {path}: {e}")
        return []
    if not isinstance(data, list):
        print(f"Error: expected a list in {path}")
        return []
    result: list[Prompt] = []
    for item in data:
        try:
            result.append(Prompt.model_validate(item))
        except ValidationError as e:
            print(f"Error: skipping invalid entry {e}")
    return result
