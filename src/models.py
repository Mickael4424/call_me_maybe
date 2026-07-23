from typing import Any
from pydantic import BaseModel


class ParameterDef(BaseModel):
    """ Describe the type of a single function parameter or return value """

    type: str


class FunctionDef(BaseModel):
    """ Describes the function that the LLM should return. """
    name: str
    description: str
    parameters: dict[str, ParameterDef]
    returns: ParameterDef


class Prompt(BaseModel):
    """ A single natural-language request. """
    prompt: str


class FunctionCall(BaseModel):
    """ The result of resolving a prompt """
    prompt: str
    name: str
    parameters: dict[str, Any]
