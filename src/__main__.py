import argparse

from llm_sdk import Small_LLM_Model
from src.constrained_generator import ConstrainedGenerator
from src.file_loader import load_function_definitions, load_prompts
from src.file_writer import write_results
from src.function_selector import FunctionSelector
from src.models import FunctionCall

from typing import Any

DEFAULT_FUNCTIONS = "data/input/functions_definition.json"
DEFAULT_INPUT = "data/input/function_calling_tests.json"
DEFAULT_OUTPUT = "data/output/function_calling_results.json"


def parse_args() -> argparse.Namespace:
    """ Parse command-line arguments """
    parser = argparse.ArgumentParser(description="call_me_maybe")
    parser.add_argument('--functions_definition', default=DEFAULT_FUNCTIONS)
    parser.add_argument('--input', default=DEFAULT_INPUT)
    parser.add_argument('--output', default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    """ Entry point - orchestrates the full function calling pipeline. """

    # arguments checking
    args = parse_args()

    # functions and prompts loading : valid file, valid json, valid list
    functions = load_function_definitions(args.functions_definition)
    prompts = load_prompts(args.input)

    if not functions or not prompts:
        return

    print(f"Loaded {len(functions)} functions")
    print(f"Loaded {len(prompts)} prompts")

    print("Loading real LLM")
    model: Any = Small_LLM_Model()
    print(f"Model ready: {type(model).__name__}")

    selector = FunctionSelector(model)
    generator = ConstrainedGenerator(model)
    results: list[FunctionCall] = []

    for i, prompt_obj in enumerate(prompts):
        print(f"\n[{i+1}]/[{len(prompts)}] {prompt_obj.prompt}")
        try:
            chosen_fn = selector.select(prompt_obj.prompt, functions)
            print(f" -> selected: {chosen_fn}")

            call = generator.generate(prompt_obj.prompt, chosen_fn)
            print(f" -> result: {call}")
            results.append(call)

        except Exception as e:
            print(f" -> Error: {e}")

    write_results(results, args.output)


if __name__ == "__main__":
    main()
