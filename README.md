*This project has been created as part of the 42 curriculum by mbouyer.*

Call me maybe: Introduction to function calling with LLMs

## Description

Small language models are unreliable at generating structured output. When prompted to producte JSON, they might succeed only 30% of the time. Using constrained decoding with small language models helps achieve 99%+ reliability.

Instead of answering a request directly, the program selects the appropriate function name and extracts typed parameters from the prompt.

The project uses `llm_sdk` with the model `Qwen/Qwen3-0.6B`. Both function selection and argument generation are done with constrained decoding: at each generation step, only tokens that keep the output structurally and semantically valid are allowed.

Example output for the prompt "What is the sum of 2 and 3?":

```json
{
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": { "a": 2.0, "b": 3.0 }
}
```


## Instructions

### Installation

```bash
uv sync
```

or

``` bash
make install
```

### Running the program

```bash
uv run python3 -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calling_results.json
```

### Default paths

| Argument  | Default |
| --- | ---|
| `--functions_definition`  | `data/input/functions_definition.json`  |
| `--input` | `data/input/function_calling_tests.json`  |
| `--output`  | `data/output/function_calling_results.json` |

> Do not commit `data/output/`; it is generated when the program runs.


### Makefile targets

| Target  | Action  |
| --- | --- |
| `install` | Install dependencies with `uv sync` |
| `run` | Run the program |
| `debug` | Run the prgram with pdb (debug mode)  |
| `clean` | Remove Python caches  |
| `lint`  | Run flake8 and mypy |
| `lint-strict` | Run flake8 and mypy --strict |


## Resources

- [w3school](https://www.w3schools.com/) - practising Python exercises: classes, files handling, modules
- [Stephane Robert's blog](https://blog.stephane-robert.info) - AI basics, LLM introduction, prompt engineering, running LLMs locally
- [uv documentation](https://docs.astral.sh/uv/)
- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Andrej Karpathy - Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)

### AI usage

Claude (Anthropic) was used throughout this project as a learning and organizational aid:
- Breaking the project down into steps
- Explaining concepts(tokenization, constrained decoding, logits masking) before implementation
- Help me designing a mock model to develop and test offliner, since my local machine could not run the real LLM
- Reviewing code and identifying bugs(type errors, logic errors)

All code was written, understood and tested by me; AI was not used to generate unreviewed code.

## Additional sections

### Algorithm explanation

This project implements constrained decoding to guarantee 100% valid
JSON output from a small language model.

#### Function selection

The `FunctionSelector` builds a prompt listing all available
functions and asks the model to complete `"Function to call: fn"`.
Only the second tokens of each function name are allowed
(e.g. `_add` for `fn_add_numbers`, `_g` for `fn_greet`), making
each function uniquely identifiable in one generation step.

#### JSON generation

The `ConstrainedGenerator` drives token-by-token generation through
a finite state machine (`JSONAutomaton`). At each step:

1. The model produces logits for all 151,936 tokens
2. `get_valid_token_ids()` returns only the tokens valid at the
   current state
3. The highest-scoring valid token is selected
4. `update_state()` advances the automaton

The automaton enforces both structural validity (correct JSON
syntax) and schema compliance (correct parameter names and types).

#### Number and string extraction

To improve argument accuracy, numbers are extracted directly from
the prompt using regex (`extract_numbers_from_prompt`) and used as
the only valid candidates for numeric parameters. String values
are extracted from quoted phrases in the prompt
(`extract_string_candidates`).

### Design decisions

The project structure follows the subject's requirements:

- `data/`: test files provided with the subject (no `output/` committed)
- `llm_sdk/`: the LLM wrapper provided with the subject
- `src/`: my implementation
- `pyproject/` and `uv.lock` for dependency management
- `README.md` and `Makefile` as required

Additional files I added:

- `.flake8`: excludes `.venv`, cache files and `llm_sdk` from linting.
- `mypy.ini`: excludes `llm_sdk` from type checking (third-party code with its own type issues)

Inside `src/`, the implementation is split by responsibility:

- `models.py`: Pydantic models(`ParameterDef`, `FunctionDef`, `Prompt`, `FunctionCall`)
- `file_loader.py`: loading functions and validating functions/prompts from JSON file
- `file_writer.py`: writing results to the output JSON file
- `vocab_utils.py`: mapping between token IDS and their string representation (`VocabHelper`), plus `safe_encode` to normalize encoder output across the mock and real models
- `json_automaton.py`: the finite state machine (`JSONAutomaton`)
  that determines which tokens are valid at each generation step
- `function_selector.py`: selecting the right function for a prompt
  via constrained decoding
- `constrained_generator.py`: generating and parsing the final JSON
  function call



### Performance analysis

During development, the pipeline was first validated locally using `MockModel`, a stand-in with a minimal 19-token vocabulary and random logits. With the mock, the pipeline ran end-to-end without crashing, but function selection was unreliable (often defaulting to the same function) since the mock's logits carry no real signal - this was expected and confirmed the mock's only purpose was testing control flow, not output quality.

The real `Qwen/Qwen3-0.6B` model could not be loaded locally on a 2014 Mac Mini due to insufficient RAM for loading a 1.5GB model in float32. Testing with the real model was done on school machines.

Regarding the constrained decoding guarantee specifically: by
construction, every token generated is checked against
`get_valid_token_ids()` before being added to the output, so output
JSON is always syntactically valid and schema-compliant regardless of
model size — this was verified by running the automaton's logic
standalone (see Testing Strategy) before connecting it to either
model.

On school machines with `Qwen/Qwen3-0.6B`:
- Function selection: 10/11 prompts correct
- JSON validity: 100% — every output is parseable
- Argument extraction: numbers extracted correctly from prompt
  (e.g. 265 and 345 for "What is the sum of 265 and 345?"),
  strings extracted from quoted phrases
- Processing time: approximately 30-60 seconds per prompt on CPU

### Challenges faced

Understanding the project took time - especially grasping how different pieces fit together:

```
prompt  -> tokenization -> logits -> function selection
        -> contrained generation -> FunctionCall
```

The hardest part was designing the JSON automaton: correctly
separating states like "expecting a parameter key" from "expecting
the literal key `parameters`", and tracking `current_param` /
`params_done` to know which parameter is being generated and which
ones remain.

Working from home, I could not load the real LLM: my 2014 Mac Mini
does not have enough RAM to load the 1.5GB model. I built a
`MockModel` replicating the same interface (`encode`,
`get_logits_from_input_ids`, `get_path_to_vocab_file`) with a small
hand-written vocabulary, which let me develop and test the entire
pipeline — automaton, generator, selector — before ever touching the
real model on school machines.

### Testing strategy
1. Run make lint to check flake8 and mypy
2. Run make clean to remove all caches files
3. Run make debug to use the debug option
4. Run the program witout argument (default input and output selected)
5. Check that the ouput file is valid JSON
6. Check that the ouput file is correct: every object contains prompt, function name and parameters.
7. Test edges cases: missing files, invalid JSON, empty prompts, decimal numbers, etc.

### Example usage

#### Basic usage

```bash
make install
make run
```

#### Custom paths:

``` bash
uv run python3 -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calling_results.json
```

#### Console output

```
Loaded 5 functions
Loaded 11 prompts
Loading real LLM...
Model ready: Small_LLM_Model
[1/11] What is the sum of 2 and 3?
  -> selected: fn_add_numbers
  -> result: {"a": 2.0, "b": 3.0}
...
Results written to data/output/function_calling_results.json
```

#### Example output:

`data/output/function_calling_results.json`:
```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": { "a": 2.0, "b": 3.0 }
  }
]
```

