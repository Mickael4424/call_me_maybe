from enum import Enum, auto
from src.vocab_utils import VocabHelper
from src.models import FunctionDef
from typing import Optional


class State(Enum):
    """State is used to establish the correct step of the output generation."""

    EXPECT_OPEN_BRACE = auto()
    EXPECT_NAME_KEY = auto()
    EXPECT_COLON_AFTER_NAME = auto()
    EXPECT_FUNC_VALUE = auto()
    EXPECT_COMMA_AFTER_NAME = auto()
    EXPECT_PARAMETERS_KEY = auto()
    EXPECT_COLON_AFTER_PARAMS = auto()
    EXPECT_PARAM_OPEN_BRACE = auto()
    EXPECT_PARAM_KEY = auto()
    EXPECT_COLON_AFTER_PARAM = auto()
    EXPECT_PARAM_VALUE = auto()
    EXPECT_STRING_CONTENT = auto()
    EXPECT_STRING_CLOSE = auto()
    EXPECT_COMMA_OR_CLOSE = auto()
    EXPECT_CLOSE_BRACE = auto()
    DONE = auto()


class JSONAutomaton:
    """Finite state machine guiding JSON generation token by token. """

    def __init__(self, vocab: VocabHelper,
                 func_def: FunctionDef,
                 number_candidates: list[str] | None = None,
                 string_candidates: list[str] | None = None,
                 replacement_override: str | None = None,
                 source_override: str | None = None
                 ) -> None:
        """Initialise the automaton for a specific function. """
        self.vocab = vocab
        self.func_def = func_def
        self.state = State.EXPECT_OPEN_BRACE
        self.current_params: Optional[str] = None
        self.params_done: list[str] = []
        self.generated_so_far: str = ""
        self.string_content: str = ""
        self.string_token_count: int = 0
        self.max_string_tokens: int = 100
        self.number_candidates: list[str] = number_candidates or []
        self.current_number: str = ""
        self.string_candidates: list[str] = string_candidates or []
        self.current_string: str = ""
        self.replacement_override = replacement_override
        self.replacement_generated = False
        self.source_override = source_override
        self.source_generated = False

    def _tokens_continuing(self, expected: str) -> list[int]:
        """Return token IDs that continue generating the expected string. """
        remaining = expected[len(self.generated_so_far):]
        valid: list[int] = []
        for tid, tstr in self.vocab.vocab.items():
            if remaining.startswith(tstr):
                valid.append(tid)
        return valid

    def _is_complete(self, expected: str) -> bool:
        """Check if the expected string has been fully generated. """
        return self.generated_so_far == expected

    def _tokens_continuing_str(self, expected: str,
                               generated: str) -> list[int]:
        """Return tokens continuing expected from already generated string. """
        remaining = expected[len(generated):]
        valid: list[int] = []
        for tid, tstr in self.vocab.vocab.items():
            if remaining.startswith(tstr):
                valid.append(tid)
        return valid

    def get_valid_token_ids(self) -> list[int]:
        """Return valid token IDs for the current automaton state. """
        if self.state == State.EXPECT_OPEN_BRACE:
            return self.vocab.tokens_equal_to("{")

        if self.state == State.EXPECT_NAME_KEY:
            return self._tokens_continuing('"name"')

        if self.state == State.EXPECT_COLON_AFTER_NAME:
            return self.vocab.tokens_equal_to(":")

        if self.state == State.EXPECT_FUNC_VALUE:
            expected = f'"{self.func_def.name}"'
            return self._tokens_continuing(expected)

        if self.state == State.EXPECT_COMMA_AFTER_NAME:
            return self.vocab.tokens_equal_to(",")

        if self.state == State.EXPECT_PARAMETERS_KEY:
            return self._tokens_continuing('"parameters"')

        if self.state == State.EXPECT_COLON_AFTER_PARAMS:
            return self.vocab.tokens_equal_to(":")

        if self.state == State.EXPECT_PARAM_OPEN_BRACE:
            return self.vocab.tokens_equal_to("{")

        if self.state == State.EXPECT_PARAM_KEY:
            remaining = [
                p for p in self.func_def.parameters
                if p not in self.params_done
            ]
            if self.generated_so_far:
                matching = [
                    p for p in remaining
                    if f'"{p}"'.startswith(self.generated_so_far)
                ]
                if len(matching) == 1:
                    return self._tokens_continuing(f'"{matching[0]}"')
                elif len(matching) > 1:
                    valid: list[int] = []
                    for param_name in matching:
                        valid += self._tokens_continuing(f'"{param_name}"')
                    return valid
            valid = []
            for param_name in remaining:
                valid += self._tokens_continuing(f'"{param_name}"')
            return valid

        if self.state == State.EXPECT_COLON_AFTER_PARAM:
            return self.vocab.tokens_equal_to(":")

        if self.state == State.EXPECT_PARAM_VALUE:
            if self.current_params is None:
                return []
            param_type = self.func_def.parameters[self.current_params].type
            if param_type == "number":
                if self.number_candidates:
                    valid = []
                    for tid, tstr in self.vocab.vocab.items():
                        test = self.current_number + tstr
                        if any(c.startswith(test)
                               for c in self.number_candidates):
                            valid.append(tid)
                    return valid
                return self.vocab.tokens_that_are_numbers()
            if param_type == "string":
                return self.vocab.tokens_equal_to('"')
            if param_type == "boolean":
                return (self.vocab.tokens_equal_to("true")
                        + self.vocab.tokens_equal_to("false"))
            return []

        if self.state == State.EXPECT_STRING_CONTENT:
            param_name = self.current_params or ""
            is_regex = "regex" in param_name.lower()
            is_replacement = "replacement" in param_name.lower()
            is_source = "source_string" in param_name.lower()

            if (is_source and self.source_override
                    and not self.source_generated):
                expected = f'"{self.source_override}"'
                return self._tokens_continuing_str(
                    expected, self.current_string
                )

            if (is_replacement and self.replacement_override
                    and not self.replacement_generated):
                expected = f'"{self.replacement_override}"'
                return self._tokens_continuing_str(
                    expected, self.current_string
                )

            if self.string_candidates and not is_regex:
                valid = []
                for tid, tstr in self.vocab.vocab.items():
                    test = self.current_string + tstr
                    if any(c.startswith(test)
                           for c in self.string_candidates):
                        valid.append(tid)
                return valid

            max_tokens = 20 if is_regex else self.max_string_tokens
            if self.string_token_count >= max_tokens:
                return self.vocab.tokens_equal_to('"')

            closing = self.vocab.tokens_equal_to('"')
            return closing + [
                tid for tid, tstr in self.vocab.vocab.items()
                if '"' not in tstr
                and '\n' not in tstr
                and '{' not in tstr
                and '}' not in tstr
                and (not is_regex or 'Ġ' not in tstr)
                and len(tstr) <= 10
                and tstr.strip()
            ]

        if self.state == State.EXPECT_STRING_CLOSE:
            return self.vocab.tokens_equal_to('"')

        if self.state == State.EXPECT_COMMA_OR_CLOSE:
            remaining = [
                p for p in self.func_def.parameters
                if p not in self.params_done
            ]
            if remaining:
                return self.vocab.tokens_equal_to(",")
            else:
                return self.vocab.tokens_equal_to("}")

        if self.state == State.EXPECT_CLOSE_BRACE:
            return self.vocab.tokens_equal_to("}")

        return []

    def update_state(self, token_str: str) -> None:
        """Advance the automaton state based on the generated token. """
        if self.state == State.EXPECT_OPEN_BRACE:
            if "{" in token_str:
                self.state = State.EXPECT_NAME_KEY

        elif self.state == State.EXPECT_NAME_KEY:
            self.generated_so_far += token_str
            if self._is_complete('"name"'):
                self.generated_so_far = ""
                self.state = State.EXPECT_COLON_AFTER_NAME

        elif self.state == State.EXPECT_COLON_AFTER_NAME:
            if ":" in token_str:
                self.state = State.EXPECT_FUNC_VALUE

        elif self.state == State.EXPECT_FUNC_VALUE:
            self.generated_so_far += token_str
            expected = f'"{self.func_def.name}"'
            if self._is_complete(expected):
                self.generated_so_far = ""
                self.state = State.EXPECT_COMMA_AFTER_NAME

        elif self.state == State.EXPECT_COMMA_AFTER_NAME:
            if "," in token_str:
                self.state = State.EXPECT_PARAMETERS_KEY

        elif self.state == State.EXPECT_PARAMETERS_KEY:
            self.generated_so_far += token_str
            if self._is_complete('"parameters"'):
                self.generated_so_far = ""
                self.state = State.EXPECT_COLON_AFTER_PARAMS

        elif self.state == State.EXPECT_COLON_AFTER_PARAMS:
            if ":" in token_str:
                self.state = State.EXPECT_PARAM_OPEN_BRACE

        elif self.state == State.EXPECT_PARAM_OPEN_BRACE:
            if "{" in token_str:
                self.state = State.EXPECT_PARAM_KEY

        elif self.state == State.EXPECT_PARAM_KEY:
            self.generated_so_far += token_str
            for param_name in self.func_def.parameters:
                if param_name not in self.params_done:
                    expected = f'"{param_name}"'
                    if self.generated_so_far == expected:
                        self.current_params = param_name
                        self.generated_so_far = ""
                        self.state = State.EXPECT_COLON_AFTER_PARAM
                        break

        elif self.state == State.EXPECT_COLON_AFTER_PARAM:
            if ":" in token_str:
                self.state = State.EXPECT_PARAM_VALUE

        elif self.state == State.EXPECT_PARAM_VALUE:
            if self.current_params:
                param_type = self.func_def.parameters[
                    self.current_params
                ].type
                if param_type == "string":
                    self.current_string = '"'
                    self.state = State.EXPECT_STRING_CONTENT
                    return
                if param_type == "number" and self.number_candidates:
                    self.current_number += token_str
                    if self.current_number in self.number_candidates:
                        self.number_candidates.remove(self.current_number)
                        self.current_number = ""
                        self.params_done.append(self.current_params)
                        self.current_params = None
                        self.state = State.EXPECT_COMMA_OR_CLOSE
                    return
                self.params_done.append(self.current_params)
                self.current_params = None
            self.state = State.EXPECT_COMMA_OR_CLOSE

        elif self.state == State.EXPECT_STRING_CONTENT:
            param_name = self.current_params or ""
            is_replacement = "replacement" in param_name.lower()
            is_source = "source_string" in param_name.lower()

            if (is_source and self.source_override
                    and not self.source_generated):
                self.current_string += token_str
                expected = f'"{self.source_override}"'
                if self.current_string == expected:
                    self.source_generated = True
                    self.current_string = ""
                    if self.current_params:
                        self.params_done.append(self.current_params)
                        self.current_params = None
                    self.state = State.EXPECT_COMMA_OR_CLOSE
                return

            if (is_replacement and self.replacement_override
                    and not self.replacement_generated):
                self.current_string += token_str
                expected = f'"{self.replacement_override}"'
                if self.current_string == expected:
                    self.replacement_generated = True
                    self.current_string = ""
                    if self.current_params:
                        self.params_done.append(self.current_params)
                        self.current_params = None
                    self.state = State.EXPECT_COMMA_OR_CLOSE
                return

            if self.string_candidates:
                self.current_string += token_str
                if self.current_string in self.string_candidates:
                    self.current_string = ""
                    if self.current_params:
                        self.params_done.append(self.current_params)
                        self.current_params = None
                    self.state = State.EXPECT_COMMA_OR_CLOSE
                return

            if token_str == '"':
                self.string_token_count = 0
                self.current_string = ""
                if self.current_params:
                    self.params_done.append(self.current_params)
                    self.current_params = None
                self.state = State.EXPECT_COMMA_OR_CLOSE
            else:
                self.current_string += token_str
                self.string_token_count += 1

        elif self.state == State.EXPECT_STRING_CLOSE:
            if self.current_params:
                self.params_done.append(self.current_params)
                self.current_params = None
            self.state = State.EXPECT_COMMA_OR_CLOSE

        elif self.state == State.EXPECT_COMMA_OR_CLOSE:
            remaining = [
                p for p in self.func_def.parameters
                if p not in self.params_done
            ]
            if "}" in token_str:
                self.state = State.EXPECT_CLOSE_BRACE
            elif "," in token_str and remaining:
                self.generated_so_far = ""
                self.current_string = ""
                self.state = State.EXPECT_PARAM_KEY

        elif self.state == State.EXPECT_CLOSE_BRACE:
            if "}" in token_str:
                self.state = State.DONE
