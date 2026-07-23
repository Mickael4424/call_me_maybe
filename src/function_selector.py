from src.models import FunctionDef
from src.vocab_utils import VocabHelper, safe_encode
from typing import Any


class FunctionSelector:
    """ Select the most appopriate function for a given prompt. """

    def __init__(self, model: Any) -> None:
        """ Initialise the selector with a language model """

        self.model = model
        self.vocab = VocabHelper(model.get_path_to_vocab_file())

    def select(
            self,
            prompt: str,
            functions: list[FunctionDef],
    ) -> FunctionDef:
        """ Select the most appropriate function from a prompt. """

        system = "Available functions\n"
        for fn in functions:
            system += f"- {fn.name}: {fn.description}\n"
        system += f"\nUser request: {prompt}\n"
        system += "The most appropriate function for this request is fn"

        encoded = safe_encode(self.model, system)

        logits = self.model.get_logits_from_input_ids(encoded)

        valid_ids: list[int] = []
        fn_second_tokens: dict[int, FunctionDef] = {}

        for fn in functions:
            fn_ids = safe_encode(self.model, fn.name)
            if len(fn_ids) >= 2:
                second_token = fn_ids[1]
                valid_ids.append(second_token)
                fn_second_tokens[second_token] = fn
            valid_ids += self.vocab.tokens_starting_with("fn")

        for i in range(len(logits)):
            if i not in valid_ids:
                logits[i] = float("-inf")

        best_id = logits.index(max(logits))

        if best_id in fn_second_tokens:
            return fn_second_tokens[best_id]

        return functions[0]
