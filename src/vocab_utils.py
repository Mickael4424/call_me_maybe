import json
import re
from typing import Any, cast


class VocabHelper:
    """ Helper to find tokens IDS or tokens strings from the vocabulary. """

    def __init__(self, vocab_path: str) -> None:
        """ Initialise the helper by loading the vocabulary file"""

        with open(vocab_path, "r") as f:
            raw = json.load(f)

        first_key = next(iter(raw))
        if isinstance(first_key, str) and not first_key.isdigit():
            self.vocab: dict[int, str] = {
                int(v): k for k, v in raw.items()
            }
        else:
            self.vocab = {
                int(k): v for k, v in raw.items()
            }

    def tokens_equal_to(self, s: str) -> list[int]:
        """ Return token IDS whose string representation exactly matches s. """

        return [
            tid for tid, tstr in self.vocab.items()
            if tstr == s
        ]

    def tokens_starting_with(self, prefix: str) -> list[int]:
        """ Return token IDS whose string starts with the given prefix """

        return [
            tid for tid, tstr in self.vocab.items()
            if tstr.startswith(prefix)
        ]

    def tokens_that_are_numbers(self) -> list[int]:
        """ Return token IDS representing numeric digit strings. """

        result: list[int] = []
        for tid, tstr in self.vocab.items():
            stripped = tstr.strip()
            if stripped and stripped.isdigit():
                result.append(tid)
        return result

    def get_token_str(self, token_id: int) -> str:
        """ Return the string representation of a token ID """

        return self.vocab.get(token_id, "")


def safe_encode(model: Any, text: str) -> list[int]:
    """ Encode text to token IDS handling both tensor and list outputs. """
    encoded = model.encode(text)
    if hasattr(encoded, 'tolist'):
        return cast(list[int], encoded.tolist()[0])
    else:
        return cast(list[int], encoded)


def extract_numbers_from_prompt(text: str) -> list[str]:
    """Extract all numbers present in the prompt text."""
    return [str(n) for n in re.findall(r'-?\d+\.?\d*', text)]


def extract_string_candidates(prompt: str) -> list[str]:
    """ Extract all strings candidates
    from the prompt for constrained decoding. """

    quoted_double = re.findall(r'"(.+?)"', prompt)
    if quoted_double:
        return [f'"{s}"' for s in quoted_double]

    quoted_single = re.findall(r"'(.+?)'", prompt)
    if quoted_single:
        return [f'"{s}"' for s in quoted_single]
    stop_words = {
        "what", "is", "the", "a", "an", "of", "and",
        "to", "in", "for", "with", "greet", "reverse",
        "string", "calculate", "replace", "substitute",
        "all", "vowels", "numbers", "word", "text"
    }

    words = prompt.lower().split()
    return [
        f'"{w.strip(chr(39))}"'
        for w in words
        if w.strip("'") not in stop_words
        and len(w.strip("'")) > 1
    ]


SYMBOL_MAP = {
    "asterisks": "*",
    "asterisk": "*",
    "stars": "*",
    "star": "*",
    "hashes": "#",
    "hash": "#",
    "dashes": "-",
    "dash": "-",
    "underscores": "_",
    "underscore": "_",
    "dots": ".",
    "dot": ".",
}


def extract_replacement_from_prompt(prompt: str) -> str | None:
    """ Extract the replacement string from the prompt text. """

    with_upper = re.search(r'\bwith\s+([A-Z]+)\b', prompt)
    if with_upper:
        return with_upper.group(1)

    words = prompt.lower().split()
    for word in words:
        if word in SYMBOL_MAP:
            return SYMBOL_MAP[word]

    with_quoted = re.search(r"with\s+['\"](.+?)['\"]", prompt)
    if with_quoted:
        return with_quoted.group(1)

    return None
