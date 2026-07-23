import json
from pathlib import Path
from src.models import FunctionCall


def write_results(results: list[FunctionCall], output_path: str) -> None:
    """ Write function call results to a JSON output file. """
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        data = [r.model_dump() for r in results]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Results written to {output_path}")
    except OSError as e:
        print(f"Error writing output: {e}")
