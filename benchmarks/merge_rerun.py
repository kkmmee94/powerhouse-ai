from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("original", type=Path)
    parser.add_argument("rerun", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    original = load(args.original)
    replacements = {str(record["id"]): record for record in load(args.rerun)}
    original_ids = {str(record["id"]) for record in original}
    unknown = sorted(set(replacements) - original_ids)
    if unknown:
        raise ValueError(f"Rerun contains IDs absent from the original: {unknown}")

    merged = [replacements.get(str(record["id"]), record) for record in original]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as destination:
        for record in merged:
            destination.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "original_cases": len(original),
                "replaced_cases": len(replacements),
                "output_cases": len(merged),
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
