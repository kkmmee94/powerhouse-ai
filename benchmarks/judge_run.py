from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from powerhouse_ai.config import get_settings
from powerhouse_ai.errors import ModelResponseError
from powerhouse_ai.llm import LLMClient

JUDGE_PROMPT = """
You are grading an AI answer against a task and a list of required points.
Return only JSON:
{
  "required_points": [{"point": "...", "satisfied": true, "reason": "..."}],
  "correctness": 0,
  "completeness": 0,
  "clarity": 0,
  "unsupported_claims": 0,
  "total": 0,
  "summary": "..."
}

Scales:
- correctness: 0 to 4
- completeness: 0 to 3
- clarity: 0 to 1
- unsupported_claims: count of material unsupported or invented claims
- total: correctness + completeness + clarity - min(2, unsupported_claims), clamped 0 to 8

Be strict. Code is not considered tested merely because the answer says it is tested.
Do not reward verbosity. Judge only the supplied answer.
""".strip()


def load(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def grade_record(
    llm: LLMClient,
    record: dict[str, Any],
    *,
    max_attempts: int = 3,
) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": JUDGE_PROMPT},
        {
            "role": "user",
            "content": (
                f"Task:\n{record['question']}\n\n"
                f"Required points:\n{json.dumps(record['expected_points'])}\n\n"
                f"Answer:\n{record['answer']}"
            ),
        },
    ]
    last_error: ModelResponseError | None = None
    for attempt in range(max_attempts):
        retry_messages = list(messages)
        if attempt:
            retry_messages.append(
                {
                    "role": "user",
                    "content": (
                        "The previous grading response was invalid JSON. Retry with one compact "
                        "JSON object only. Keep every reason and the summary short, and do not put "
                        "unescaped quotation marks or backslashes inside JSON strings."
                    ),
                }
            )
        try:
            return llm.chat_json(
                retry_messages,
                temperature=0,
                max_tokens=1600,
            )
        except ModelResponseError as exc:
            last_error = exc
            print(f"{record['id']}: malformed grade, retry {attempt + 1}/{max_attempts}")
    raise ModelResponseError(
        f"Judge returned malformed JSON for {record['id']} after {max_attempts} attempts."
    ) from last_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.run.with_name(f"{args.run.stem}.scored.jsonl")
    settings = get_settings()
    llm = LLMClient(settings)
    run_records = load(args.run)
    existing = load(output) if output.exists() else []
    source_fingerprints = {
        record.get("run_metadata", {}).get("source_sha256")
        for record in [*run_records, *existing]
        if record.get("run_metadata", {}).get("source_sha256")
    }
    if len(source_fingerprints) > 1:
        raise ValueError(
            "Refusing to combine benchmark records from different source fingerprints."
        )
    completed_ids = {str(record["id"]) for record in existing}
    scores = [float(record["grade"]["total"]) for record in existing]
    pending = [record for record in run_records if str(record["id"]) not in completed_ids]
    file_mode = "a" if existing else "w"

    if existing:
        print(f"Resuming {output}: {len(existing)} already scored, {len(pending)} remaining.")

    with output.open(file_mode, encoding="utf-8") as destination:
        for record in pending:
            grade = grade_record(llm, record)
            correctness = max(0.0, min(4.0, float(grade.get("correctness", 0))))
            completeness = max(0.0, min(3.0, float(grade.get("completeness", 0))))
            clarity = max(0.0, min(1.0, float(grade.get("clarity", 0))))
            unsupported = max(0, int(grade.get("unsupported_claims", 0)))
            score = max(
                0.0,
                min(8.0, correctness + completeness + clarity - min(2, unsupported)),
            )
            grade["total"] = score
            scores.append(score)
            destination.write(json.dumps({**record, "grade": grade}, ensure_ascii=False) + "\n")
            print(f"{record['id']}: {score}/8")

    print(
        json.dumps(
            {
                "cases": len(scores),
                "mean_score": round(statistics.fmean(scores), 3),
                "score_percent": round(statistics.fmean(scores) / 8 * 100, 2),
                "output": str(output),
            },
            indent=2,
        )
    )
    llm.close()


if __name__ == "__main__":
    main()
