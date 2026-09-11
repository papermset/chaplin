import argparse
import csv
from collections import Counter
from pathlib import Path
import json

from benchmark.metrics import counts, rates
from benchmark.recorder import atomic_json
from benchmark.schema import MODES, validate_label


def evaluate(directory):
    rows, excluded, statuses = [], [], Counter()
    for path in sorted(Path(directory).rglob("sample.json")):
        sample = json.loads(path.read_text(encoding="utf-8"))
        validate_label(sample)
        statuses[sample.get("status", "unknown")] += 1
        reason = None
        if sample.get("ground_truth") is None:
            reason = "unlabelled"
        elif not isinstance(sample.get("raw_vsr"), str) or not isinstance(
            sample.get("llm_output"), str
        ):
            reason = "incomplete_outputs"
        if reason:
            excluded.append({"path": str(path.relative_to(directory)), "reason": reason})
            continue
        row = {
            "sample_id": sample["sample_id"],
            "mode": sample["mode"],
            "speaker_id": sample["speaker_id"],
            "ground_truth": sample["ground_truth"],
            "raw_vsr": sample["raw_vsr"],
            "llm_output": sample["llm_output"],
            "rerank_status": sample.get("rerank", {}).get("status", "unknown"),
        }
        for kind, key in (("raw", "raw_vsr"), ("llm", "llm_output")):
            row.update(
                {
                    f"{kind}_{k}": v
                    for k, v in rates(counts(sample["ground_truth"], sample[key])).items()
                }
            )
        rows.append(row)
    groups = {}
    for group in ("all", *MODES):
        members = [row for row in rows if group == "all" or row["mode"] == group]
        metrics = {
            "samples": len(members),
            "rerank_status_counts": dict(Counter(r["rerank_status"] for r in members)),
        }
        for kind in ("raw", "llm"):
            metrics[kind] = rates(
                {
                    key: sum(row[f"{kind}_{key}"] for row in members)
                    for key in ("word_errors", "reference_words", "char_errors", "reference_chars")
                }
            )
        groups[group] = metrics
    return {
        "schema_version": 1,
        "normalization": "NFKC, casefold, remove Unicode punctuation, collapse whitespace; CER excludes spaces",
        "aggregation": "micro (sum edits / sum reference units); empty denominators are null",
        "evaluated_samples": len(rows),
        "excluded_samples": excluded,
        "status_counts": dict(statuses),
        "groups": groups,
    }, rows


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate saved paired VSR/LLM outputs, without loading models."
    )
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if not args.directory.is_dir():
        parser.error("directory does not exist")
    summary, rows = evaluate(args.directory)
    output = args.output_dir or args.directory / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "metrics.json", summary)
    with (output / "samples.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = (
            list(rows[0])
            if rows
            else ["sample_id", "mode", "speaker_id", "ground_truth", "raw_vsr", "llm_output"]
        )
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)
    with (output / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            [
                "group",
                "samples",
                "output",
                "word_errors",
                "reference_words",
                "char_errors",
                "reference_chars",
                "wer",
                "cer",
            ],
        )
        writer.writeheader()
        for group, metrics in summary["groups"].items():
            for kind in ("raw", "llm"):
                writer.writerow(
                    {"group": group, "samples": metrics["samples"], "output": kind, **metrics[kind]}
                )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
