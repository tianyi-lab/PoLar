#!/usr/bin/env python3
"""Merge sharded MCTS JSON outputs into PoLar `merged_mcts_samples.json`.

PoLar expects (see https://github.com/tianyi-lab/PoLar):

  {data_root}/{model_path}/dart-math-diff-{N}/merged_mcts_samples.json

Each sample needs at least:
  - question / gt_ans (or sample_info.* aliases)
  - final_valid_transitions: list of layer-index paths
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge MCTS shard JSONs into PoLar supervision format."
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Directory containing shard files like 0_5.json, 5_10.json, ...",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("merged_mcts_samples.json"),
        help="Output path (relative paths are resolved under directory).",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=None,
        help="If set, fail when merged sample count != this value.",
    )
    parser.add_argument(
        "--tsv",
        type=Path,
        default=None,
        help="Optional TSV dump of sample_id / global_idx / query_id.",
    )
    return parser.parse_args()


def sort_key(path: Path) -> Tuple[int, str]:
    stem = path.stem
    try:
        prefix = stem.split("_", 1)[0]
        return int(prefix), stem
    except ValueError:
        return sys.maxsize, stem


def _as_path_list(items: Any) -> List[List[int]]:
    """Normalize transitions to list[list[int]]."""
    if not items:
        return []
    out: List[List[int]] = []
    for item in items:
        if isinstance(item, dict) and "path" in item:
            out.append(list(item["path"]))
        elif isinstance(item, (list, tuple)):
            out.append(list(item))
    return out


def normalize_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    sample_info = sample.get("sample_info") or {}
    question = (
        sample.get("question")
        or sample_info.get("question")
        or sample_info.get("query")
    )
    gt_ans = (
        sample.get("gt_ans")
        or sample.get("ground_truth")
        or sample_info.get("ground_truth")
        or sample_info.get("answer")
    )
    initial_score = sample.get(
        "initial_score", sample.get("initial_transition_metric")
    )

    final_valid = sample.get("final_valid_transitions")
    if final_valid is None:
        final_valid = sample.get("valid_transitions")
    final_invalid = sample.get("final_invalid_transitions")
    if final_invalid is None:
        final_invalid = sample.get("invalid_transitions")

    normalized = {
        "sample_id": sample.get("sample_id"),
        "global_idx": sample.get("global_idx"),
        "query_id": sample.get("query_id"),
        "question": question,
        "gt_ans": gt_ans,
        "initial_score": initial_score,
        "final_valid_transitions": _as_path_list(final_valid),
        "final_invalid_transitions": _as_path_list(final_invalid),
        "sample_info": {
            "question": question,
            "ground_truth": gt_ans,
            "prompt_text": sample_info.get("prompt_text", ""),
        },
    }
    return normalized


def main() -> None:
    args = parse_args()
    directory = args.directory
    if not directory.is_dir():
        raise SystemExit(f"[ERROR] Directory not found: {directory}")

    # Skip previously merged outputs if present in the same folder.
    skip_names = {"merged_mcts_samples.json", "merged_samples.json"}
    json_paths = sorted(
        (
            p
            for p in directory.glob("*.json")
            if p.name not in skip_names and "_" in p.stem
        ),
        key=sort_key,
    )
    if not json_paths:
        raise SystemExit(f"[ERROR] No shard JSON files found in {directory}")

    merged_header: Dict[str, Any] = {}
    merged_samples: List[Dict[str, Any]] = []

    for idx, path in enumerate(json_paths):
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        samples = data.get("samples")
        if samples is None:
            raise SystemExit(f"[ERROR] File missing 'samples': {path}")

        if idx == 0:
            merged_header = {k: v for k, v in data.items() if k != "samples"}
        else:
            for key, value in merged_header.items():
                if data.get(key) != value:
                    print(
                        f"[WARN] header field '{key}' differs in {path.name}; "
                        "keeping first-file value"
                    )

        for sample in samples:
            merged_samples.append(normalize_sample(sample))
        print(f"[INFO] Merged {len(samples):>4} samples from {path.name}")

    if args.expected_count is not None and len(merged_samples) != args.expected_count:
        raise SystemExit(
            f"[ERROR] Sample count {len(merged_samples)} != "
            f"expected {args.expected_count}"
        )

    merged_data = dict(merged_header)
    merged_data["samples"] = merged_samples

    output_path = (
        args.output if args.output.is_absolute() else directory / args.output
    )
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(merged_data, handle, ensure_ascii=False, indent=2)

    print(f"[INFO] Wrote merged JSON to {output_path}")
    print(f"[INFO] Total samples: {len(merged_samples)}")

    if args.tsv:
        tsv_path = args.tsv if args.tsv.is_absolute() else directory / args.tsv
        with tsv_path.open("w", encoding="utf-8") as handle:
            handle.write("sample_id\tglobal_idx\tquery_id\n")
            for sample in merged_samples:
                handle.write(
                    f"{sample.get('sample_id')}\t{sample.get('global_idx')}\t"
                    f"{sample.get('query_id')}\n"
                )
        print(f"[INFO] Wrote TSV to {tsv_path}")


if __name__ == "__main__":
    main()
