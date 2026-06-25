#!/usr/bin/env python3
"""Convert Magpie parquet datasets to speculators-compatible JSONL.

Example (full conversion):
    python scripts/convert_magpie_for_speculators.py \
        --input /path/to/Magpie-Qwen2.5-Pro-1M-v0.1/data \
        --output ./data/magpie_qwen25_pro.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator

from datasets import load_dataset
from tqdm import tqdm


def _to_conversations(example: dict[str, Any]) -> list[dict[str, str]] | None:
    conversations = example.get("conversations")
    if not conversations:
        instruction = example.get("instruction")
        response = example.get("response")
        if not instruction or not response:
            return None
        conversations = [
            {"from": "human", "value": instruction},
            {"from": "gpt", "value": response},
        ]

    normalized: list[dict[str, str]] = []
    for turn in conversations:
        role = turn.get("from") or turn.get("role") or ""
        content = turn.get("value") or turn.get("content") or ""
        if not content:
            continue
        if role in ("human", "user"):
            role = "user"
        elif role in ("gpt", "assistant"):
            role = "assistant"
        elif role != "system":
            continue
        normalized.append({"role": role, "content": content})

    if len(normalized) < 2 or normalized[-1]["role"] != "assistant":
        return None
    return normalized


def _parquet_paths(input_path: str) -> list[Path]:
    path = Path(input_path)
    if path.is_dir():
        files = sorted(path.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files found under {path}")
        return files
    if path.suffix == ".parquet":
        return [path]
    raise ValueError(
        f"Unsupported input: {input_path}. "
        "Use a parquet directory, a .parquet file, or --hf-dataset."
    )


def _iter_examples(input_path: str, *, hf_dataset: str | None) -> Iterator[dict[str, Any]]:
    if hf_dataset:
        dataset = load_dataset(hf_dataset, split="train")
        yield from tqdm(dataset, desc="Converting", unit="sample")
        return

    parquet_files = _parquet_paths(input_path)
    for parquet_file in tqdm(parquet_files, desc="Shards", unit="file"):
        shard = load_dataset("parquet", data_files=str(parquet_file), split="train")
        yield from tqdm(
            shard,
            desc=parquet_file.name,
            leave=False,
            unit="sample",
        )


def convert_magpie(
    input_path: str,
    output_path: str,
    *,
    hf_dataset: str | None = None,
    max_samples: int | None = None,
) -> int:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with output.open("w", encoding="utf-8") as f:
        for example in _iter_examples(input_path, hf_dataset=hf_dataset):
            conversations = _to_conversations(example)
            if conversations is None:
                continue

            f.write(json.dumps({"conversations": conversations}, ensure_ascii=False))
            f.write("\n")
            written += 1

            if max_samples is not None and written >= max_samples:
                break

    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Magpie parquet to speculators JSONL"
    )
    parser.add_argument(
        "--input",
        help="Magpie parquet directory or single .parquet file",
    )
    parser.add_argument(
        "--hf-dataset",
        help="HuggingFace dataset id (alternative to --input)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL path for prepare_data.py",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Stop after N samples (default: convert all)",
    )
    args = parser.parse_args()
    if not args.input and not args.hf_dataset:
        parser.error("Provide --input or --hf-dataset")
    if args.input and args.hf_dataset:
        parser.error("Use only one of --input or --hf-dataset")
    return args


def main() -> None:
    args = parse_args()
    written = convert_magpie(
        args.input or "",
        args.output,
        hf_dataset=args.hf_dataset,
        max_samples=args.max_samples,
    )
    print(f"Done. Wrote {written:,} samples to {args.output}")


if __name__ == "__main__":
    main()
