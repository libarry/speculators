#!/usr/bin/env python3
"""
Load local open-perfectblend, sample target-model replies via an OpenAI-compatible
API, and save speculators-ready JSONL (role/content format).

Typical usage (vLLM started by script/sever.sh):

    python examples/train/regenerate_open_perfectblend.py \\
        --dataset-path /home/libowen/spec/open-perfectblend \\
        --endpoint http://127.0.0.1:8000/v1/chat/completions \\
        --api-model qwen \\
        --output-dir ./output/perfectblend_qwen3_8b \\
        --limit 1000

Outputs (under --output-dir):
    regenerated.jsonl          # JSONL for scripts/prepare_data.py
    regenerated.errors.jsonl   # failed rows (if any)

Sampling defaults match ``scripts/response_regeneration/script.py``: only
``model``, ``messages``, and ``max_tokens`` (8192) are sent unless optional
generation flags are explicitly set.
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import aiohttp
from datasets import load_dataset
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load local open-perfectblend, sample assistant replies via "
            "target-model API, and save speculators-ready JSONL."
        )
    )
    parser.add_argument(
        "--dataset-path",
        default="/home/libowen/spec/open-perfectblend",
        help="Directory with data/train-*.parquet, or a parquet/jsonl file/glob",
    )
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:8000/v1/chat/completions",
        help="OpenAI-compatible chat completions endpoint",
    )
    parser.add_argument(
        "--api-model",
        default=None,
        help="Model name in API requests (e.g. qwen). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./output/perfectblend_regen"),
        help="Directory for regenerated.jsonl and regenerated.errors.jsonl",
    )
    parser.add_argument("--limit", type=int, default=None, help="Stop after N rows")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=64,
        help="Max concurrent requests (default: 64, same as response_regeneration)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="max_tokens per assistant turn (default: 8192, same as response_regeneration)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (default: unset, use server default)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Nucleus sampling top_p (default: unset, use server default)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling via extra_body (default: unset, use server default)",
    )
    parser.add_argument(
        "--min-p",
        type=float,
        default=None,
        help="Min-p sampling via extra_body (default: unset, use server default)",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help="Repetition penalty as presence_penalty (default: unset)",
    )
    thinking_group = parser.add_mutually_exclusive_group()
    thinking_group.add_argument(
        "--enable-thinking",
        dest="thinking_mode",
        action="store_const",
        const="enable",
        help="Set chat_template_kwargs.enable_thinking=True",
    )
    thinking_group.add_argument(
        "--disable-thinking",
        dest="thinking_mode",
        action="store_const",
        const="disable",
        help="Set chat_template_kwargs.enable_thinking=False",
    )
    parser.set_defaults(thinking_mode=None)
    parser.add_argument("--resume", action="store_true", help="Skip rows already in output")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.temperature is not None and not 0.0 <= args.temperature <= 2.0:
        raise ValueError("--temperature must be in [0.0, 2.0]")
    if args.top_p is not None and not 0.0 < args.top_p <= 1.0:
        raise ValueError("--top-p must be in (0.0, 1.0]")
    if args.top_k is not None and args.top_k <= 0:
        raise ValueError("--top-k must be greater than 0")
    if args.min_p is not None and not 0.0 <= args.min_p <= 1.0:
        raise ValueError("--min-p must be in [0.0, 1.0]")
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be greater than 0")
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be greater than 0")
    if args.repetition_penalty is not None and args.repetition_penalty < 0:
        raise ValueError("--repetition-penalty must be non-negative")


def build_chat_payload(
    api_model: str,
    messages: list[dict[str, str]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Build chat payload; default fields match response_regeneration/script.py."""
    payload: dict[str, Any] = {
        "model": api_model,
        "messages": messages,
        "max_tokens": args.max_tokens,
    }
    if args.temperature is not None:
        payload["temperature"] = args.temperature
    if args.top_p is not None:
        payload["top_p"] = args.top_p
    if args.repetition_penalty is not None:
        payload["presence_penalty"] = args.repetition_penalty

    extra_body: dict[str, Any] = {}
    if args.top_k is not None:
        extra_body["top_k"] = args.top_k
    if args.min_p is not None:
        extra_body["min_p"] = args.min_p
    if args.thinking_mode == "enable":
        extra_body.setdefault("chat_template_kwargs", {})["enable_thinking"] = True
    elif args.thinking_mode == "disable":
        extra_body.setdefault("chat_template_kwargs", {})["enable_thinking"] = False
    if extra_body:
        payload["extra_body"] = extra_body

    return payload


def _fmt_optional(value: Any) -> str:
    return "server default" if value is None else str(value)


def resolve_dataset_path(dataset_path: str) -> list[str]:
    path = os.path.abspath(dataset_path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset path does not exist: {path}")

    if path.endswith(".jsonl") or path.endswith(".json"):
        return [path]

    if os.path.isdir(path):
        parquet_files = sorted(glob.glob(os.path.join(path, "data", "train-*.parquet")))
        if parquet_files:
            return parquet_files
        jsonl_files = sorted(glob.glob(os.path.join(path, "*.jsonl")))
        if jsonl_files:
            return jsonl_files
        raise FileNotFoundError(
            f"No data/train-*.parquet or *.jsonl found under {path}"
        )

    return [path]


def load_local_dataset(data_files: list[str]):
    if data_files[0].endswith(".jsonl") or data_files[0].endswith(".json"):
        return load_dataset("json", data_files=data_files, split="train", streaming=True)
    return load_dataset("parquet", data_files=data_files, split="train", streaming=True)


def extract_user_turns(row: dict[str, Any]) -> list[dict[str, str]]:
    """Keep system + user turns; drop original assistant replies."""
    convs = row.get("conversations")
    if not isinstance(convs, list):
        return []

    turns: list[dict[str, str]] = []
    for message in convs:
        if not isinstance(message, dict):
            continue
        role = message.get("role") or message.get("from")
        content = message.get("content") or message.get("value")
        if not content or not isinstance(content, str):
            continue
        if role == "system":
            turns.append({"role": "system", "content": content})
        elif role in ("user", "human"):
            turns.append({"role": "user", "content": content})
    return turns if any(t["role"] == "user" for t in turns) else []


def load_seen_indices(path: Path) -> set[int]:
    seen: set[int] = set()
    if not path.is_file():
        return seen
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "source_index" in obj:
                seen.add(int(obj["source_index"]))
    return seen


async def detect_api_model(endpoint: str) -> str:
    models_url = endpoint.replace("/v1/chat/completions", "/v1/models")
    timeout = aiohttp.ClientTimeout(total=10)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(models_url) as response:
            response.raise_for_status()
            data = await response.json()
    models = data.get("data", [])
    if not models:
        raise RuntimeError(f"No models returned from {models_url}")
    model_id = models[0]["id"]
    print(f"Auto-detected API model: {model_id}")
    return model_id


async def post_chat(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    endpoint: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    async with sem:
        async with session.post(endpoint, json=payload) as response:
            if not response.ok:
                body = (await response.text())[:500]
                raise RuntimeError(f"HTTP {response.status}: {body}")
            return await response.json()


async def regenerate_conversation(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    endpoint: str,
    api_model: str,
    turns: list[dict[str, str]],
    args: argparse.Namespace,
) -> list[dict[str, str]]:
    prefix: list[dict[str, str]] = []
    out: list[dict[str, str]] = []

    for turn in turns:
        if turn["role"] == "system":
            prefix.append(turn)
            out.append(turn)
            continue

        prefix.append({"role": "user", "content": turn["content"]})
        out.append({"role": "user", "content": turn["content"]})

        payload = build_chat_payload(api_model, prefix, args)
        data = await post_chat(session, sem, endpoint, payload)
        choice = data["choices"][0]
        message = choice["message"]
        content = message.get("content")
        if not content:
            raise ValueError("empty assistant content from API")

        assistant_turn: dict[str, str] = {"role": "assistant", "content": content}
        reasoning = message.get("reasoning_content") or message.get("reasoning")
        if reasoning:
            assistant_turn["reasoning_content"] = reasoning

        prefix.append({"role": "assistant", "content": content})
        out.append(assistant_turn)

    return out


async def worker(
    queue: asyncio.Queue,
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    endpoint: str,
    api_model: str,
    args: argparse.Namespace,
    out_path: Path,
    err_path: Path,
    progress: tqdm,
    stats: dict[str, int],
) -> None:
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            return

        idx = item["idx"]
        source = item.get("source")
        turns = item["turns"]
        start = time.time()

        try:
            conversations = await regenerate_conversation(
                session, sem, endpoint, api_model, turns, args
            )
            record = {
                "conversations": conversations,
                "source_index": idx,
                "source_id": f"regenerated_{idx}",
            }
            if source:
                record["source"] = source

            with out_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

            stats["ok"] += 1
            stats["latency_total"] += time.time() - start
        except Exception as exc:  # noqa: BLE001
            error_record = {
                "source_index": idx,
                "source_id": f"regenerated_{idx}",
                "error": repr(exc),
                "partial_turns": turns,
            }
            with err_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(error_record, ensure_ascii=False) + "\n")
            stats["errors"] += 1
        finally:
            progress.set_postfix(ok=stats["ok"], errors=stats["errors"], refresh=False)
            progress.update(1)
            queue.task_done()


async def run_sampling(args: argparse.Namespace) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "regenerated.jsonl"
    err_path = args.output_dir / "regenerated.errors.jsonl"

    endpoint = args.endpoint
    if "0.0.0.0" in endpoint:
        print(
            "Warning: connecting to 0.0.0.0 is unreliable; "
            "use http://127.0.0.1:8000/... instead."
        )

    api_model = args.api_model or await detect_api_model(endpoint)
    data_files = resolve_dataset_path(args.dataset_path)
    dataset = load_local_dataset(data_files)
    seen = load_seen_indices(out_path) if args.resume else set()

    print(f"Endpoint:           {endpoint}")
    print(f"API model:          {api_model}")
    print(f"Max tokens:         {args.max_tokens}")
    print(f"Temperature:        {_fmt_optional(args.temperature)}")
    print(f"Top-p:              {_fmt_optional(args.top_p)}")
    print(f"Top-k:              {_fmt_optional(args.top_k)}")
    print(f"Min-p:              {_fmt_optional(args.min_p)}")
    print(f"Repetition penalty: {_fmt_optional(args.repetition_penalty)}")
    print(f"Thinking mode:      {_fmt_optional(args.thinking_mode)}")
    print(f"Concurrency:        {args.concurrency}")
    print(f"Dataset files:      {len(data_files)} file(s)")
    print(f"Output:             {out_path}")
    print(f"Errors:             {err_path}")
    print(f"Resume:             {args.resume} (skip {len(seen)} indices)")
    print()

    queue: asyncio.Queue = asyncio.Queue(maxsize=args.concurrency * 4)
    sem = asyncio.Semaphore(args.concurrency)
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=90, sock_read=None)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        with tqdm(total=args.limit, desc="Sampling", unit="row", dynamic_ncols=True) as progress:
            stats = {"ok": 0, "errors": 0, "latency_total": 0.0}
            workers = [
                asyncio.create_task(
                    worker(
                        queue,
                        session,
                        sem,
                        endpoint,
                        api_model,
                        args,
                        out_path,
                        err_path,
                        progress,
                        stats,
                    )
                )
                for _ in range(args.concurrency)
            ]

            queued = 0
            for index, row in enumerate(dataset):
                if args.limit is not None and queued >= args.limit:
                    break
                if index in seen:
                    continue

                turns = extract_user_turns(row)
                if not turns:
                    continue

                await queue.put(
                    {
                        "idx": index,
                        "source": row.get("source"),
                        "turns": turns,
                    }
                )
                queued += 1

            for _ in workers:
                await queue.put(None)
            await asyncio.gather(*workers)

            if stats["ok"]:
                avg = stats["latency_total"] / stats["ok"]
                print(
                    f"\nSampled {stats['ok']} rows "
                    f"(errors: {stats['errors']}, avg {avg:.2f}s/row)"
                )
            else:
                print(f"\nNo rows sampled (errors: {stats['errors']})")

    return out_path


def main() -> None:
    args = parse_args()
    validate_args(args)

    jsonl_path = asyncio.run(run_sampling(args))
    ok_count = sum(1 for _ in jsonl_path.open(encoding="utf-8"))
    if ok_count == 0:
        print("No successful samples written.", file=sys.stderr)
        sys.exit(1)

    print("\nDone.")
    print(f"  JSONL: {jsonl_path}")
    print("\nNext step — tokenize for training:")
    print(
        f"  python scripts/prepare_data.py "
        f"--model <target_model_path> --data {jsonl_path} "
        f"--output <output_dir>"
    )


if __name__ == "__main__":
    main()
