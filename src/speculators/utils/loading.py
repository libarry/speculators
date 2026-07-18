import json
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from loguru import logger
from safetensors import safe_open

_WEIGHT_ALIASES: dict[str, list[str]] = {
    "embed_tokens.weight": ["tok_embeddings.weight"],
    "lm_head.weight": ["output.weight"],
    "model.norm.weight": ["norm.weight"],
}


def _resolve_key(name: str, weight_map: dict[str, str]) -> str | None:
    """Try exact match, then suffix match, then known aliases."""
    for candidate in [name, *_WEIGHT_ALIASES.get(name, [])]:
        if candidate in weight_map:
            return candidate
        matched = next((k for k in weight_map if k.endswith(candidate)), None)
        if matched:
            return matched
    return None


def is_config_only_dir(path: str | Path) -> bool:
    """Return True if ``path`` is a local directory with a ``config.json`` but no
    weight files (``*.safetensors`` / ``*.bin``).

    Used to distinguish a saved speculator *config* (from which a fresh draft is
    initialized) from a full checkpoint whose weights should be loaded.

    :param path: A local directory path. Hub ids and non-directories return False.
    :return: True when the directory holds a config but no weights.
    """
    directory = Path(path)
    if not directory.is_dir():
        return False
    has_config = (directory / "config.json").is_file()
    # Weight files, plus sharded-checkpoint index files (e.g.
    # model.safetensors.index.json) -- the latter end in .json and would not match
    # the *.safetensors / *.bin globs, so a shard manifest must be checked explicitly
    # to avoid treating an incomplete sharded checkpoint as config-only.
    has_weights = (
        any(directory.glob("*.safetensors"))
        or any(directory.glob("*.bin"))
        or any(directory.glob("*.safetensors.index.json"))
        or any(directory.glob("*.bin.index.json"))
    )
    return has_config and not has_weights


def _list_safetensors_shards(model_path: str) -> list[str]:
    """List top-level ``*.safetensors`` shard filenames for a local dir or Hub repo."""
    model_path_obj = Path(model_path)
    if model_path_obj.is_dir():
        return sorted(p.name for p in model_path_obj.glob("*.safetensors") if p.is_file())

    from huggingface_hub import list_repo_files  # noqa: PLC0415

    return sorted(
        f
        for f in list_repo_files(repo_id=model_path)
        if f.endswith(".safetensors") and Path(f).name == f
    )


def _build_weight_map_from_safetensors(model_path: str) -> dict[str, str]:
    """Build a virtual weight_map by scanning all ``*.safetensors`` shards.

    Supports arbitrarily named shards (e.g. ``pytorch_model-00001-of-00002.safetensors``)
    when ``model.safetensors.index.json`` / ``model.safetensors`` are absent.
    """
    shard_files = _list_safetensors_shards(model_path)
    if not shard_files:
        raise FileNotFoundError(
            f"No .safetensors weight files found for {model_path}. "
            "Expected model.safetensors.index.json, model.safetensors, "
            "or one or more *.safetensors shards."
        )

    logger.info(
        "Building weight_map by scanning {} safetensors file(s) under {}",
        len(shard_files),
        model_path,
    )
    weight_map: dict[str, str] = {}
    for shard_file in shard_files:
        shard_path = _resolve_file(model_path, shard_file)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key in weight_map:
                    raise ValueError(
                        f"Duplicate tensor key '{key}' found in "
                        f"'{weight_map[key]}' and '{shard_file}'"
                    )
                weight_map[key] = shard_file
    return weight_map


def list_checkpoint_keys(checkpoint_dir: str | Path) -> list[str]:
    """List all tensor keys in a checkpoint without loading weights.

    Supports sharded safetensors (via index) and single/multi safetensors formats.

    :param checkpoint_dir: Path to a local checkpoint directory.
    :return: List of tensor key names present in the checkpoint.
    """
    checkpoint_dir = Path(checkpoint_dir)

    index_path = checkpoint_dir / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open() as f:
            return list(json.load(f)["weight_map"].keys())

    return list(_build_weight_map_from_safetensors(str(checkpoint_dir)).keys())


def load_model_layers(
    layer_names: list[str], model_path: str
) -> dict[str, torch.Tensor]:
    """
    Load one or more named tensors from a HF repo using safetensors shards.
    Supports both exact keys and suffix pattern matching.

    :param layer_names: list of tensor names or suffix patterns to load, e.g.
    ["model.embed_tokens.weight", "lm_head.weight"]
    :param model_path: either a local directory of huggingface model
    containing model.safetensors.index
    :return: dict mapping input names/patterns to loaded tensors
    """
    # Prefer the HF index when present; otherwise scan all *.safetensors shards.
    try:
        index_file = _resolve_file(model_path, "model.safetensors.index.json")
        with Path(index_file).open() as f:
            index = json.load(f)
        weight_map: dict[str, str] = index["weight_map"]
    except (FileNotFoundError, EntryNotFoundError):
        logger.warning(
            "`model.safetensors.index.json` file not found. "
            "Falling back to scanning *.safetensors files."
        )
        weight_map = _build_weight_map_from_safetensors(model_path)

    # Resolve names: try exact match, then suffix match, then known aliases
    name_to_key = {}  # Maps input name to actual checkpoint key
    for name in layer_names:
        key = _resolve_key(name, weight_map)
        if key:
            name_to_key[name] = key
        else:
            logger.warning(f"Tensor '{name}' not found in weight_map.")

    # group requested names by shard filename
    shard_to_names: dict[str, list[tuple[str, str]]] = {}
    for name, key in name_to_key.items():
        shard = weight_map[key]
        shard_to_names.setdefault(shard, []).append((name, key))

    if not shard_to_names:
        raise ValueError("None of the requested tensor names were found in the index.")

    # fetch each required shard and extract only the requested tensors
    out: dict[str, Any] = {}
    for shard_file, name_key_pairs in shard_to_names.items():
        shard_path = _resolve_file(model_path, shard_file)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for name, key in name_key_pairs:
                out[name] = f.get_tensor(key)
    return out


def _resolve_file(model_path: str, file_name: str) -> Path:
    """
    If model_path is a local directory, return path/<filename> if it exists.
    Otherwise treat model_path as a HF repo_id and download with hf_hub_download.

    :param model_path: local directory or HF repo_id
    :param file_name: filename to look for or download
    :return: local path to the resolved file
    """
    model_path_obj = Path(model_path)
    if model_path_obj.is_dir():
        logger.info("Loading from local directory: {}", model_path)
        p = model_path_obj / file_name
        if not p.exists():
            raise FileNotFoundError(f"Expected local file missing: {p}")
        return p
    # Treat as repo_id on the Hub
    logger.info(f"Loading from huggingface directory: {model_path}: {file_name}")
    return Path(hf_hub_download(repo_id=model_path, filename=file_name))
