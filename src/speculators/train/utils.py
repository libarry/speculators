import datetime
import importlib.metadata
import logging
import os
import re
import shlex
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

from speculators.data_generation.preprocessing import get_tokenizer, load_processor

logger = logging.getLogger("speculators")


def resolve_mask_token_id(
    verifier_name_or_path: str,
    vocab_size: int,
    mask_token_id: int | None = None,
    *,
    trust_remote_code: bool = False,
) -> int:
    """Resolve mask_token_id from explicit value, tokenizer, or fallback.

    Resolution order:
        1. Explicit mask_token_id if provided
        2. Tokenizer's existing mask_token_id
        3. Add <|MASK|> to tokenizer if embed_tokens has unused slots
        4. Fallback to pad/eos/unk token
    """
    if mask_token_id is not None:
        logger.info(f"Using explicit mask_token_id={mask_token_id}")
        return mask_token_id

    processor = load_processor(
        verifier_name_or_path,
        trust_remote_code=trust_remote_code,
    )
    tokenizer = get_tokenizer(processor)

    if tokenizer.mask_token_id is not None:
        logger.info(f"Using tokenizer mask_token_id={tokenizer.mask_token_id}")
        return tokenizer.mask_token_id

    if len(tokenizer) < vocab_size:
        tokenizer.add_special_tokens({"mask_token": "<|MASK|>"})
        added_id: int = tokenizer.mask_token_id  # type: ignore[assignment]
        logger.warning(
            f"Added <|MASK|> to tokenizer, mask_token_id={added_id} "
            f"(tokenizer len={len(tokenizer)}, vocab_size={vocab_size})"
        )
        return added_id

    for token_name in ("pad_token_id", "eos_token_id", "unk_token_id"):
        token_id = getattr(tokenizer, token_name, None)
        if token_id is not None:
            warnings.warn(
                f"Tokenizer does not have mask_token and no unused embedding slots. "
                f"Using {token_name}={token_id} as fallback.",
                stacklevel=2,
            )
            return token_id

    raise ValueError(
        "Could not resolve mask_token_id: no --mask-token-id provided, tokenizer has "
        "no mask_token, no unused embedding slots, and no pad/eos/unk fallback tokens."
    )


# Per-replica means (``*_total = 1``), as opposed to token-count totals such as
# ``full_acc_0_total``.  Under Ulysses SP every rank in an SP group shares one
# packed sequence; WORLD-SUM of these replica weights would count the sequence
# ``sp_size`` times unless they are scaled by ``1/sp_size`` before reduce.
_REPLICA_TOTAL_RE = re.compile(
    r"(?:^loss(?:_\d+)?_total$)|(?:_loss(?:_\d+)?_total$)"
    r"|(?:^confidence_loss_total$)|(?:^eal_total$)"
)


def is_replica_weighted_total_key(key: str) -> bool:
    """Return True if ``key`` is a per-replica ``*_total`` (not a token count)."""
    return bool(_REPLICA_TOTAL_RE.search(key))


def scale_replica_totals_for_sp(metrics: dict, sp_size: int) -> dict:
    """In-place scale replica-weight ``*_total`` values by ``1/sp_size``.

    After a WORLD ``ReduceOp.SUM``, ``loss_sum / loss_total`` then averages over
    DP replicas only: SP ranks already contribute shards of one global loss
    (see Eagle3 ``_sp_scale_loss``). Token-count totals are left unchanged so
    accuracy stays ``correct / tokens`` across the full sequence.
    """
    if sp_size <= 1:
        return metrics
    inv = 1.0 / sp_size
    for key, value in metrics.items():
        if is_replica_weighted_total_key(key):
            metrics[key] = value * inv
    return metrics


def normalize_counted_metrics(
    metrics: dict[str, float], world_size: int = 1
) -> dict[str, float]:
    """Normalize metrics after ReduceOp.SUM across ranks.

    For any key ending in '_total', finds the matching '_sum' key,
    computes sum / total, and stores the result under the prefix
    (e.g. 'loss_sum' / 'loss_total' -> 'loss').
    The raw sum/total keys are removed.

    Any remaining metrics (not part of a sum/total pair) are divided
    by world_size to compute the average across ranks.

    With sequence parallel, pass ``world_size=dp_size`` (not ``DP×SP``) and
    call :func:`scale_replica_totals_for_sp` *before* the reduce.
    """
    normalized_keys: set[str] = set()
    for tk in [k for k in metrics if k.endswith("_total")]:
        prefix = tk.removesuffix("_total")
        sk = f"{prefix}_sum"
        if sk in metrics:
            total = metrics[tk]
            metrics[prefix] = metrics[sk] / total if total > 0 else 0.0
            del metrics[sk]
            normalized_keys.add(prefix)
        del metrics[tk]

    if world_size > 1:
        for k in metrics:
            if k not in normalized_keys:
                metrics[k] /= world_size

    return metrics


def save_train_command(save_path: str, argv: list[str] | None = None) -> None:
    """Write the launch command and provenance header to save_path/train_command.txt.

    ``argv`` is the exact command the run was resolved from (``TrainConfig`` records
    it during resolution); it falls back to the live ``sys.argv`` when a caller has
    no recorded argv, so a direct call is unchanged.
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        sha = "unknown"

    pkg_versions: list[str] = []
    for pkg in ("speculators", "vllm", "transformers", "torch", "compressed-tensors"):
        try:
            ver = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            ver = "not installed"
        pkg_versions.append(f"# {pkg}: {ver}")

    header = "\n".join(
        [
            f"# Timestamp: {datetime.datetime.now(datetime.timezone.utc).isoformat()}",
            f"# Git SHA: {sha}",
            f"# World size: {os.environ.get('WORLD_SIZE', '1')}",
            *pkg_versions,
        ]
    )

    command = shlex.join(argv or sys.argv)
    content = f"{header}\n{command}\n"

    path = Path(save_path)
    path.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=save_path, prefix=".train_command_", suffix=".tmp")
    tmp_path = Path(tmp)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
        tmp_path.replace(path / "train_command.txt")
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
