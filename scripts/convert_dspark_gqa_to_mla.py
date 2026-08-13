#!/usr/bin/env python3
"""Convert a Speculators GQA DSpark/DFlash draft checkpoint to MLA layout.

Lossy warm-start: expand GQA KV → MHA, remap NeoX RoPE dims to interleaved
MLA rope layout, pad/truncate nope/v dims, share k_rope. By default uses
dense Q and full-rank KV factorization (no LoRA truncation); pass
``--q-lora-rank`` / ``--kv-lora-rank`` to compress. Non-attention weights
are copied as-is. config.json is rewritten with attention_type=mla.

Example:
    python scripts/convert_dspark_gqa_to_mla.py \\
        ./output/dspark_gqa/checkpoints/5 \\
        -o ./output/dspark_mla_init

    # Optional: compress to match a verifier MLA geometry
    python scripts/convert_dspark_gqa_to_mla.py ./gqa_ckpt -o ./mla_ckpt \\
        --q-lora-rank 2048 --kv-lora-rank 512 \\
        --qk-nope-head-dim 192 --qk-rope-head-dim 64 --v-head-dim 256
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

ATTN_RE = re.compile(r"^layers\.(\d+)\.self_attn\.")


def load_config(path: Path) -> dict:
    with (path / "config.json").open() as f:
        return json.load(f)


def load_weights(path: Path) -> dict[str, torch.Tensor]:
    st = path / "model.safetensors"
    if not st.exists():
        raise FileNotFoundError(f"No model.safetensors at {path}")
    weights = {}
    with safe_open(st, framework="pt") as f:
        for key in f.keys():  # noqa: SIM118
            weights[key] = f.get_tensor(key)
    return weights


def low_rank_factor(weight: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """W (out, in) ≈ A @ B with A (out, r), B (r, in)."""
    w = weight.float()
    rank = min(rank, w.shape[0], w.shape[1])
    u, s, vh = torch.linalg.svd(w, full_matrices=False)
    u, s, vh = u[:, :rank], s[:rank], vh[:rank, :]
    scale = s.clamp_min(0).sqrt()
    return u * scale.unsqueeze(0), scale.unsqueeze(1) * vh


def resize_dim(w: torch.Tensor, src: int, dst: int, dim: int = 1) -> torch.Tensor:
    """Pad/truncate a size-`src` axis to `dst` (default head-feature axis)."""
    if dst == src:
        return w
    if dst < src:
        return w.narrow(dim, 0, dst)
    # F.pad pads from the last dim; build pad for `dim`.
    pad_pairs = []
    for i in range(w.ndim - 1, -1, -1):
        if i == dim:
            pad_pairs.extend([0, dst - src])
        else:
            pad_pairs.extend([0, 0])
    return F.pad(w, pad_pairs)


def neox_head_to_mla_qk(
    heads: torch.Tensor,
    *,
    qk_nope: int,
    qk_rope: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a NeoX-RoPE head into MLA ``[nope | rope]`` with interleaved rope.

    GQA/Qwen applies NeoX half-rotate on the full ``head_dim``: frequency ``i``
    lives at indices ``(i, i + head_dim//2)``. MLA applies interleaved RoPE only
    on ``qk_rope`` dims, pairing ``(2i, 2i+1)``.

    We keep the lowest ``qk_rope//2`` NeoX frequencies for rope and rearrange:
        interleaved[2i]     = neox[i]
        interleaved[2i + 1] = neox[i + head_dim//2]
    Leftover NeoX dims (higher freqs) become nope (pad/truncate to ``qk_nope``).

    :param heads: ``(num_heads, head_dim, hidden)``
    :return: ``nope (H, qk_nope, D)``, ``rope (H, qk_rope, D)``
    """
    if qk_rope % 2 != 0:
        raise ValueError(f"qk_rope_head_dim must be even for NeoX↔interleaved remap, got {qk_rope}")
    _num_heads, head_dim, _hidden = heads.shape
    if head_dim % 2 != 0:
        raise ValueError(f"GQA head_dim must be even, got {head_dim}")
    half = head_dim // 2
    rope_half = qk_rope // 2
    if rope_half > half:
        raise ValueError(
            f"qk_rope_head_dim={qk_rope} exceeds GQA head_dim={head_dim}"
        )

    first = heads[:, :half, :]       # freqs 0..half-1
    second = heads[:, half:, :]      # same freqs, NeoX second half

    # Lowest frequencies → MLA rope, NeoX → interleaved dim order.
    rope = torch.stack(
        (first[:, :rope_half, :], second[:, :rope_half, :]),
        dim=2,
    ).reshape(heads.shape[0], qk_rope, heads.shape[2])

    # Remaining higher-frequency NeoX dims → nope (approximate; GQA had RoPE on all).
    leftover = torch.cat(
        (first[:, rope_half:, :], second[:, rope_half:, :]),
        dim=1,
    )
    nope = resize_dim(leftover, leftover.shape[1], qk_nope, dim=1)
    return nope, rope


def convert_layer(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    q_norm: torch.Tensor | None,
    k_norm: torch.Tensor | None,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    hidden_size: int,
    q_lora_rank: int | None,
    kv_lora_rank: int,
    qk_nope: int,
    qk_rope: int,
    v_dim: int,
) -> dict[str, torch.Tensor]:
    dtype = q.dtype
    groups = num_heads // num_kv_heads

    q, k, v, o = q.float(), k.float(), v.float(), o.float()
    if q_norm is not None:
        q = q.reshape(num_heads, head_dim, hidden_size) * q_norm.float().view(1, -1, 1)
    else:
        q = q.reshape(num_heads, head_dim, hidden_size)
    if k_norm is not None:
        k = k.reshape(num_kv_heads, head_dim, hidden_size) * k_norm.float().view(1, -1, 1)
    else:
        k = k.reshape(num_kv_heads, head_dim, hidden_size)

    # GQA -> MHA
    k = k.repeat_interleave(groups, dim=0)
    v = v.reshape(num_kv_heads, head_dim, hidden_size).repeat_interleave(groups, dim=0)

    # NeoX full-head RoPE → MLA [nope | interleaved rope]
    q_nope, q_rope = neox_head_to_mla_qk(q, qk_nope=qk_nope, qk_rope=qk_rope)
    k_nope, k_rope_heads = neox_head_to_mla_qk(k, qk_nope=qk_nope, qk_rope=qk_rope)
    q = torch.cat([q_nope, q_rope], dim=1).reshape(num_heads * (qk_nope + qk_rope), hidden_size)
    k_rope = k_rope_heads.mean(dim=0)  # shared MQA rope key

    v = resize_dim(v, head_dim, v_dim, dim=1)
    v_heads = v  # (H, v_dim, D)

    # o_proj: (D, H*d) -> (D, H*v_dim); V has no RoPE, plain resize.
    o = o.reshape(hidden_size, num_heads, head_dim)
    o = resize_dim(o, head_dim, v_dim, dim=2)
    o = o.reshape(hidden_size, num_heads * v_dim)

    kv_dense = torch.cat([k_nope, v_heads], dim=1).reshape(
        num_heads * (qk_nope + v_dim), hidden_size
    )
    kv_b, kv_a = low_rank_factor(kv_dense, kv_lora_rank)

    out: dict[str, torch.Tensor] = {
        "kv_a_proj_with_mqa.weight": torch.cat([kv_a, k_rope], dim=0).to(dtype),
        "kv_a_layernorm.weight": torch.ones(kv_lora_rank, dtype=dtype),
        "kv_b_proj.weight": kv_b.to(dtype),
        "o_proj.weight": o.to(dtype),
    }
    if q_lora_rank is not None:
        q_b, q_a = low_rank_factor(q, q_lora_rank)
        out["q_a_proj.weight"] = q_a.to(dtype)
        out["q_a_layernorm.weight"] = torch.ones(q_lora_rank, dtype=dtype)
        out["q_b_proj.weight"] = q_b.to(dtype)
    else:
        out["q_proj.weight"] = q.to(dtype)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input_path", type=Path, help="GQA Speculators checkpoint dir")
    p.add_argument("-o", "--output-path", type=Path, required=True, help="Output MLA checkpoint dir")
    p.add_argument(
        "--q-lora-rank",
        type=int,
        default=None,
        help="Q LoRA rank; omit for dense q_proj (default, no Q compression)",
    )
    p.add_argument(
        "--kv-lora-rank",
        type=int,
        default=None,
        help="KV LoRA rank (default: full H*(nope+v), no SVD truncation)",
    )
    p.add_argument("--qk-nope-head-dim", type=int, default=None)
    p.add_argument("--qk-rope-head-dim", type=int, default=None)
    p.add_argument("--v-head-dim", type=int, default=None)
    args = p.parse_args()

    src = args.input_path
    cfg = load_config(src)
    weights = load_weights(src)

    if cfg.get("attention_type", "gqa") == "mla":
        raise SystemExit("Source already has attention_type=mla")

    tl = cfg["transformer_layer_config"]
    hidden = int(tl["hidden_size"])
    num_heads = int(tl["num_attention_heads"])
    num_kv = int(tl.get("num_key_value_heads", num_heads))
    head_dim = int(tl.get("head_dim", hidden // num_heads))
    num_layers = int(tl["num_hidden_layers"])

    rope = args.qk_rope_head_dim if args.qk_rope_head_dim is not None else min(64, head_dim // 2)
    # Keep rope even so NeoX (i, i+d/2) can map to interleaved (2i, 2i+1).
    if rope % 2:
        rope -= 1
    nope = args.qk_nope_head_dim if args.qk_nope_head_dim is not None else head_dim - rope
    v_dim = args.v_head_dim if args.v_head_dim is not None else head_dim
    if rope <= 0 or rope > head_dim or rope % 2:
        raise SystemExit(f"Invalid qk_rope_head_dim={rope} for head_dim={head_dim}")

    # Default: no compression — dense Q + full-rank KV factorization.
    max_kv_rank = num_heads * (nope + v_dim)
    kv_rank = args.kv_lora_rank if args.kv_lora_rank is not None else max_kv_rank
    q_rank = args.q_lora_rank  # None → dense q_proj

    print(
        f"MLA dims: q_lora={q_rank} kv_lora={kv_rank} "
        f"nope={nope} rope={rope} v={v_dim}"
    )
    print(
        "RoPE remap: NeoX (i, i+d/2) → interleaved (2i, 2i+1); "
        f"keep lowest {rope // 2} freqs for rope, rest → nope"
    )

    # Rewrite config
    out_cfg = copy.deepcopy(cfg)
    mla_fields = {
        "attention_type": "mla",
        "q_lora_rank": q_rank,
        "kv_lora_rank": kv_rank,
        "qk_nope_head_dim": nope,
        "qk_rope_head_dim": rope,
        "v_head_dim": v_dim,
        "mla_use_output_gate": False,
    }
    out_cfg.update(mla_fields)
    out_cfg["transformer_layer_config"].update(mla_fields)
    out_cfg["transformer_layer_config"]["num_key_value_heads"] = num_heads

    # Convert weights
    out_w: dict[str, torch.Tensor] = {
        k: v for k, v in weights.items() if not ATTN_RE.match(k)
    }
    for i in tqdm(range(num_layers), desc="Converting layers", unit="layer"):
        pref = f"layers.{i}.self_attn."
        converted = convert_layer(
            weights[pref + "q_proj.weight"],
            weights[pref + "k_proj.weight"],
            weights[pref + "v_proj.weight"],
            weights[pref + "o_proj.weight"],
            weights.get(pref + "q_norm.weight"),
            weights.get(pref + "k_norm.weight"),
            num_heads=num_heads,
            num_kv_heads=num_kv,
            head_dim=head_dim,
            hidden_size=hidden,
            q_lora_rank=q_rank,
            kv_lora_rank=kv_rank,
            qk_nope=nope,
            qk_rope=rope,
            v_dim=v_dim,
        )
        for name, tensor in converted.items():
            out_w[pref + name] = tensor

    dst = args.output_path
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "config.json").write_text(json.dumps(out_cfg, indent=2) + "\n")
    print("Saving model.safetensors ...")
    save_file({k: t.contiguous() for k, t in out_w.items()}, str(dst / "model.safetensors"))
    print(f"Wrote {dst}")


if __name__ == "__main__":
    main()
