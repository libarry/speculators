"""Configuration for PARD-2 speculator."""

from typing import Literal

from pydantic import Field

from speculators import SpeculatorModelConfig

__all__ = ["Pard2SpeculatorConfig"]


@SpeculatorModelConfig.register("pard2")
class Pard2SpeculatorConfig(SpeculatorModelConfig):
    """Configuration for PARD-2 target-aligned parallel draft training."""

    speculators_model_type: Literal["pard2"] = "pard2"
    architectures: list[str] = Field(
        default_factory=lambda: ["Pard2Speculator"],
        description="Model architectures that can load these weights",
    )

    draft_name_or_path: str = Field(
        default="",
        description="Hugging Face id or path to the draft (base) causal LM",
    )

    para_num: int = Field(
        default=16,
        ge=1,
        le=32,
        description="Number of parallel draft blocks (para_num)",
    )

    down_sample_ratio: float = Field(
        default=0.7,
        gt=0.0,
        le=1.0,
        description="COD geometric decay ratio",
    )

    down_sample_ratio_min: float = Field(
        default=0.1,
        gt=0.0,
        le=1.0,
        description="Minimum COD retention floor",
    )

    mask_token_ids: list[int] = Field(
        default_factory=list,
        description="Placeholder token ids for parallel draft positions",
    )

    feat_scale: float = Field(
        default=0.02,
        description="Scale applied to projected target features",
    )

    proj_bias: bool = Field(
        default=False,
        description="Whether target_proj uses bias",
    )

    target_layer_ids: list[int] = Field(
        default_factory=lambda: [-1, -8, -16, -24],
        description="Target verifier layer ids concatenated for target_feat",
    )

    target_feat_dim: int = Field(
        default=0,
        description="Concatenated target hidden dim (hidden_size * num target layers)",
    )

    target_feat_mask: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Probability of dropping target features per batch during training",
    )

    ce_alpha: float = Field(default=0.1, description="CE loss weight")
    kd_alpha: float = Field(default=1.0, description="KD loss weight")
    kd_temperature: float = Field(default=1.0, description="KD temperature")
    prev_prob_loss: bool = Field(
        default=True,
        description="Use CAT prev_prob weighting for CE/KD",
    )

    end_token_id: int | None = Field(
        default=None,
        description="Only compute loss after the last occurrence of this token",
    )

    pard_token: int = Field(
        default=-1,
        description="Primary parallel draft placeholder token id for inference metadata",
    )
