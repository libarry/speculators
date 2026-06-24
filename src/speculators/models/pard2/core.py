"""PARD-2 draft model with target-aligned feature injection."""

from __future__ import annotations

from typing import Any, ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.model import SpeculatorModel
from speculators.models.pard2.config import Pard2SpeculatorConfig
from speculators.models.utils import resolve_target_layer_ids
from speculators.proposals.greedy import GreedyTokenProposalConfig
from speculators.utils.loading import load_model_layers

__all__ = ["Pard2DraftModel", "load_verifier_lm_head"]


def load_verifier_lm_head(path: str) -> nn.Linear:
    weights = load_model_layers(["lm_head.weight"], path)
    weight = weights["lm_head.weight"]
    return nn.Linear(weight.shape[1], weight.shape[0], bias=False)


@SpeculatorModel.register("pard2")
class Pard2DraftModel(SpeculatorModel):
    """PARD-2 target-aligned parallel draft model."""

    config_class: ClassVar[type[Pard2SpeculatorConfig]] = Pard2SpeculatorConfig  # type: ignore[misc]
    _keys_to_ignore_on_save: ClassVar[list[str]] = [  # type: ignore[misc]
        "verifier_lm_head.weight",
        "verifier_lm_head.bias",
    ]
    _keys_to_ignore_on_load_missing: ClassVar[list[str]] = [  # type: ignore[misc]
        "verifier_lm_head.weight",
        "verifier_lm_head.bias",
    ]

    def __init__(self, config: Pard2SpeculatorConfig):
        super().__init__(config=config)
        draft_config = AutoConfig.from_pretrained(config.draft_name_or_path)
        self.draft_model = AutoModelForCausalLM.from_pretrained(
            config.draft_name_or_path,
            torch_dtype=torch.bfloat16,
        )
        hidden_size = draft_config.hidden_size
        self.target_proj = nn.Linear(
            config.target_feat_dim,
            hidden_size,
            bias=config.proj_bias,
        )
        self.verifier_lm_head = load_verifier_lm_head(
            config.speculators_config.verifier.name_or_path  # type: ignore[union-attr]
        )
        for param in self.verifier_lm_head.parameters():
            param.requires_grad_(False)

        self.ce_alpha = config.ce_alpha
        self.kd_alpha = config.kd_alpha
        self.kd_temperature = config.kd_temperature
        self.target_feat_mask = config.target_feat_mask
        self.prev_prob_loss = config.prev_prob_loss
        self.feat_scale = config.feat_scale

        self.post_init()

    @property
    def layers(self) -> nn.ModuleList:
        return self.draft_model.model.layers  # type: ignore[attr-defined,union-attr]

    @property
    def target_layer_ids(self) -> list[int]:
        return list(self.config.target_layer_ids)

    def gradient_checkpointing_enable(self, *args: Any, **kwargs: Any):
        if hasattr(self.draft_model, "gradient_checkpointing_enable"):
            return self.draft_model.gradient_checkpointing_enable(*args, **kwargs)
        return None

    def gradient_checkpointing_disable(self, *args: Any, **kwargs: Any):
        if hasattr(self.draft_model, "gradient_checkpointing_disable"):
            return self.draft_model.gradient_checkpointing_disable(*args, **kwargs)
        return None

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        target_feat: torch.Tensor | None = None,
        teacher_hidden: torch.Tensor | None = None,
        prev_prob: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[None, torch.Tensor, dict[str, torch.Tensor]]:
        del kwargs

        if input_ids is None:
            raise ValueError("input_ids are required")

        inputs_embeds = self.draft_model.get_input_embeddings()(input_ids)

        if target_feat is not None:
            tf = target_feat.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            tf_proj = self.target_proj(tf) * self.feat_scale
            if tf_proj.shape[:2] != inputs_embeds.shape[:2]:
                raise ValueError(
                    f"target_feat seq mismatch: proj={tuple(tf_proj.shape)} "
                    f"vs embeds={tuple(inputs_embeds.shape)}"
                )
            bsz = tf_proj.shape[0]
            keep_mask = (
                torch.rand(bsz, 1, 1, device=tf_proj.device) > self.target_feat_mask
            ).to(tf_proj.dtype)
            inputs_embeds = inputs_embeds + tf_proj * keep_mask

        outputs = self.draft_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )

        student_logits = outputs.logits
        ce_loss: torch.Tensor | None = None
        kd_loss: torch.Tensor | None = None
        token_weight: torch.Tensor | None = None

        if labels is not None:
            shift_labels = labels[..., 1:].contiguous()
            valid_mask = (shift_labels != -100).to(student_logits.dtype)

            if prev_prob is not None and self.prev_prob_loss:
                shift_prev_prob = prev_prob[..., 1:].to(
                    device=student_logits.device,
                    dtype=student_logits.dtype,
                )
                token_weight = valid_mask * shift_prev_prob
            else:
                token_weight = valid_mask

            shift_student_logits = student_logits[..., :-1, :].contiguous().float()
            vocab_size = shift_student_logits.size(-1)
            ce_per_token = F.cross_entropy(
                shift_student_logits.view(-1, vocab_size),
                shift_labels.view(-1),
                reduction="none",
                ignore_index=-100,
            ).view_as(shift_labels)
            denom = token_weight.sum().clamp_min(1e-6)
            ce_loss = (ce_per_token * token_weight).sum() / denom

        if teacher_hidden is not None:
            teacher_hidden_t = teacher_hidden.to(
                device=student_logits.device,
                dtype=student_logits.dtype,
            )
            teacher_logits = self.verifier_lm_head(teacher_hidden_t.float())
            temperature = float(self.kd_temperature)
            shift_teacher_logits = teacher_logits[..., :-1, :].contiguous().float()
            shift_student_logits = student_logits[..., :-1, :].contiguous().float()
            teacher_prob = F.softmax(shift_teacher_logits / temperature, dim=-1)
            student_log_prob = F.log_softmax(shift_student_logits / temperature, dim=-1)
            token_kd = (
                F.kl_div(student_log_prob, teacher_prob, reduction="none").sum(dim=-1)
                * (temperature**2)
            )

            if labels is not None and token_weight is not None:
                kd_weight = token_weight.to(token_kd.dtype)
                kd_loss = (token_kd * kd_weight).sum() / kd_weight.sum().clamp_min(1e-6)
            elif prev_prob is not None and self.prev_prob_loss:
                kd_weight = prev_prob[..., 1:].to(
                    device=token_kd.device,
                    dtype=token_kd.dtype,
                )
                kd_loss = (token_kd * kd_weight).sum() / kd_weight.sum().clamp_min(1e-6)
            else:
                kd_loss = token_kd.mean()

        if ce_loss is not None and kd_loss is not None:
            final_loss = ce_loss * self.ce_alpha + kd_loss * self.kd_alpha
        elif ce_loss is not None:
            final_loss = ce_loss
        elif kd_loss is not None:
            final_loss = kd_loss
        else:
            raise ValueError("No loss was computed")

        metrics: dict[str, torch.Tensor] = {
            "loss_sum": final_loss.detach().clone(),
            "loss_total": torch.tensor(1.0, device=final_loss.device),
        }
        if ce_loss is not None:
            metrics["ce_loss_sum"] = ce_loss.detach().clone()
            metrics["ce_loss_total"] = torch.tensor(1.0, device=final_loss.device)
        if kd_loss is not None:
            metrics["kd_loss_sum"] = kd_loss.detach().clone()
            metrics["kd_loss_total"] = torch.tensor(1.0, device=final_loss.device)

        return None, final_loss, metrics

    @classmethod
    def from_training_args(
        cls,
        verifier_config: PretrainedConfig,
        t2d: torch.Tensor | None = None,  # noqa: ARG003
        d2t: torch.Tensor | None = None,  # noqa: ARG003
        **kwargs: Any,
    ) -> "Pard2DraftModel":
        if t2d is not None or d2t is not None:
            raise ValueError("PARD-2 does not use draft vocabulary mappings")

        draft_name_or_path = kwargs.get("draft_name_or_path")
        if not draft_name_or_path:
            raise ValueError("--draft-name-or-path is required for pard2 training")

        verifier_name_or_path = kwargs["verifier_name_or_path"]
        target_layer_ids = resolve_target_layer_ids(
            kwargs.get("target_layer_ids"),
            verifier_name_or_path,
        )
        target_feat_dim = verifier_config.hidden_size * len(target_layer_ids)

        mask_token_ids = kwargs.get("mask_token_ids") or []
        para_num = int(kwargs.get("para_num", 16))
        if len(mask_token_ids) < para_num - 1:
            raise ValueError(
                f"Need at least {para_num - 1} mask token ids, got {len(mask_token_ids)}"
            )

        config = Pard2SpeculatorConfig(
            draft_name_or_path=draft_name_or_path,
            para_num=para_num,
            down_sample_ratio=float(kwargs.get("down_sample_ratio", 0.7)),
            down_sample_ratio_min=float(kwargs.get("down_sample_ratio_min", 0.1)),
            mask_token_ids=list(mask_token_ids),
            feat_scale=float(kwargs.get("feat_scale", 0.02)),
            proj_bias=bool(kwargs.get("proj_bias", False)),
            target_layer_ids=[int(x) for x in target_layer_ids],
            target_feat_dim=target_feat_dim,
            target_feat_mask=float(kwargs.get("target_feat_mask", 0.2)),
            ce_alpha=float(kwargs.get("ce_alpha", 0.1)),
            kd_alpha=float(kwargs.get("kd_alpha", 1.0)),
            kd_temperature=float(kwargs.get("kd_temperature", 1.0)),
            prev_prob_loss=bool(kwargs.get("prev_prob_loss", True)),
            end_token_id=kwargs.get("end_token_id"),
            pard_token=int(mask_token_ids[0]) if mask_token_ids else -1,
            speculators_config=SpeculatorsConfig(
                algorithm="pard2",
                proposal_methods=[
                    GreedyTokenProposalConfig(
                        speculative_tokens=para_num,
                    )
                ],
                default_proposal_method="greedy",
                verifier=VerifierConfig.from_config(
                    verifier_config,
                    name_or_path=verifier_name_or_path,
                ),
            ),
        )
        return cls(config=config)

    @staticmethod
    def get_trainer_kwargs(**kwargs: Any) -> tuple[dict, dict]:
        del kwargs
        return {}, {}
