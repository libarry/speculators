#!/bin/bash
# PARD-2 最小端到端离线训练脚本（Qwen3-8B verifier，1 条样本）
#
# 流程：准备数据 → 启动 vLLM → 离线生成 hidden states → 停止 vLLM → 训练 → 导出
#
# 用法（在 speculators 仓库根目录）：
#   bash examples/train/pard2_qwen3_8b_minimal_1sample_offline.sh
#
# 要求：已安装 speculators、vLLM（含 hidden-states 提取）、多卡 NPU/GPU 可加载 Qwen3-8B（FSDP 分片）。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# ============ 配置 ============
VERIFIER="/data/models/qwen/Qwen3-8B"              # 目标 verifier（训练监督模型）
DRAFT="/home/libowen/Qwen3-0.6B/"               # PARD-2 draft 底座（与 PARD 官方 Qwen3 配置一致）
DATA_FILE="$SCRIPT_DIR/../data/pard2_minimal_1sample.jsonl"
OUTPUT_DIR="./output/pard2_qwen3_8b_minimal"
HIDDEN_STATES_DIR="$OUTPUT_DIR/hidden_states"
VLLM_PORT=8118
# 训练用多卡 FSDP 分片；vLLM 阶段仍可用子集（见下方注释块）
GPUS="0,1,2,3"
NUM_GPUS=4
MAX_SAMPLES=1
SEQ_LENGTH=128
EPOCHS=1
LR=3e-5

# PARD-2 超参（对齐 PARD 官方 example_pard2_qwen3.yaml — general / loss）
PARD_TARGET_LAYER_IDS=(-1 -8 -16 -24)
PARA_NUM=16
DOWN_SAMPLE_RATIO=0.7
DOWN_SAMPLE_RATIO_MIN=0.1
FEAT_SCALE=0.02
TARGET_FEAT_MASK=0.1
CE_ALPHA=0.1
KD_ALPHA=1.0
# collator（example_pard2_qwen3.yaml）
END_TOKEN_ID=151644
MASK_TOKEN_ID=151670
# train（example_pard2_qwen3.yaml；下方 SEQ_LENGTH/EPOCHS 为 smoke 用，正式训练改为 1024/4）
SCHEDULER_TYPE="cosine_with_min_lr"
SCHEDULER_MIN_LR_RATE=0.1
SCHEDULER_WARMUP_RATIO=0.03
PER_DEVICE_TRAIN_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=2
DATALOADER_NUM_WORKERS=4
# =======================================

# 将 PARD 的负向层索引转为 vLLM eagle_aux_hidden_state_layer_ids（1-based）
VLLM_TARGET_LAYER_IDS="$(
python - <<'PY'
from transformers import AutoConfig

verifier = "/data/models/qwen/Qwen3-8B"
pard_layers = [-1, -8, -16, -24]
cfg = AutoConfig.from_pretrained(verifier)
if hasattr(cfg, "text_config"):
    cfg = cfg.text_config
n = int(cfg.num_hidden_layers)
# hidden_states 长度为 n+1；索引 i<0 时 1-based 层号 = n + 1 + i
ids = [n + 1 + i if i < 0 else i for i in pard_layers]
print(" ".join(str(x) for x in ids))
PY
)"

echo "Verifier: $VERIFIER"
echo "Draft:    $DRAFT"
echo "Output:   $OUTPUT_DIR"
echo "Train:    $NUM_GPUS NPUs ($GPUS)"
echo "vLLM target layers (1-based): $VLLM_TARGET_LAYER_IDS"

# ---------------------------------------------------------------------------
# 阶段 1：数据预处理
# 对原始对话做 chat template、tokenize，并生成 loss_mask / token_freq
# ---------------------------------------------------------------------------
# echo "=== 阶段 1/6：数据预处理 ==="
# python scripts/prepare_data.py \
#     --model "$VERIFIER" \
#     --data "$DATA_FILE" \
#     --output "$OUTPUT_DIR" \
#     --max-samples "$MAX_SAMPLES" \
#     --seq-length "$SEQ_LENGTH" \
#     --num-preprocessing-workers 1 \
#     --overwrite

# # ---------------------------------------------------------------------------
# # 阶段 2：启动 vLLM（hidden states 提取模式）
# # target-layer-ids 必须与训练时 --target-layer-ids 对应层一致
# # ---------------------------------------------------------------------------
# echo "=== 阶段 2/6：启动 vLLM 服务 ==="
# ASCEND_RT_VISIBLE_DEVICES="$GPUS" python scripts/launch_vllm.py "$VERIFIER" \
#     --hidden-states-path "$HIDDEN_STATES_DIR" \
#     --target-layer-ids $VLLM_TARGET_LAYER_IDS \
#     --no-include-last-layer \
#     -- --port "$VLLM_PORT" --gpu-memory-utilization 0.90 &
# VLLM_PID=$!

# cleanup() {
#     echo "停止 vLLM 服务..."
#     kill "$VLLM_PID" 2>/dev/null || true
#     wait "$VLLM_PID" 2>/dev/null || true
# }
# trap cleanup EXIT

# echo "等待 vLLM 就绪..."
# until curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do
#     sleep 2
# done
# echo "vLLM 已就绪。"

# # ---------------------------------------------------------------------------
# # 阶段 3：离线生成 verifier hidden states
# # 每个样本写出 hs_{idx}.safetensors，供离线训练读取
# # ---------------------------------------------------------------------------
# echo "=== 阶段 3/6：离线生成 hidden states ==="
# python scripts/data_generation_offline.py \
#     --preprocessed-data "$OUTPUT_DIR" \
#     --endpoint "http://localhost:${VLLM_PORT}/v1" \
#     --output "$HIDDEN_STATES_DIR" \
#     --max-samples "$MAX_SAMPLES" \
#     --concurrency 1 \
#     --validate-outputs

# # ---------------------------------------------------------------------------
# # 阶段 4：停止 vLLM，释放显存给训练
# # ---------------------------------------------------------------------------
# echo "=== 阶段 4/6：停止 vLLM，释放 GPU ==="
# cleanup
# trap - EXIT

# ---------------------------------------------------------------------------
# 阶段 5：PARD-2 离线训练
# COD 重采样在 collate 阶段在线完成，无需额外数据准备
# ---------------------------------------------------------------------------
echo "=== 阶段 5/6：PARD-2 训练 ==="
ASCEND_RT_VISIBLE_DEVICES="$GPUS" torchrun \
    --standalone --nproc_per_node "$NUM_GPUS" \
    scripts/train.py \
    --speculator-type pard2 \
    --draft-name-or-path "$DRAFT" \
    --verifier-name-or-path "$VERIFIER" \
    --target-layer-ids "${PARD_TARGET_LAYER_IDS[@]}" \
    --para-num "$PARA_NUM" \
    --down-sample-ratio "$DOWN_SAMPLE_RATIO" \
    --down-sample-ratio-min "$DOWN_SAMPLE_RATIO_MIN" \
    --feat-scale "$FEAT_SCALE" \
    --target-feat-mask "$TARGET_FEAT_MASK" \
    --ce-alpha "$CE_ALPHA" \
    --kd-alpha "$KD_ALPHA" \
    --end-token-id "$END_TOKEN_ID" \
    --mask-token-id "$MASK_TOKEN_ID" \
    --data-path "$OUTPUT_DIR" \
    --hidden-states-path "$HIDDEN_STATES_DIR" \
    --save-path "$OUTPUT_DIR/checkpoints" \
    --total-seq-len "$SEQ_LENGTH" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --scheduler-type "$SCHEDULER_TYPE" \
    --scheduler-min-lr-rate "$SCHEDULER_MIN_LR_RATE" \
    --scheduler-warmup-ratio "$SCHEDULER_WARMUP_RATIO" \
    --per-device-train-batch-size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
    --num-workers "$DATALOADER_NUM_WORKERS" \
    --on-missing raise

# ---------------------------------------------------------------------------
# 阶段 6：导出为 PARD/vLLM 推理目录结构
# ---------------------------------------------------------------------------
echo "=== 阶段 6/6：导出 checkpoint ==="
CKPT_DIR="$OUTPUT_DIR/checkpoints/0"
if [[ ! -d "$CKPT_DIR" ]]; then
    CKPT_DIR="$(find "$OUTPUT_DIR/checkpoints" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
fi
python scripts/export_pard2_checkpoint.py --checkpoint-dir "$CKPT_DIR"

echo "完成。"
echo "  训练数据:     $OUTPUT_DIR"
echo "  Hidden states: $HIDDEN_STATES_DIR"
echo "  Checkpoint:   $OUTPUT_DIR/checkpoints"
echo "  推理导出:     $CKPT_DIR/pard_model/ 与 $CKPT_DIR/warp_model.bin"
