#!/bin/bash
# PARD-2 最小端到端在线训练脚本（Qwen3-8B verifier，1 条样本）
#
# 流程：准备数据 → 启动 vLLM → 在线训练（按需生成 hidden states，默认不落盘）→ 导出
#
# 用法（在 speculators 仓库根目录）：
#   bash examples/train/pard2_qwen3_8b_minimal_1sample_online.sh
#
# 要求：
#   - 已安装 speculators、vLLM（含 hidden-states 提取）
#   - vLLM 与训练进程需能访问同一 hidden states 中转目录（同机或共享存储）
#   - 在线训练建议 vLLM 与训练使用不同 NPU/GPU（见下方 VLLM_GPUS / TRAIN_GPUS）
#
# 混合模式（首 epoch 在线生成并缓存，后续 epoch 读盘）：
#   将 ON_GENERATE 改为 cache，多 epoch 时可在第 2 个 epoch 起停掉 vLLM 以省显存。
###############避免通信超时##########
export HCCL_CONNECT_TIMEOUT=1800
export HCCL_EXEC_TIMEOUT=1800
##################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# ============ 配置 ============
VERIFIER="/data/models/qwen/Qwen3-8B"              # 目标 verifier（训练监督模型）
DRAFT="/home/libowen/Qwen3-0.6B/"                # PARD-2 draft 底座
DATA_FILE="$SCRIPT_DIR/../../../data/metamath_qwen3_8b.jsonl"
OUTPUT_DIR="./output/pard2_qwen3_8b_minimal_online"
HIDDEN_STATES_DIR="/dev/shm/pard2_hs"      # vLLM 中转目录；cache 模式下持久缓存
VLLM_PORT=8119
MAX_SAMPLES=100000
SEQ_LENGTH=2048
VLLM_SEQ_LENGTH=16384
EPOCHS=4
LR=3e-5

# vLLM 推理：可见设备数须等于 VLLM_TP * VLLM_DP（多副本并发处理 hidden states 请求）
VLLM_GPUS="0,1"
VLLM_TP=1
VLLM_DP=2
# Ascend 上 ACL graph capture 额外占显存；OOM 时可开 eager 并降低 utilization
VLLM_GPU_MEMORY_UTIL=0.90
VLLM_ENFORCE_EAGER=0          # 1=禁用 NPU graph；0=开 graph，配合下方 6 个 capture size
# 默认会 capture 51 张图；hidden states 单请求场景 6 个即可，显著降低 capture 显存
VLLM_COMPILATION_CONFIG='{"cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32]}'
VLLM_MAX_MODEL_LEN="$VLLM_SEQ_LENGTH"
# 训练：与 vLLM 卡不重叠
TRAIN_GPUS="2,4,5,6"
NUM_TRAIN_GPUS=4
# DataLoader 预取：worker 在训练当前 batch 时提前向 vLLM 发下一批请求
DATALOADER_NUM_WORKERS=2
DATALOADER_PREFETCH_FACTOR=4
DEVICE_ENV_PREFIX="ASCEND_RT_VISIBLE_DEVICES"      # NVIDIA 环境改为 CUDA_VISIBLE_DEVICES

run_on_devices() {
    local devices="$1"
    shift
    env "${DEVICE_ENV_PREFIX}=${devices}" "$@"
}

# 在线 hidden states 策略
#   delete — 纯在线，加载后删除，几乎不占磁盘（推荐）
#   cache  — 混合模式，写入 HIDDEN_STATES_DIR，后续 epoch 复用
ON_MISSING="generate"
ON_GENERATE="delete"

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
# train（example_pard2_qwen3.yaml）
SCHEDULER_TYPE="cosine_with_min_lr"
SCHEDULER_MIN_LR_RATE=0.1
SCHEDULER_WARMUP_RATIO=0.03
PER_DEVICE_TRAIN_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=1
# =======================================

# 将 PARD 的负向层索引转为 vLLM eagle_aux_hidden_state_layer_ids（1-based）
VLLM_TARGET_LAYER_IDS="$(
python - <<PY
from transformers import AutoConfig

verifier = "${VERIFIER}"
pard_layers = [$(IFS=,; echo "${PARD_TARGET_LAYER_IDS[*]// /,}")]
cfg = AutoConfig.from_pretrained(verifier)
if hasattr(cfg, "text_config"):
    cfg = cfg.text_config
n = int(cfg.num_hidden_layers)
ids = [n + 1 + i if i < 0 else i for i in pard_layers]
print(" ".join(str(x) for x in ids))
PY
)"

echo "Verifier:     $VERIFIER"
echo "Draft:        $DRAFT"
echo "Output:       $OUTPUT_DIR"
_vllm_gpu_count="$(echo "$VLLM_GPUS" | tr ',' '\n' | wc -l)"
if [[ "$_vllm_gpu_count" -ne $((VLLM_TP * VLLM_DP)) ]]; then
    echo "错误: VLLM_GPUS 数量 ($_vllm_gpu_count) 须等于 VLLM_TP * VLLM_DP ($((VLLM_TP * VLLM_DP)))" >&2
    exit 1
fi
echo "vLLM GPUs:    $VLLM_GPUS (TP=$VLLM_TP, DP=$VLLM_DP)"
echo "vLLM graphs:  $VLLM_COMPILATION_CONFIG (enforce_eager=$VLLM_ENFORCE_EAGER)"
echo "Train GPUs:   $TRAIN_GPUS (x$NUM_TRAIN_GPUS)"
echo "DataLoader:   num_workers=$DATALOADER_NUM_WORKERS, prefetch_factor=$DATALOADER_PREFETCH_FACTOR"
echo "Online mode:  on_missing=$ON_MISSING, on_generate=$ON_GENERATE"
echo "vLLM layers:  $VLLM_TARGET_LAYER_IDS (1-based)"

mkdir -p "$HIDDEN_STATES_DIR"

# ---------------------------------------------------------------------------
# 阶段 1：数据预处理
# ---------------------------------------------------------------------------
echo "=== 阶段 1/4：数据预处理 ==="
python scripts/prepare_data.py \
    --model "$VERIFIER" \
    --data "$DATA_FILE" \
    --output "$OUTPUT_DIR" \
    --max-samples "$MAX_SAMPLES" \
    --seq-length "$SEQ_LENGTH" \
    --num-preprocessing-workers 64 \
    --overwrite

# ---------------------------------------------------------------------------
# 阶段 2：启动 vLLM（hidden states 提取，训练期间保持运行）
# ---------------------------------------------------------------------------
echo "=== 阶段 2/4：启动 vLLM 服务 ==="
run_on_devices "$VLLM_GPUS" python scripts/launch_vllm.py "$VERIFIER" \
    --hidden-states-path "$HIDDEN_STATES_DIR" \
    --target-layer-ids $VLLM_TARGET_LAYER_IDS \
    --no-include-last-layer \
    -- \
    --port "$VLLM_PORT" \
    --tensor-parallel-size "$VLLM_TP" \
    --data-parallel-size "$VLLM_DP" \
    --max-model-len "$VLLM_MAX_MODEL_LEN" \
    --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTIL" \
    --compilation-config "$VLLM_COMPILATION_CONFIG" \
    $([ "$VLLM_ENFORCE_EAGER" = "1" ] && echo --enforce-eager) &
VLLM_PID=$!

cleanup() {
    echo "停止 vLLM 服务..."
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "等待 vLLM 就绪..."
until curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do
    sleep 2
done
echo "vLLM 已就绪。"

# ---------------------------------------------------------------------------
# 阶段 3：PARD-2 在线训练
# hidden states 由 vLLM 按需生成；COD 重采样仍在 collate 阶段完成
# ---------------------------------------------------------------------------
##############debug#################
export ASCEND_LAUNCH_BLOCKING=1
##################################

echo "=== 阶段 3/4：PARD-2 在线训练 ==="
run_on_devices "$TRAIN_GPUS" torchrun \
    --standalone --nproc_per_node "$NUM_TRAIN_GPUS" \
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
    --vllm-endpoint "http://localhost:${VLLM_PORT}/v1" \
    --on-missing "$ON_MISSING" \
    --on-generate "$ON_GENERATE" \
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
    --prefetch-factor "$DATALOADER_PREFETCH_FACTOR"

# ---------------------------------------------------------------------------
# 阶段 4：导出为 PARD/vLLM 推理目录结构
# ---------------------------------------------------------------------------
echo "=== 阶段 4/4：导出 checkpoint ==="
cleanup
trap - EXIT

CKPT_DIR="$OUTPUT_DIR/checkpoints/0"
if [[ ! -d "$CKPT_DIR" ]]; then
    CKPT_DIR="$(find "$OUTPUT_DIR/checkpoints" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
fi
python scripts/export_pard2_checkpoint.py --checkpoint-dir "$CKPT_DIR"

echo "完成。"
echo "  训练数据:      $OUTPUT_DIR"
echo "  HS 中转/缓存:  $HIDDEN_STATES_DIR"
echo "  Checkpoint:    $OUTPUT_DIR/checkpoints"
echo "  推理导出:      $CKPT_DIR/pard_model/ 与 $CKPT_DIR/warp_model.bin"
