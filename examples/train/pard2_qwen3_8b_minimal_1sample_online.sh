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

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# ============ 配置 ============
VERIFIER="/data/models/qwen/Qwen3-8B"              # 目标 verifier（训练监督模型）
DRAFT="/home/libowen/Qwen3-0.6B/"                # PARD-2 draft 底座
DATA_FILE="$SCRIPT_DIR/../../../data/magpie_qwen25_pro.jsonl"
OUTPUT_DIR="./output/pard2_qwen3_8b_minimal_online"
HIDDEN_STATES_DIR="/dev/shm/pard2_hs"      # vLLM 中转目录；cache 模式下持久缓存
VLLM_PORT=8119
MAX_SAMPLES=1000
SEQ_LENGTH=8192
EPOCHS=2
LR=3e-5

# vLLM 推理：物理卡 5,6，TP=2（可见设备数须等于 VLLM_TP）
VLLM_GPUS="4,5,6,7"
VLLM_TP=4
# Ascend 上 ACL graph capture 额外占显存；OOM 时可开 eager 并降低 utilization
VLLM_GPU_MEMORY_UTIL=0.90
VLLM_ENFORCE_EAGER=1          # 1=禁用 NPU graph，避免 capture 阶段 OOM
VLLM_MAX_MODEL_LEN="$SEQ_LENGTH"
# 训练：与 vLLM 卡不重叠
TRAIN_GPUS="0,1,2,3"
NUM_TRAIN_GPUS=4
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

# PARD-2 超参（对齐 PARD 官方 example_pard2_qwen3.yaml）
PARD_TARGET_LAYER_IDS=(-1 -8 -16 -24)
PARA_NUM=16
DOWN_SAMPLE_RATIO=0.7
DOWN_SAMPLE_RATIO_MIN=0.1
FEAT_SCALE=0.02
TARGET_FEAT_MASK=0.1
CE_ALPHA=0.1
KD_ALPHA=1.0
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
echo "vLLM GPUs:    $VLLM_GPUS (TP=$VLLM_TP)"
echo "Train GPUs:   $TRAIN_GPUS (x$NUM_TRAIN_GPUS)"
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
    --num-preprocessing-workers 1 \
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
    --max-model-len "$VLLM_MAX_MODEL_LEN" \
    --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTIL" \
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
    --data-path "$OUTPUT_DIR" \
    --hidden-states-path "$HIDDEN_STATES_DIR" \
    --vllm-endpoint "http://localhost:${VLLM_PORT}/v1" \
    --on-missing "$ON_MISSING" \
    --on-generate "$ON_GENERATE" \
    --save-path "$OUTPUT_DIR/checkpoints" \
    --total-seq-len "$SEQ_LENGTH" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --scheduler-type cosine \
    --num-workers 1

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
