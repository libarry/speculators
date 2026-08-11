#!/bin/bash
# Ascend online DSpark (MLA): train a GLM5.2-matched MLA draft against a live
# Mooncake-backed vLLM server.
#
# MLA dims are taken from /home/libowen/GLM5.2/config.json so the draft attention
# layout matches the verifier:
#   q_lora_rank=2048, kv_lora_rank=512,
#   qk_nope_head_dim=192, qk_rope_head_dim=64, v_head_dim=256
#   num_attention_heads=64, hidden_size=6144
#   rope_parameters: rope_theta=8000000, rope_type=default
#
# Verifier layer capture (same Plan A as glm52_train.sh):
#   VLLM_EXTRACT_LAYER_IDS="1 2 3 4 5" -> planes [L1..L5]
#   TARGET_LAYER_IDS="1 2 3 4 5"       -> fc uses L1..L5; last plane is verifier
#
# Prerequisites:
#   1. mooncake_master --port 50051 (on the infer node)
#   2. vLLM with Mooncake HS backend up on VLLM_HOST:VLLM_PORT
#      (pair: bash examples/train/glm52_infer.sh)
#   3. TRAIN_GPUS must NOT overlap VLLM_GPUS (when co-located)
#
# Dual-machine example (infer=192.168.9.129, train=this host):
#   VLLM_HOST=192.168.9.129 MOONCAKE_MASTER=192.168.9.129:50051 \
#     bash examples/train/glm52_mla_train.sh
#
# Usage (from repo root `speculators/`):
#   bash examples/train/glm52_mla_train.sh

set -euo pipefail

# ============ Configuration ============
MODEL="/home/libowen/GLM5.2"
DATASET="/home/libowen/spec/data/metamath_qwen3_8b.jsonl"
OUTPUT_DIR="./output/dspark_glm52_mla"

# GPU assignments (online training needs separate GPUs for vLLM and training)
TRAIN_GPUS="${TRAIN_GPUS:-6,7}"
NUM_TRAIN_GPUS="${NUM_TRAIN_GPUS:-2}"

# Draft fc aux layers (must match vLLM planes [:, :-1] after data.py split)
TARGET_LAYER_IDS="1 2 3 4 5"
MAX_SAMPLES=100

# Infer node address for HTTP + Mooncake master (dual-machine: set explicitly).
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8005}"

SEQ_LENGTH=2048
EPOCHS=5
LR=3e-4

# DSpark + MLA (dims match GLM5.2 config.json)
SPECULATOR_TYPE="dspark"
ATTENTION_TYPE="mla"
DRAFT_ATTN_IMPL="sdpa"   # simple_flex_attention | sdpa | eager (use sdpa/eager on Ascend NPU)
BLOCK_SIZE=8
MAX_ANCHORS=256
NUM_LAYERS=3
DRAFT_VOCAB_SIZE=32000

# GLM5.2 MLA layout (GlmMoeDsaConfig)
Q_LORA_RANK=2048
KV_LORA_RANK=512
QK_NOPE_HEAD_DIM=192
QK_ROPE_HEAD_DIM=64
V_HEAD_DIM=256

# Markov + confidence head settings
MARKOV_RANK=256
MARKOV_HEAD_TYPE="vanilla"   # vanilla | gated | rnn
LOSS_FN='{"ce": 0.1, "tv": 0.9}'
CONFIDENCE_HEAD_ALPHA=1.0

# THIS train node's routable IP (must differ from the infer node's IP).
HOST_IP="${HOST_IP:-$(hostname -I | awk '{print $1}')}"
export MOONCAKE_LOCAL_HOSTNAME="${MOONCAKE_LOCAL_HOSTNAME:-$HOST_IP}"

# Must match vLLM side protocol; master points at the infer node when remote.
MOONCAKE_PROTOCOL="${MOONCAKE_PROTOCOL:-ascend}"
MOONCAKE_MASTER="${MOONCAKE_MASTER:-${VLLM_HOST}:50051}"

# Root-cause fix for Ascend ADXL Transfer slice failed status=503900
export HCCL_INTRA_ROCE_ENABLE="${HCCL_INTRA_ROCE_ENABLE:-1}"
export HCCL_WHITELIST_DISABLE="${HCCL_WHITELIST_DISABLE:-1}"
export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-16667-16700}"
export ASCEND_CONNECT_TIMEOUT="${ASCEND_CONNECT_TIMEOUT:-10000}"
export ASCEND_TRANSFER_TIMEOUT="${ASCEND_TRANSFER_TIMEOUT:-20000}"
export ASCEND_BUFFER_POOL="${ASCEND_BUFFER_POOL:-4:8}"

# Ascend / training environment
export HCCL_OP_EXPANSION_MODE="AIV"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
# =======================================

# Step 1: Prepare data
echo "=== Step 1: Preparing data ==="
python scripts/prepare_data.py \
    --model "$MODEL" \
    --data "$DATASET" \
    --output "$OUTPUT_DIR" \
    --max-samples "$MAX_SAMPLES" \
    --seq-length "$SEQ_LENGTH" \
    --assistant-pattern '<\|assistant\|>((?:(?!<\|user\|>|<\|assistant\|>).)*)'

echo "Waiting for vLLM server to be ready at ${VLLM_HOST}:${VLLM_PORT}..."
until curl -sf "http://${VLLM_HOST}:${VLLM_PORT}/health" > /dev/null 2>&1; do
    sleep 2
done
echo "vLLM server ready."

# Step 2: Train MLA DSpark against the live Mooncake-backed vLLM server
echo "=== Step 2: Training (GLM5.2 MLA DSpark / Mooncake/Ascend) ==="
echo "MOONCAKE_LOCAL_HOSTNAME=${MOONCAKE_LOCAL_HOSTNAME}"
echo "protocol=${MOONCAKE_PROTOCOL} master=${MOONCAKE_MASTER}"
echo "vllm-endpoint=http://${VLLM_HOST}:${VLLM_PORT}/v1"
echo "ASCEND_RT_VISIBLE_DEVICES=${TRAIN_GPUS}"
echo "target-layer-ids=${TARGET_LAYER_IDS}"
echo "attention-type=${ATTENTION_TYPE}"
echo "mla dims: q_lora=${Q_LORA_RANK} kv_lora=${KV_LORA_RANK} qk_nope=${QK_NOPE_HEAD_DIM} qk_rope=${QK_ROPE_HEAD_DIM} v=${V_HEAD_DIM}"
echo "HCCL_INTRA_ROCE_ENABLE=${HCCL_INTRA_ROCE_ENABLE}"
echo "HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE}"

ASCEND_RT_VISIBLE_DEVICES="$TRAIN_GPUS" torchrun \
    --standalone --nproc_per_node "$NUM_TRAIN_GPUS" \
    scripts/train.py \
    --verifier-name-or-path "$MODEL" \
    --data-path "$OUTPUT_DIR" \
    --vllm-endpoint "http://${VLLM_HOST}:${VLLM_PORT}/v1" \
    --save-path "$OUTPUT_DIR/checkpoints" \
    --draft-vocab-size "$DRAFT_VOCAB_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --total-seq-len "$SEQ_LENGTH" \
    --speculator-type "$SPECULATOR_TYPE" \
    --attention-type "$ATTENTION_TYPE" \
    --q-lora-rank "$Q_LORA_RANK" \
    --kv-lora-rank "$KV_LORA_RANK" \
    --qk-nope-head-dim "$QK_NOPE_HEAD_DIM" \
    --qk-rope-head-dim "$QK_ROPE_HEAD_DIM" \
    --v-head-dim "$V_HEAD_DIM" \
    --draft-attn-impl "$DRAFT_ATTN_IMPL" \
    --block-size "$BLOCK_SIZE" \
    --max-anchors "$MAX_ANCHORS" \
    --num-layers "$NUM_LAYERS" \
    --target-layer-ids $TARGET_LAYER_IDS \
    --markov-rank "$MARKOV_RANK" \
    --markov-head-type "$MARKOV_HEAD_TYPE" \
    --enable-confidence-head \
    --confidence-head-with-markov \
    --loss-fn "$LOSS_FN" \
    --confidence-head-alpha "$CONFIDENCE_HEAD_ALPHA" \
    --on-missing generate \
    --on-generate delete \
    --hidden-states-backend mooncake \
    --mooncake-master "$MOONCAKE_MASTER" \
    --mooncake-metadata-server P2PHANDSHAKE \
    --mooncake-protocol "$MOONCAKE_PROTOCOL" \
    --num-workers "${NUM_WORKERS:-4}" \
    --prefetch-factor "${PREFETCH_FACTOR:-4}" \
    --fsdp-shard

echo "Done. Checkpoints saved to $OUTPUT_DIR/checkpoints/"
