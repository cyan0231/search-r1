#!/usr/bin/env bash
set -eu

PROJECT=$(cd "$(dirname "$0")/.." && pwd)
PYTHON=$(command -v python)
GPUS=(0 2 3 4 5)
PORTS=(8001 8002 8003 8004 8005)

mkdir -p "$PROJECT/logs" "$PROJECT/tmp"

for i in "${!GPUS[@]}"; do
    gpu=${GPUS[$i]}
    port=${PORTS[$i]}
    nohup env \
        CUDA_DEVICE_ORDER=PCI_BUS_ID \
        CUDA_VISIBLE_DEVICES="$gpu" \
        OMP_NUM_THREADS=4 \
        HF_HUB_OFFLINE=1 \
        HF_HOME="$PROJECT/.cache/huggingface" \
        VLLM_CACHE_ROOT="$PROJECT/.cache/vllm-gpu$gpu" \
        TORCHINDUCTOR_CACHE_DIR="$PROJECT/.cache/torchinductor-gpu$gpu" \
        TMPDIR="$PROJECT/tmp" \
        "$PYTHON" -m vllm.entrypoints.openai.api_server \
        --model "$PROJECT/models/Qwen2.5-3B-Instruct" \
        --served-model-name qwen-search \
        --host 127.0.0.1 \
        --port "$port" \
        --dtype half \
        --tensor-parallel-size 1 \
        --max-model-len 8192 \
        --max-num-seqs 32 \
        --max-num-batched-tokens 8192 \
        --gpu-memory-utilization 0.90 \
        --generation-config vllm \
        --enable-prefix-caching \
        --disable-log-requests \
        --disable-uvicorn-access-log \
        > "$PROJECT/logs/vllm-gpu$gpu.log" 2>&1 < /dev/null &
    echo "$!" > "$PROJECT/logs/vllm-gpu$gpu.pid"
    echo "已启动 GPU $gpu 的服务进程，端口 $port；启动结果见 $PROJECT/logs/vllm-gpu$gpu.log"
done
