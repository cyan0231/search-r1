#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
mkdir -p tmp
if curl --noproxy '*' -fsS http://127.0.0.1:8000/docs -o /dev/null 2>/dev/null; then
    echo "检索端口 8000 已就绪"
    exit 0
fi
exec 9>tmp/training-retrieval.lock
flock -n 9 || { echo "检索进程正在启动"; exit 1; }
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1 HF_HOME="$PWD/.cache/huggingface" TMPDIR="$PWD/tmp"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
lib=$(python -c 'import sysconfig; print(sysconfig.get_path("purelib") + "/nvidia")')
export LD_LIBRARY_PATH="$lib/cuda_runtime/lib:$lib/cublas/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$RETRIEVAL_PYTHON" -u search/retrieval_http_gpu.py
