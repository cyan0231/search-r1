#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
OUT=$(mktemp -d "$ROOT/results/test-grpo-step720-XXXXXXXX")
RUN="$ROOT/runs/grpo-20261120-144806"
ACTOR="$ROOT/checkpoints/grpo-20261120-144806/global_step_720/actor"
MODEL="$OUT/model"
PY=$(command -v python)
export OUT MODEL HF_HUB_OFFLINE=1 HF_HOME="$ROOT/.cache/huggingface" TMPDIR="$ROOT/tmp" CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
mkdir -p "$TMPDIR"
pids=()
cleanup() {
  rc=$?
  trap - EXIT
  for pid in "${pids[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done
  for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
  echo "exit=$rc $(date -Is)"
  exit "$rc"
}
trap cleanup EXIT
wait_ready() {
  local pid=$1 url=$2 i
  for ((i=0;i<240;i++)); do
    kill -0 "$pid" || return 1
    if curl --noproxy '*' -fsS --max-time 4 "$url" -o /dev/null 2>/dev/null; then return 0; fi
    sleep 5
  done
  return 1
}
echo "merge $(date -Is)"
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$ROOT/vendor/verl:$RUN" "$PY" -m verl.model_merger merge --backend fsdp --local_dir "$ACTOR" --target_dir "$MODEL"
echo "retrieval $(date -Is)"
setsid bash "$RUN/retrieval.sh" >"$OUT/retrieval.log" 2>&1 &
pids+=("$!")
wait_ready "$!" http://127.0.0.1:8000/docs
gpus=(0 2 3 4)
for i in "${!gpus[@]}"; do
  gpu=${gpus[i]}
  port=$((8001+i))
  setsid env CUDA_VISIBLE_DEVICES="$gpu" VLLM_CACHE_ROOT="$ROOT/.cache/vllm-gpu$gpu" TORCHINDUCTOR_CACHE_DIR="$ROOT/.cache/torchinductor-gpu$gpu" "$PY" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --served-model-name qwen-search --host 127.0.0.1 --port "$port" --dtype half --tensor-parallel-size 1 --max-model-len 8192 --max-num-seqs 32 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.90 --generation-config vllm --enable-prefix-caching --disable-log-requests --disable-uvicorn-access-log >"$OUT/gpu$gpu.log" 2>&1 &
  pids+=("$!")
done
for i in "${!gpus[@]}"; do wait_ready "${pids[i+1]}" "http://127.0.0.1:$((8001+i))/health"; done
echo "evaluate $(date -Is)"
"$PY" -u - <<'PY'
import asyncio, os, sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'scripts'))
import eval_baseline_multi as ev
ev.PORTS=[8001,8002,8003,8004]
ev.MODEL_PATH=Path(os.environ['MODEL'])
ev.OUTPUT_DIR=Path(os.environ['OUT'])
asyncio.run(ev.main())
PY
