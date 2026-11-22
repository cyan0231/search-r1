#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
export PYTHONPATH="$ROOT/vendor/verl:$(dirname "$(realpath "$0")")${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
MODE=${1:-smoke}
case "$MODE" in
    smoke)
        BATCH=${BATCH:-20}
        MICRO=${MICRO:-1}
        STEPS=1
        WARMUP=0
        VALIDATE=False
        INTERVAL=-1
        BEST=False
        ;;
    full)
        BATCH=${BATCH:-40}
        MICRO=${MICRO:-2}
        STEPS=1400
        WARMUP=0.285
        VALIDATE=True
        INTERVAL=180
        BEST=True
        ;;
    *) echo 'Usage: bash scripts/train_grpo.sh smoke|full' >&2; exit 2 ;;
esac
LOG_PROB_MICRO=${LOG_PROB_MICRO:-8}
MAX_PROMPT_LENGTH=512
MAX_MODEL_LEN=$((4096 + 512))
mkdir -p "$ROOT/tmp"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,2,3,4
export VLLM_USE_V1=1 VLLM_ATTENTION_BACKEND=FLASH_ATTN
export HF_HUB_OFFLINE=1 HF_HOME="$ROOT/.cache/huggingface" TMPDIR="$ROOT/tmp"
export OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false RAY_USAGE_STATS_ENABLED=0
export RAY_TMPDIR="$ROOT/tmp" TRITON_CACHE_DIR="$ROOT/.cache/triton" XDG_CACHE_HOME="$ROOT/.cache"
export NCCL_DEBUG=WARN HYDRA_FULL_ERROR=1
curl --noproxy '*' --silent --show-error --fail http://127.0.0.1:8000/docs >/dev/null
exec python -u -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=False \
  data.train_files="$ROOT/data/grpo-search/train.parquet" data.val_files="$ROOT/data/grpo-search/validation.parquet" \
  data.return_raw_chat=True data.train_batch_size="$BATCH" data.val_batch_size=160 data.shuffle=True +data.seed=42 \
  data.dataloader_num_workers=0 data.max_prompt_length="$MAX_PROMPT_LENGTH" data.max_response_length="$((MAX_MODEL_LEN - MAX_PROMPT_LENGTH))" \
  actor_rollout_ref.model.path="$ROOT/models/Qwen2.5-3B-Instruct" actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio="$WARMUP" actor_rollout_ref.actor.optim.weight_decay=0.01 \
  actor_rollout_ref.actor.use_dynamic_bsz=False actor_rollout_ref.actor.ppo_mini_batch_size="$BATCH" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$MICRO" actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.use_torch_compile=False actor_rollout_ref.actor.entropy_checkpointing=True \
  actor_rollout_ref.actor.entropy_from_logits_with_chunking=True actor_rollout_ref.actor.use_kl_loss=False actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.rollout.mode=async actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.temperature=1 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.val_kwargs.n=1 actor_rollout_ref.rollout.val_kwargs.temperature=0 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.90 actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN" \
  actor_rollout_ref.rollout.max_num_batched_tokens=8192 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$LOG_PROB_MICRO" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$LOG_PROB_MICRO" \
  actor_rollout_ref.rollout.agent.num_workers=5 actor_rollout_ref.rollout.agent.agent_loop_config_path="$ROOT/configs/search_agent.yaml" \
  custom_reward_function.path="$ROOT/scripts/em_reward.py" \
  trainer.nnodes=1 trainer.n_gpus_per_node=4 trainer.logger='[console]' trainer.total_epochs=1 \
  trainer.total_training_steps="$STEPS" trainer.save_freq="$INTERVAL" trainer.test_freq="$INTERVAL" trainer.val_before_train="$VALIDATE" \
  +trainer.search_best_only="$BEST" trainer.resume_mode=auto \
  trainer.default_local_dir="$ROOT/checkpoints/grpo-20261120-144806" \
  ray_init.num_cpus=24 hydra.run.dir=. hydra.output_subdir=null hydra/job_logging=disabled
