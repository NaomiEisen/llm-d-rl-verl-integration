#!/usr/bin/env bash
# GRPO | Qwen3-4B | FSDP training | NVIDIA GPUs or Ascend NPUs
#
# INFER_BACKEND controls rollout backend: vllm

set -xeuo pipefail

# ---- user-adjustable ----
# DEVICE is auto-detected by probing torch_npu; override only for special cases.
DEVICE=${DEVICE:-$(python3 -c 'import torch_npu' 2>/dev/null && echo npu || echo gpu)}
INFER_BACKEND=${INFER_BACKEND:-vllm-llmd-pd}
MODEL_PATH=${MODEL_PATH:-/tmp/verl/models/Qwen3-4B}
TRAIN_FILE=${TRAIN_FILE:-/tmp/verl/data/gsm8k/train.parquet}
TEST_FILE=${TEST_FILE:-/tmp/verl/data/gsm8k/test.parquet}
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-512}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-256}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-2}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}

ACTOR_LR=${ACTOR_LR:-1e-6}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}
ENTROPY_COEFF=${ENTROPY_COEFF:-0}

ROLLOUT_TP=${ROLLOUT_TP:-2}
PREFILL_REPLICAS=${PREFILL_REPLICAS:-2}
DECODE_REPLICAS=${DECODE_REPLICAS:-2}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.2}
ROLLOUT_N=${ROLLOUT_N:-5}

PROJECT_NAME=${PROJECT_NAME:-verl_grpo_gsm8k_examples_pd}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_4b_grpo_vllm_pd_fsdp_8gpu}
SAVE_FREQ=${SAVE_FREQ:--1}
TEST_FREQ=${TEST_FREQ:-5}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-15}
MAX_STEPS=${MAX_STEPS:-80}

GENERATIONS_ROOT=${GENERATIONS_ROOT:-/tmp/verl/generations}
VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-${GENERATIONS_ROOT}/val}
ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-${GENERATIONS_ROOT}/train}

# ---- end user-adjustable ----

case "${DEVICE}" in
    gpu)
        ;;
    npu)
        export VLLM_USE_V1=1
        export TASK_QUEUE_ENABLE=2
        export CPU_AFFINITY_CONF=1
        export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2${LD_PRELOAD:+:$LD_PRELOAD}"
        NGPUS_PER_NODE=16
        ROLLOUT_GPU_MEM_UTIL=0.9
        ;;
    *)
        echo "Unsupported DEVICE=${DEVICE}. Expected 'gpu' or 'npu'." >&2
        exit 1
        ;;
esac

########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    data.train_files=${TRAIN_FILE}
    data.val_files=${TEST_FILE}
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.truncation='error'
    algorithm.use_kl_in_reward=False
)

MODEL=(
    actor_rollout_ref.model.path=${MODEL_PATH}
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF}
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=3000
    actor_rollout_ref.actor.use_dynamic_bsz=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.name=${INFER_BACKEND}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.enable_chunked_prefill=False
    actor_rollout_ref.rollout.enforce_eager=True
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    # P/D layout — do NOT set disaggregation.enabled=True; verl only allows that
    # for sglang. Our PDEngineReplicaFactory is registered as "vllm-llmd-pd" and
    # expects num_replicas = (prefill + decode), which falls out naturally from
    # world_size / tp_size when enabled=False.
    actor_rollout_ref.rollout.disaggregation.prefill_replicas=${PREFILL_REPLICAS}
    actor_rollout_ref.rollout.disaggregation.decode_replicas=${DECODE_REPLICAS}
    # NIXL KV transfer
    +actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config.kv_connector=NixlConnector
    +actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config.kv_role=kv_both
    +actor_rollout_ref.rollout.engine_kwargs.vllm.no_disable_hybrid_kv_cache_manager=true
)

REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192
)

TRAINER=(
    trainer.critic_warmup=0
    trainer.logger='["console","file"]'
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.default_local_dir=/tmp/checkpoints
    trainer.validation_data_dir=${VALIDATION_DATA_DIR}
    trainer.rollout_data_dir=${ROLLOUT_DATA_DIR}
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.total_epochs=${TOTAL_EPOCHS}
    trainer.total_training_steps=${MAX_STEPS}
)

EXTRA=(
    '+ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_ROOT=/tmp/verl/logs'
    # --- llm-d EPP router integration (PD mode) ---
    # register_pd patches _ROLLOUT_REGISTRY in every FSDP worker before get_rollout_class() runs
    +actor_rollout_ref.model.external_lib=llm_d_rl_verl_integration.epp_router.register_pd
    +actor_rollout_ref.rollout.agent.agent_loop_manager_class=llm_d_rl_verl_integration.epp_router.agent_loop_manager.EPPAgentLoopManager
    +actor_rollout_ref.rollout.custom.epp_config_file=/tmp/llm-d-rl-verl-integration/llm_d_rl_verl_integration/shared/epp-example-config-pd.yaml
    +actor_rollout_ref.rollout.custom.epp_endpoints_file=/tmp/epp-endpoints.yaml
    +actor_rollout_ref.rollout.custom.sidecar_connector=nixlv2
    # +actor_rollout_ref.rollout.custom.epp_grpc_port=9002  # default 9002
    # ---
    actor_rollout_ref.rollout.disable_log_stats=False
    actor_rollout_ref.rollout.enable_prefix_caching=True
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.enable_prompt_tokens_details=true'
)

########################### launch ###########################
python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@"
