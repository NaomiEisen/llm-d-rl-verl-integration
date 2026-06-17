# Running the KubeRay Example

Single-node 8-GPU GRPO training on GSM8K with Qwen3-4B using the llm-d RL verl integration.

## Prerequisites

- Kubernetes cluster with GPU nodes
- KubeRay CRD and operator installed

## Directory structure

```
examples/
  configs/
    epp-config.yaml        # EPP config — standard routing
    epp-config-pd.yaml     # EPP config — PD disaggregation
  deployments/
    configmap.yaml         # Kubernetes ConfigMap bundling both EPP configs
    ray-cluster.yaml       # RayCluster definition
```

## Step 1 — Edit the deployment manifests

In `deployments/ray-cluster.yaml`, replace:
- `<your-namespace>` — Kubernetes namespace to deploy into
- `<node-name>` — node name for the GPU node (appears twice: head and worker affinity)

## Step 2 — Deploy

```bash
# ConfigMap must exist before the cluster starts (pods mount it at /etc/llmd-configs/)
kubectl apply -f examples/deployments/configmap.yaml
kubectl apply -f examples/deployments/ray-cluster.yaml
```

Wait for both pods to be ready:
```bash
kubectl get pods -w
```

The `postStart` hook on each pod installs the integration package and pre-downloads GSM8K and Qwen3-4B (~15 min on first run). Training should not start until both pods report `Ready`.

## Step 3 — Run training

Exec into the head pod, then run one of the commands below.

```bash
kubectl exec -it <head-pod-name> -- bash
```

All commands use verl's own `run_qwen3_4b_fsdp.sh` as the base script and pass the integration overrides via `$@`. `hydra.run.dir` is required because the default `./outputs/` path is read-only in the container.

### EPP — direct gRPC routing

```bash
MODEL_PATH=/tmp/verl/models/Qwen3-4B \
TRAIN_FILE=/tmp/verl/data/gsm8k/train.parquet \
TEST_FILE=/tmp/verl/data/gsm8k/test.parquet \
SAVE_FREQ=-1 \
bash /opt/verl/examples/grpo_trainer/run_qwen3_4b_fsdp.sh \
    trainer.logger='["console","file"]' \
    trainer.default_local_dir=/tmp/checkpoints \
    trainer.total_training_steps=50 \
    '+ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_ROOT=/tmp/verl/logs' \
    +actor_rollout_ref.rollout.agent.agent_loop_manager_class=llm_d_rl_verl_integration.epp_router.agent_loop_manager.EPPAgentLoopManager \
    +actor_rollout_ref.rollout.custom.epp_config_file=/etc/llmd-configs/epp-config.yaml \
    +actor_rollout_ref.rollout.custom.epp_endpoints_file=/tmp/epp-endpoints.yaml \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.enable_prefix_caching=True \
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.enable_prompt_tokens_details=true' \
    'hydra.run.dir=/tmp/hydra-outputs'
```

### EPP — direct gRPC routing, PD disaggregated

```bash
INFER_BACKEND=vllm-llmd-pd \
MODEL_PATH=/tmp/verl/models/Qwen3-4B \
TRAIN_FILE=/tmp/verl/data/gsm8k/train.parquet \
TEST_FILE=/tmp/verl/data/gsm8k/test.parquet \
SAVE_FREQ=-1 \
PROJECT_NAME=verl_grpo_gsm8k_examples_pd \
EXPERIMENT_NAME=qwen3_4b_grpo_vllm_epp_pd_fsdp_8gpu \
bash /opt/verl/examples/grpo_trainer/run_qwen3_4b_fsdp.sh \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.disaggregation.prefill_replicas=2 \
    actor_rollout_ref.rollout.disaggregation.decode_replicas=2 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config.kv_connector=NixlConnector \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config.kv_role=kv_both \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.no_disable_hybrid_kv_cache_manager=true \
    trainer.logger='["console","file"]' \
    trainer.default_local_dir=/tmp/checkpoints \
    trainer.validation_data_dir=/tmp/verl/generations/val \
    trainer.rollout_data_dir=/tmp/verl/generations/train \
    trainer.total_training_steps=80 \
    '+ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_ROOT=/tmp/verl/logs' \
    +actor_rollout_ref.model.external_lib=llm_d_rl_verl_integration.register_pd \
    +actor_rollout_ref.rollout.agent.agent_loop_manager_class=llm_d_rl_verl_integration.epp_router.agent_loop_manager.EPPAgentLoopManager \
    +actor_rollout_ref.rollout.custom.epp_config_file=/etc/llmd-configs/epp-config-pd.yaml \
    +actor_rollout_ref.rollout.custom.epp_endpoints_file=/tmp/epp-endpoints.yaml \
    +actor_rollout_ref.rollout.custom.sidecar_connector=nixlv2 \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.enable_prefix_caching=True \
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.enable_prompt_tokens_details=true' \
    'hydra.run.dir=/tmp/hydra-outputs'
```

### Envoy + EPP — HTTP proxy routing

```bash
MODEL_PATH=/tmp/verl/models/Qwen3-4B \
TRAIN_FILE=/tmp/verl/data/gsm8k/train.parquet \
TEST_FILE=/tmp/verl/data/gsm8k/test.parquet \
SAVE_FREQ=-1 \
bash /opt/verl/examples/grpo_trainer/run_qwen3_4b_fsdp.sh \
    trainer.logger='["console","file"]' \
    trainer.default_local_dir=/tmp/checkpoints \
    trainer.total_training_steps=50 \
    '+ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_ROOT=/tmp/verl/logs' \
    +actor_rollout_ref.rollout.agent.agent_loop_manager_class=llm_d_rl_verl_integration.llmd_stack.agent_loop_manager.EnvoyAgentLoopManager \
    +actor_rollout_ref.rollout.custom.epp_config_file=/etc/llmd-configs/epp-config.yaml \
    +actor_rollout_ref.rollout.custom.epp_endpoints_file=/tmp/epp-endpoints.yaml \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.enable_prefix_caching=True \
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.enable_prompt_tokens_details=true' \
    'hydra.run.dir=/tmp/hydra-outputs'
```

### Envoy + EPP — HTTP proxy routing, PD disaggregated

```bash
INFER_BACKEND=vllm-llmd-pd \
MODEL_PATH=/tmp/verl/models/Qwen3-4B \
TRAIN_FILE=/tmp/verl/data/gsm8k/train.parquet \
TEST_FILE=/tmp/verl/data/gsm8k/test.parquet \
SAVE_FREQ=-1 \
PROJECT_NAME=verl_grpo_gsm8k_examples_pd \
EXPERIMENT_NAME=qwen3_4b_grpo_vllm_envoy_pd_fsdp_8gpu \
bash /opt/verl/examples/grpo_trainer/run_qwen3_4b_fsdp.sh \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.disaggregation.prefill_replicas=2 \
    actor_rollout_ref.rollout.disaggregation.decode_replicas=2 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config.kv_connector=NixlConnector \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config.kv_role=kv_both \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.no_disable_hybrid_kv_cache_manager=true \
    trainer.logger='["console","file"]' \
    trainer.default_local_dir=/tmp/checkpoints \
    trainer.validation_data_dir=/tmp/verl/generations/val \
    trainer.rollout_data_dir=/tmp/verl/generations/train \
    trainer.total_training_steps=80 \
    '+ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_ROOT=/tmp/verl/logs' \
    +actor_rollout_ref.model.external_lib=llm_d_rl_verl_integration.register_pd \
    +actor_rollout_ref.rollout.agent.agent_loop_manager_class=llm_d_rl_verl_integration.llmd_stack.agent_loop_manager.EnvoyAgentLoopManager \
    +actor_rollout_ref.rollout.custom.epp_config_file=/etc/llmd-configs/epp-config-pd.yaml \
    +actor_rollout_ref.rollout.custom.epp_endpoints_file=/tmp/epp-endpoints.yaml \
    +actor_rollout_ref.rollout.custom.sidecar_connector=nixlv2 \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.enable_prefix_caching=True \
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.enable_prompt_tokens_details=true' \
    'hydra.run.dir=/tmp/hydra-outputs'
```

## Step 4 — Verify it's working

TODO

## EPP config

The configs in `examples/configs/` are starting points. Customize the scorer weights or swap plugins to tune routing for your workload. The path is passed via `EPP_CONFIG_FILE` — you can mount your own ConfigMap or point to any file accessible on the head node.

See the [main README](../README.md) for the full config reference and architecture overview.
