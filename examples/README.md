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

Exec into the head pod and run one of the training scripts:

```bash
kubectl exec -it <head-pod-name> -- bash
bash /opt/verl/examples/grpo_trainer/run_qwen3_4b_fsdp.sh \
    actor_rollout_ref.rollout.agent.agent_loop_manager_class=llm_d_rl_verl_integration.epp_router.agent_loop_manager.EPPAgentLoopManager \
    +actor_rollout_ref.rollout.custom.epp_config_file=/etc/llmd-configs/epp-config.yaml \
    +actor_rollout_ref.rollout.custom.epp_endpoints_file=/tmp/epp-endpoints.yaml
```

## Step 4 — Verify it's working

TODO

## EPP config

The configs in `examples/configs/` are starting points. Customize the scorer weights or swap plugins to tune routing for your workload. The path is passed via `EPP_CONFIG_FILE` — you can mount your own ConfigMap or point to any file accessible on the head node.

See the [main README](../README.md) for the full config reference and architecture overview.
