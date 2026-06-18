"""Base AgentLoopManager for llm-d integrations.

Subclasses override ``_create_llm_client()`` to return a custom client,
and optionally override ``_on_servers_ready()`` for extra setup work
(e.g. launching EPP).

No changes to verl core required — wire in via YAML:
    actor_rollout_ref.rollout.agent.agent_loop_manager_class: epp_router.agent_loop_manager.EPPAgentLoopManager
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import numpy as np
import ray
from omegaconf import DictConfig, OmegaConf
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from verl.experimental.agent_loop.agent_loop import AgentLoopManager
from verl.utils.ray_utils import auto_await
from verl.workers.rollout.llm_server import LLMServerClient

logger = logging.getLogger(__name__)


class LlmdAgentLoopManager(AgentLoopManager):
    """Base class for llm-d AgentLoopManager variants.

    Lifecycle (runs in ``__init__``, before workers are spawned):
    1. ``super().__init__()`` — stores config, original llm_client.
    2. Retrieve server addresses from the GlobalRequestLoadBalancer actor.
    3. Call ``_on_servers_ready(server_addresses)`` — subclass hook for
       writing endpoints YAML, launching EPP, etc.
    4. Build a replacement client via ``_create_llm_client(server_addresses)``.
    5. Replace ``self.llm_client`` so workers receive the new client.
    """

    def __init__(
        self,
        config: DictConfig,
        llm_client: LLMServerClient,
        teacher_client=None,
        reward_loop_worker_handles=None,
    ):
        super().__init__(
            config=config,
            llm_client=llm_client,
            teacher_client=teacher_client,
            reward_loop_worker_handles=reward_loop_worker_handles,
        )

        # Set up metrics output directory and flush frequency.
        # Configurable via rollout.custom.metrics_dir and rollout.custom.metrics_flush_freq.
        rollout_cfg = self.rollout_config
        custom = OmegaConf.to_container(rollout_cfg.get("custom") or {}, resolve=True)
        self._metrics_output_dir: str = custom.get("metrics_dir", "/tmp/verl")
        self._flush_freq: int = int(custom.get("metrics_flush_freq", 10))
        os.makedirs(self._metrics_output_dir, exist_ok=True)
        self._gen_timeline_path = os.path.join(self._metrics_output_dir, "gen_timeline.jsonl")
        self._per_sample_path = os.path.join(self._metrics_output_dir, "per_sample.jsonl")
        # Truncate at startup so each run starts fresh.
        open(self._gen_timeline_path, "w").close()
        open(self._per_sample_path, "w").close()
        # In-memory buffers — flushed to disk every _flush_freq steps.
        self._timeline_buffer: list[str] = []
        self._per_sample_buffer: list[str] = []
        self._steps_since_flush: int = 0
        # Captured in _performance_metrics before verl aggregates away per-sample timings.
        self._last_gen_times: np.ndarray | None = None
        logger.info(
            "[LlmdAgentLoopManager] metrics output dir: %s  flush_freq: %d",
            self._metrics_output_dir, self._flush_freq,
        )

        server_addresses: list[str] = ray.get(llm_client._load_balancer.get_all_servers.remote())
        logger.info("[LlmdAgentLoopManager] servers: %s", server_addresses)

        self._on_servers_ready(server_addresses)
        self._write_endpoint_gpu_map(server_addresses)

        new_client = self._create_llm_client()
        if new_client is not None:
            self.llm_client = new_client

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    def _on_servers_ready(self, server_addresses: list[str]) -> None:
        """Called after server addresses are retrieved, before client creation.

        Override to write YAML files, launch EPP, etc.
        Default implementation does nothing.
        """

    def _create_llm_client(self) -> LLMServerClient | None:
        """Create and return the replacement LLMServerClient.

        Return ``None`` to keep the original verl client unchanged.
        """
        return None

    # ------------------------------------------------------------------
    # Endpoint GPU map — written once at startup for both integrations
    # ------------------------------------------------------------------

    def _write_endpoint_gpu_map(self, server_addresses: list[str]) -> None:
        """Write endpoint → replica rank + Ray-assigned GPU IDs map, once at startup."""
        from verl.utils.device import get_resource_name
        resource_name = get_resource_name()

        def _query_worker_gpu_info(_, rn=resource_name):
            import ray as _ray
            return _ray.get_runtime_context().get_accelerator_ids().get(rn, [])

        gpu_map = {}
        for i, addr in enumerate(server_addresses):
            actor_name = f"vllm_server_{i}_0"
            try:
                handle = ray.get_actor(actor_name)
                workers = ray.get(handle.__ray_call__.remote(lambda self: self.workers))
                worker_gpu_ids = ray.get([
                    worker.__ray_call__.remote(_query_worker_gpu_info)
                    for worker in workers
                ])
                gpu_map[addr] = {
                    "replica_rank": i,
                    "gpu_ids": [gpu_id for gpu_ids in worker_gpu_ids for gpu_id in gpu_ids],
                }
            except Exception:
                gpu_map[addr] = {"replica_rank": i}
        out_path = os.path.join(self._metrics_output_dir, "endpoint_gpu_map.json")
        with open(out_path, "w") as f:
            json.dump(gpu_map, f, indent=2)
        logger.info("[LlmdAgentLoopManager] endpoint_gpu_map → %s: %s", out_path, gpu_map)

    # ------------------------------------------------------------------
    # _performance_metrics — capture per-sample gen_time_s before aggregation
    # ------------------------------------------------------------------

    def _performance_metrics(self, metrics, output):
        """Capture per-sample generation time before verl reduces metrics to aggregates."""
        self._last_gen_times = np.array(
            [metric["generate_sequences"] for chunk in metrics for metric in chunk],
            dtype=np.float32,
        )
        return super()._performance_metrics(metrics, output)

    # ------------------------------------------------------------------
    # generate_sequences — timed wrapper for gen timeline + per-sample log
    # ------------------------------------------------------------------

    @auto_await
    async def generate_sequences(self, prompts):
        step = int(prompts.meta_info.get("global_steps", -1))
        validate = bool(prompts.meta_info.get("validate", False))
        t_start = time.time()
        output = await super().generate_sequences(prompts)
        t_end = time.time()

        # Accumulate timeline entry.
        self._timeline_buffer.append(json.dumps({
            "phase": "gen",
            "step": step,
            "validate": validate,
            "start_time": round(t_start, 3),
            "end_time": round(t_end, 3),
            "duration_s": round(t_end - t_start, 3),
        }))

        # prompt_len / response_len from verl's final padded attention_mask.
        # Shape: [batch, prompt_length + response_length]; 1=real token, 0=padding.
        attention_mask = output.batch["attention_mask"]
        prompt_width = output.batch["prompts"].shape[1]
        prompt_len = attention_mask[:, :prompt_width].sum(dim=1).cpu().numpy().astype(np.int32)
        response_len = attention_mask[:, prompt_width:].sum(dim=1).cpu().numpy().astype(np.int32)

        # gen_time_s is captured in _performance_metrics during super().generate_sequences().
        gen_time = self._last_gen_times

        # endpoint is stamped by the custom LLM client into TokenOutput.extra_fields.
        endpoint_raw = output.non_tensor_batch.pop("_llmd_endpoint", None)

        if gen_time is not None:
            per_sample: dict = {
                "gen_time_s": gen_time,
                "prompt_len": prompt_len,
                "response_len": response_len,
            }
            if endpoint_raw is not None:
                per_sample["endpoint"] = [
                    str(x) if x is not None else "unknown" for x in endpoint_raw
                ]
            output.meta_info["per_sample_data"] = per_sample

            record: dict = {"step": step, "validate": validate}
            for key, val in per_sample.items():
                if hasattr(val, "tolist"):
                    record[key] = val.tolist()
                else:
                    record[key] = list(val)
            self._per_sample_buffer.append(json.dumps(record))

        self._steps_since_flush += 1
        if self._steps_since_flush >= self._flush_freq:
            self._flush_buffers()

        return output

    def _flush_buffers(self) -> None:
        if self._timeline_buffer:
            with open(self._gen_timeline_path, "a") as f:
                f.write("\n".join(self._timeline_buffer) + "\n")
            self._timeline_buffer.clear()
        if self._per_sample_buffer:
            with open(self._per_sample_path, "a") as f:
                f.write("\n".join(self._per_sample_buffer) + "\n")
            self._per_sample_buffer.clear()
        logger.debug("[LlmdAgentLoopManager] flushed metrics buffers (step %d)", self._steps_since_flush)
        self._steps_since_flush = 0

    def __del__(self):
        try:
            self._flush_buffers()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def head_node_strategy() -> NodeAffinitySchedulingStrategy:
        """Return a scheduling strategy that pins a Ray actor to the head node."""
        gcs_address = ray.get_runtime_context().gcs_address
        head_ip = gcs_address.split(":")[0]
        for node in ray.nodes():
            if node["Alive"] and node["NodeManagerAddress"] == head_ip:
                return NodeAffinitySchedulingStrategy(node_id=node["NodeID"], soft=False)
        raise RuntimeError(
            f"Could not find a live Ray node with GCS IP {head_ip}. "
            f"Nodes: {[n['NodeManagerAddress'] for n in ray.nodes()]}"
        )

    @staticmethod
    def infer_roles(server_addresses: list[str], rollout_cfg: Any) -> list[str]:
        """Infer prefill/decode roles from disaggregation config."""
        disagg = rollout_cfg.disaggregation
        n_prefill = int(getattr(disagg, "prefill_replicas", 1))
        n_decode = int(getattr(disagg, "decode_replicas", 1))
        roles = ["prefill"] * n_prefill + ["decode"] * n_decode
        if len(roles) < len(server_addresses):
            roles += ["decode"] * (len(server_addresses) - len(roles))
        return roles[: len(server_addresses)]
