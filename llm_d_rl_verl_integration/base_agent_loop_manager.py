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

import ray
from omegaconf import DictConfig, OmegaConf
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from verl.experimental.agent_loop.agent_loop import AgentLoopManager
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
        """Write endpoint → replica rank + node IPs + GPU UUIDs map, once at startup.

        For each replica, queries every vLLM worker actor via __ray_call__ to collect:
          - Ray logical GPU IDs (from get_accelerator_ids())
          - Hardware UUIDs (via pynvml) — stable identifiers for cross-referencing
            with external GPU monitoring tools (nvidia-smi, DCGM, etc.)

        pynvml is optional: if not installed, only replica_rank and node_ips are written.
        """
        from verl.utils.device import get_resource_name
        resource_name = get_resource_name()

        def _query_worker_gpu_info(_, rn=resource_name):
            import os as _os
            import ray as _ray
            logical_ids = _ray.get_runtime_context().get_accelerator_ids().get(rn, [])
            node_ip = _ray.util.get_node_ip_address()
            result = {"node_ip": node_ip, "logical_ids": logical_ids}
            try:
                import pynvml
                # Map Ray logical indices to physical GPU indices via CUDA_VISIBLE_DEVICES,
                # since nvmlDeviceGetHandleByIndex expects physical driver indices.
                cuda_visible = _os.environ.get("CUDA_VISIBLE_DEVICES", "")
                physical_ids = [int(x) for x in cuda_visible.split(",") if x.strip().isdigit()]
                pynvml.nvmlInit()
                uuids = []
                for logical_id in logical_ids:
                    physical_idx = physical_ids[int(logical_id)] if physical_ids else int(logical_id)
                    handle = pynvml.nvmlDeviceGetHandleByIndex(physical_idx)
                    uuid = pynvml.nvmlDeviceGetUUID(handle)
                    uuids.append(uuid.decode("utf-8") if isinstance(uuid, bytes) else uuid)
                pynvml.nvmlShutdown()
                result["gpu_uuids"] = uuids
            except Exception:
                pass
            return result

        gpu_map = {}
        for i, addr in enumerate(server_addresses):
            actor_name = f"vllm_server_{i}_0"
            try:
                handle = ray.get_actor(actor_name)
                workers = ray.get(handle.__ray_call__.remote(lambda self: self.workers))
                worker_infos = ray.get([
                    worker.__ray_call__.remote(_query_worker_gpu_info)
                    for worker in workers
                ])
                node_ips = list({info["node_ip"] for info in worker_infos if "node_ip" in info})
                gpu_uuids = [u for info in worker_infos for u in info.get("gpu_uuids", [])]
                entry = {"replica_rank": i, "node_ips": node_ips}
                if gpu_uuids:
                    entry["gpu_uuids"] = gpu_uuids
                gpu_map[addr] = entry
            except Exception:
                gpu_map[addr] = {"replica_rank": i}
        out_path = os.path.join(self._metrics_output_dir, "endpoint_gpu_map.json")
        with open(out_path, "w") as f:
            json.dump(gpu_map, f, indent=2)
        logger.info("[LlmdAgentLoopManager] endpoint_gpu_map → %s: %s", out_path, gpu_map)

    # ------------------------------------------------------------------
    # generate_sequences — timed wrapper for gen timeline + per-sample log
    # ------------------------------------------------------------------

    async def generate_sequences(self, prompts):
        step = int(prompts.meta_info.get("global_steps", -1))
        t_start = time.time()
        output = await super().generate_sequences(prompts)
        t_end = time.time()

        # Accumulate in memory.
        self._timeline_buffer.append(json.dumps({
            "phase": "gen",
            "step": step,
            "start_time": round(t_start, 3),
            "end_time": round(t_end, 3),
            "duration_s": round(t_end - t_start, 3),
        }))

        per_sample = output.meta_info.get("per_sample_data")
        if per_sample is not None:
            record: dict = {"step": step}
            for key, val in per_sample.items():
                record[key] = val.tolist() if hasattr(val, "tolist") else list(val)
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
