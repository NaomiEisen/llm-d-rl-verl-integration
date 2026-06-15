"""AgentLoopManager for a pre-deployed llm-d stack (EPP + Envoy running externally).

Writes the EPP endpoints YAML so EPP can discover verl's vLLM replicas,
then routes all generation requests through the pre-deployed Envoy endpoint.

YAML config (no verl code changes needed):
    actor_rollout_ref:
      rollout:
        agent:
          agent_loop_manager_class: llm_d_rl_verl_integration.envoy.agent_loop_manager.EnvoyAgentLoopManager
        custom:
          envoy_address: "localhost:8081"
          epp_endpoints_file: /tmp/epp-endpoints.yaml  # must match EPP's config
"""

from __future__ import annotations

import logging

from omegaconf import OmegaConf

from llm_d_rl_verl_integration.shared.base_agent_loop_manager import LlmdAgentLoopManager
from llm_d_rl_verl_integration.envoy.llm_client import EnvoyLLMClient
from llm_d_rl_verl_integration.shared.endpoints import write_pd_endpoints, write_rollout_endpoints
from verl.workers.rollout.llm_server import LLMServerClient

logger = logging.getLogger(__name__)


class EnvoyAgentLoopManager(LlmdAgentLoopManager):
    """Writes endpoint discovery YAML for EPP, then routes via pre-deployed Envoy."""

    def _on_servers_ready(self, server_addresses: list[str]) -> None:
        rollout_cfg = self.rollout_config
        custom = OmegaConf.to_container(rollout_cfg.get("custom") or {}, resolve=True)

        envoy_address = custom.get("envoy_address")
        if not envoy_address:
            raise RuntimeError(
                "rollout.custom.envoy_address is required for EnvoyAgentLoopManager"
            )
        self._envoy_address = envoy_address
        self._model_name = self.model_config.path

        endpoints_file = custom.get("epp_endpoints_file")
        if endpoints_file:
            disagg = getattr(rollout_cfg, "disaggregation", None)
            pd_mode = bool(disagg and getattr(disagg, "enabled", False))
            if pd_mode:
                server_roles = _infer_roles(server_addresses, rollout_cfg)
                write_pd_endpoints(endpoints_file, server_addresses, server_roles, self.model_config)
            else:
                write_rollout_endpoints(endpoints_file, server_addresses, self.model_config)
            logger.info("[EnvoyAgentLoopManager] wrote endpoints to %s", endpoints_file)

    def _create_llm_client(self, server_addresses: list[str]) -> LLMServerClient:
        return EnvoyLLMClient(
            config=self.config,
            load_balancer_handle=self.llm_client._load_balancer,
            envoy_address=self._envoy_address,
            model_name=self._model_name,
        )


def _infer_roles(server_addresses: list[str], rollout_cfg) -> list[str]:
    disagg = rollout_cfg.disaggregation
    n_prefill = int(getattr(disagg, "prefill_replicas", 1))
    n_decode = int(getattr(disagg, "decode_replicas", 1))
    roles = ["prefill"] * n_prefill + ["decode"] * n_decode
    if len(roles) < len(server_addresses):
        roles += ["decode"] * (len(server_addresses) - len(roles))
    return roles[: len(server_addresses)]
