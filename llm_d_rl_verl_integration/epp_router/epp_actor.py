"""Ray actor that writes endpoints YAML and launches the EPP subprocess."""

from __future__ import annotations

import logging
from typing import Any, Optional

import ray

from llm_d_rl_verl_integration.shared.endpoints import write_pd_endpoints, write_rollout_endpoints
from llm_d_rl_verl_integration.epp_router.epp_launcher import EPPLauncher

logger = logging.getLogger(__name__)


@ray.remote
class EPPActor:
    """Pinned to the head node. Owns the EPP subprocess lifecycle."""

    async def start(
        self,
        rollout_config: Any,
        server_addresses: list[str],
        model_config: Any,
        epp_endpoints_file: Optional[str],
        server_roles: Optional[list[str]] = None,
    ) -> str:
        """Write endpoints YAML, launch EPP, return routable ``host:port`` for workers."""
        if epp_endpoints_file:
            if server_roles and any(r is not None for r in server_roles):
                write_pd_endpoints(epp_endpoints_file, server_addresses, server_roles, model_config)
            else:
                write_rollout_endpoints(epp_endpoints_file, server_addresses, model_config)

        self._launcher = EPPLauncher(rollout_config)
        grpc_port = await self._launcher.launch()

        host = ray.util.get_node_ip_address()
        grpc_addr = f"{host}:{grpc_port}"
        logger.info("EPPActor ready at %s", grpc_addr)
        return grpc_addr

    def stop(self) -> None:
        if getattr(self, "_launcher", None) is not None:
            self._launcher.stop()
            self._launcher = None
