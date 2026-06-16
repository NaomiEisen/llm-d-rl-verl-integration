"""Ray actor that manages the EPP + Envoy stack for the Envoy+EPP routing path.

Pin to the same node as the AgentLoopManager using NodeAffinitySchedulingStrategy
so the endpoints file is written on the same node that EPP reads from.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
from typing import Any, Optional

import ray

from llm_d_rl_verl_integration.epp_router.epp_launcher import EPPLauncher
from llm_d_rl_verl_integration.shared.endpoints import write_pd_endpoints, write_rollout_endpoints

logger = logging.getLogger(__name__)

_ENVOY_BINARY = "/usr/local/bin/envoy"
_ENVOY_LOG = "/tmp/envoy.log"
_DEFAULT_ENVOY_PORT = 8081
_BUNDLED_ENVOY_CONFIG = os.path.join(os.path.dirname(__file__), "envoy.yaml")


async def _wait_envoy_ready(port: int, timeout: float = 120.0) -> None:
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_connection("127.0.0.1", port), timeout=1.0
            )
            writer.close()
            await writer.wait_closed()
            return
        except (OSError, asyncio.TimeoutError):
            await asyncio.sleep(0.5)
    raise RuntimeError(f"Timed out after {timeout}s waiting for Envoy on ::{port}")


@ray.remote
class LlmdStackActor:
    """Owns the EPP and Envoy subprocess lifecycle.

    Must be scheduled on the same node as the AgentLoopManager so the
    endpoints file written here is readable by EPP on the same node.
    Use NodeAffinitySchedulingStrategy when creating this actor.
    """

    def __init__(self) -> None:
        self._epp_launcher: Optional[EPPLauncher] = None
        self._envoy_proc: Optional[subprocess.Popen] = None
        self._envoy_address: Optional[str] = None

    async def start(
        self,
        server_addresses: list[str],
        model_config: Any,
        rollout_config: Any,
        server_roles: Optional[list[str]] = None,
    ) -> str:
        """Write endpoints file, start EPP then Envoy. Returns ``host:port`` for Envoy."""
        custom = rollout_config.get("custom") or {}
        endpoints_file = custom.get("epp_endpoints_file")
        envoy_config = custom.get("envoy_config", _BUNDLED_ENVOY_CONFIG)
        envoy_port = int(custom.get("envoy_port", _DEFAULT_ENVOY_PORT))

        # Write endpoints on this node (co-located with EPP).
        if endpoints_file:
            pd_mode = rollout_config.get("name") == "vllm-llmd-pd"
            if pd_mode and server_roles and any(r is not None for r in server_roles):
                write_pd_endpoints(endpoints_file, server_addresses, server_roles, model_config)
            else:
                write_rollout_endpoints(endpoints_file, server_addresses, model_config)
            logger.info("[LlmdStackActor] wrote endpoints to %s", endpoints_file)

        # Start EPP and wait until healthy.
        self._epp_launcher = EPPLauncher(rollout_config)
        await self._epp_launcher.launch()
        logger.info("[LlmdStackActor] EPP ready")

        # Start Envoy and wait for its listener port.
        await self._start_envoy(envoy_config, envoy_port)
        logger.info("[LlmdStackActor] Envoy ready on :%d", envoy_port)

        host = ray.util.get_node_ip_address()
        self._envoy_address = f"{host}:{envoy_port}"
        return self._envoy_address

    async def _start_envoy(self, config_path: str, port: int) -> None:
        if not os.path.isfile(_ENVOY_BINARY):
            raise RuntimeError(f"Envoy binary not found at {_ENVOY_BINARY!r}")
        if not os.path.isfile(config_path):
            raise RuntimeError(f"Envoy config not found: {config_path!r}")

        cmd = [
            _ENVOY_BINARY,
            "--service-node", "envoy-proxy",
            "--log-level", os.environ.get("VERL_ENVOY_LOG_LEVEL", "info"),
            "--concurrency", "8",
            "--drain-strategy", "immediate",
            "--drain-time-s", "60",
            "--disable-hot-restart",
            "-c", config_path,
        ]
        logger.info("[LlmdStackActor] starting Envoy: %s", " ".join(cmd))
        self._envoy_proc = subprocess.Popen(
            cmd,
            stdout=open(_ENVOY_LOG, "w"),
            stderr=subprocess.STDOUT,
        )
        await _wait_envoy_ready(port)

    def stop(self) -> None:
        if self._epp_launcher is not None:
            self._epp_launcher.stop()
            self._epp_launcher = None
        if self._envoy_proc is not None:
            self._envoy_proc.terminate()
            self._envoy_proc = None

    def __del__(self) -> None:
        self.stop()
