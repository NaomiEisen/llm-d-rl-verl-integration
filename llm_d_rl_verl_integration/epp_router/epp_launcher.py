"""EPP subprocess launcher."""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import subprocess
import threading
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)

_EPP_BINARY = "/usr/local/bin/epp"
_EPP_DEBUG_LOG = "/tmp/epp_debug.log"
DEFAULT_EPP_GRPC_PORT = 9002


def _drain_output(proc: subprocess.Popen, tail: Optional[deque] = None) -> None:
    """Drain EPP stdout/stderr in a background thread."""
    stream = proc.stdout
    if stream is None:
        return

    def _run():
        try:
            with open(_EPP_DEBUG_LOG, "a", encoding="utf-8") as f:
                with stream:
                    for raw in iter(stream.readline, b""):
                        line = raw.decode(errors="replace").rstrip()
                        if line:
                            f.write(line + "\n")
                            f.flush()
                            if tail is not None:
                                tail.append(line)
        except Exception:
            logger.exception("error draining EPP output")

    threading.Thread(target=_run, name="epp-output", daemon=True).start()


class EPPLauncher:
    """Manages the EPP subprocess lifecycle."""

    def __init__(self, rollout_config):
        self._config = rollout_config
        self._process: Optional[subprocess.Popen] = None
        self._tail: deque = deque(maxlen=120)

    async def launch(self) -> int:
        """Spawn EPP and wait until its gRPC port is ready. Returns the gRPC port."""
        custom = self._config.get("custom") or {}
        epp_config_file = custom.get("epp_config_file")
        if not epp_config_file:
            raise RuntimeError(
                "rollout.custom.epp_config_file is required when using llm-d integration"
            )
        if not os.path.isfile(_EPP_BINARY):
            raise RuntimeError(f"EPP binary not found at {_EPP_BINARY!r}")
        if not os.path.isfile(epp_config_file):
            raise RuntimeError(f"EPP config file not found: {epp_config_file!r}")

        grpc_port = int(custom.get("epp_grpc_port", DEFAULT_EPP_GRPC_PORT))
        endpoint_selector = custom.get("epp_endpoint_selector", "app=verl-vllm")
        endpoint_target_ports = custom.get("epp_endpoint_target_ports", "8000")
        pod_name = os.environ.get("POD_NAME", os.environ.get("HOSTNAME", "verl-epp-abc12-xyz34"))

        cmd = [
            _EPP_BINARY,
            "--config-file", epp_config_file,
            "--grpc-port", str(grpc_port),
            "--grpc-health-port", str(grpc_port + 1),
            "--metrics-port", "9090",
            "--endpoint-selector", endpoint_selector,
            "--endpoint-target-ports", endpoint_target_ports,
            "--secure-serving=false",
            "--tracing=false",
            "-v=5",
        ]

        env = {**os.environ, "POD_NAME": pod_name}
        logger.info("Launching EPP: %s", " ".join(cmd))
        print(f"[verl EPP] spawning: {' '.join(cmd)} (POD_NAME={pod_name})", flush=True)
        self._process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
        _drain_output(self._process, self._tail)
        print(f"[verl EPP] waiting for gRPC on 127.0.0.1:{grpc_port}", flush=True)

        await self._wait_ready(grpc_port, epp_config_file)
        return grpc_port

    async def _wait_ready(self, port: int, config_file: str) -> None:
        timeout = float(os.environ.get("VERL_EPP_START_TIMEOUT", "120"))
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout

        while loop.time() < deadline:
            if self._process is not None and self._process.poll() is not None:
                print(f"[verl EPP] subprocess exited early with code {self._process.returncode}", flush=True)
                if self._tail:
                    print("[verl EPP] --- last epp stdout/stderr ---", flush=True)
                    for line in self._tail:
                        print(line, flush=True)
                    print("[verl EPP] --- end epp tail ---", flush=True)
                raise RuntimeError(
                    f"EPP subprocess exited with code {self._process.returncode} before gRPC ready on port {port}. "
                    f"Full output at {_EPP_DEBUG_LOG!r}; config {config_file!r}."
                )
            try:
                def _probe():
                    s = socket.create_connection(("127.0.0.1", port), timeout=3.0)
                    s.close()
                await asyncio.wait_for(loop.run_in_executor(None, _probe), timeout=5.0)
                return
            except (OSError, asyncio.TimeoutError):
                await asyncio.sleep(0.5)

        raise RuntimeError(f"Timed out waiting {timeout}s for EPP gRPC on port {port}")

    def stop(self) -> None:
        if self._process is not None:
            self._process.terminate()
            self._process = None
