"""Base AgentLoopManager for llm-d integrations.

Subclasses override ``_create_llm_client()`` to return a custom client,
and optionally override ``_on_servers_ready()`` for extra setup work
(e.g. launching EPP).

No changes to verl core required — wire in via YAML:
    actor_rollout_ref.rollout.agent.agent_loop_manager_class: epp_router.agent_loop_manager.EPPAgentLoopManager
"""

from __future__ import annotations

import logging
from typing import Any

import ray
from omegaconf import DictConfig

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

        server_addresses: list[str] = ray.get(llm_client._load_balancer.get_all_servers.remote())
        logger.info("[LlmdAgentLoopManager] servers: %s", server_addresses)

        self._on_servers_ready(server_addresses)

        new_client = self._create_llm_client(server_addresses)
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

    def _create_llm_client(self, server_addresses: list[str]) -> LLMServerClient | None:
        """Create and return the replacement LLMServerClient.

        Return ``None`` to keep the original verl client unchanged.
        """
        return None
