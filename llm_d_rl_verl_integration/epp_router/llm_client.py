"""LLMServerClient that routes via EPP gRPC, then delegates inference to the
chosen vLLM actor handle exactly as original verl does.

Non-PD: EPP picks endpoint → call actor.generate.remote() → vLLM handles it.
PD:     EPP picks decode endpoint + sidecar headers → call actor.generate.remote(sidecar_headers=...)
        → PDDecodeVLLMHttpServer.generate() → HTTP to local sidecar.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import socket

import ray

from verl.workers.rollout.llm_server import LLMServerClient
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__name__)

_HEART_LOG = f"/tmp/heart_debug_{socket.gethostname()}.log"


def _hlog(msg: str) -> None:
    with open(_HEART_LOG, "a") as _f:
        _f.write(msg + "\n")


class EPPLLMClient(LLMServerClient):
    """Routes each request through EPP gRPC to pick a server, then calls
    that server's Ray actor directly — same as original verl flow.

    Args:
        config: verl DictConfig.
        load_balancer_handle: original GlobalRequestLoadBalancer (kept for
            compatibility but not used for routing decisions).
        grpc_addr: EPP gRPC address (``host:port``).
        address_to_handle: ``{server_address: ray_actor_handle}`` map built
            at startup. server_address must match what EPP returns as the
            ``x-gateway-destination-endpoint`` header.
        model_name: model identifier sent in the EPP request body.
        pd_mode: if True, forward sidecar_headers returned by EPP to
            actor.generate.remote() so PDDecodeVLLMHttpServer can reach the sidecar.
    """

    def __init__(
        self,
        config,
        load_balancer_handle=None,
        *,
        grpc_addr: str,
        address_to_handle: dict[str, ray.actor.ActorHandle],
        model_name: str,
        pd_mode: bool = False,
        **kwargs,
    ):
        super().__init__(config=config, load_balancer_handle=load_balancer_handle, **kwargs)
        self._grpc_addr = grpc_addr
        self._address_to_handle = address_to_handle
        self._model_name = model_name
        self._pd_mode = pd_mode
        self._epp_client = None  # created on workers after unpickling via __setstate__

    def __setstate__(self, state):
        self.__dict__.update(state)
        from llm_d_rl_verl_integration.epp_router.grpc_client import EPPGrpcClient
        self._epp_client = EPPGrpcClient(self._grpc_addr)

    async def generate(
        self,
        request_id,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data=None,
        video_data=None,
        **kwargs,
    ) -> TokenOutput:
        endpoint, sidecar_headers = await self._epp_client.pick(self._model_name, prompt_ids)

        _hlog(
            f"❤️ [EPPLLMClient] request_id={request_id} "
            f"pd_mode={self._pd_mode} "
            f"endpoint={endpoint!r} "
            f"sidecar_headers={sidecar_headers!r} "
            f"known_endpoints={list(self._address_to_handle.keys())}"
        )

        if endpoint is None:
            raise RuntimeError(f"EPP returned no endpoint for request {request_id}")

        actor = self._address_to_handle.get(endpoint)
        if actor is None:
            raise RuntimeError(
                f"EPP returned endpoint {endpoint!r} which is not in the known server map. "
                f"Known: {list(self._address_to_handle.keys())}"
            )

        extra_kwargs: dict[str, Any] = {}
        if self._pd_mode and sidecar_headers:
            extra_kwargs["sidecar_headers"] = sidecar_headers

        _hlog(
            f"❤️ [EPPLLMClient] calling actor.generate pd_mode={self._pd_mode} "
            f"passing_sidecar_headers={'sidecar_headers' in extra_kwargs} "
            f"x-prefiller-host-port={sidecar_headers.get('x-prefiller-host-port')!r}"
        )

        return await actor.generate.remote(
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            request_id=request_id,
            image_data=image_data,
            video_data=video_data,
            **extra_kwargs,
        )
