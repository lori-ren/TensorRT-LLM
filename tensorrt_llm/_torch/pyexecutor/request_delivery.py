# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

from .llm_request import LlmResponse

if TYPE_CHECKING:
    from ray.actor import ActorHandle

__all__ = ["ResponseChannel"]


class ResponseChannel:
    """Local responses waiting for pickup and the condition their waiters sleep on."""

    def __init__(self) -> None:
        self.response_lock = threading.Lock()
        self.response_cv = threading.Condition(self.response_lock)
        self.responses: dict[int, list[LlmResponse]] = {}
        self.result_wait_queues: dict[int, ActorHandle | None] = {}

    def register(self, request_ids: Iterable[int], result_wait_queue: ActorHandle) -> None:
        with self.response_cv:
            for request_id in request_ids:
                self.result_wait_queues[request_id] = result_wait_queue

    def release(self, request_id: int) -> None:
        self.result_wait_queues.pop(request_id, None)

    def publish(self, responses: Iterable[tuple[int, LlmResponse]]) -> None:
        with self.response_cv:
            for req_id, resp in responses:
                if req_id in self.responses.keys():
                    self.responses[req_id].append(resp)
                else:
                    self.responses.update({req_id: [resp]})
                if (
                    type(resp) is LlmResponse
                    and req_id in self.result_wait_queues
                    and self.result_wait_queues[req_id] is not None
                ):
                    self.result_wait_queues[req_id].put_response.remote(resp.client_id, resp)
            self.response_cv.notify_all()

    def await_any(
        self, timeout: float | None, *, is_shutdown: Callable[[], bool]
    ) -> list[LlmResponse]:
        def any_responses_ready() -> bool:
            return len(self.responses) > 0 or is_shutdown()

        responses = []
        with self.response_cv:
            self.response_cv.wait_for(any_responses_ready, timeout=timeout)
            for response in self.responses.values():
                responses += response
            self.responses = {}

        return responses

    def await_single(
        self,
        id: int,
        timeout: float | None,
        *,
        is_shutdown: Callable[[], bool],
        read_error: Callable[[], BaseException | None],
        mark_error_delivered: Callable[[], None],
    ) -> list[LlmResponse]:
        with self.response_cv:

            def key_has_response() -> bool:
                # Wake on shutdown too so that an event-loop crash
                # cannot trap callers here forever (nvbug 6038228).
                return id in self.responses or is_shutdown()

            self.response_cv.wait_for(key_has_response, timeout=timeout)
            if id in self.responses:
                return self.responses.pop(id)
            if is_shutdown():
                # The event-loop thread terminated before producing a
                # response for this request. Re-raise the original
                # exception (if any) so callers see a meaningful error
                # instead of hanging here or hitting a KeyError below.
                error = read_error()
                if error is not None:
                    # The caller is about to see the original exception, so
                    # the crash is reportable without killing the world.
                    mark_error_delivered()
                    raise RuntimeError(f"Event loop terminated with error: {error}") from error
                raise RuntimeError(
                    f"Event loop shut down before a response was received for request {id}."
                )
            # Timed out without shutdown: return [] like the other timeouts.
            return []
