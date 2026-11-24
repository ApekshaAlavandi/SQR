from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from google.cloud import pubsub_v1

from scout_ai.transport.flows import Processor, ProcessOutcome

logger = logging.getLogger(__name__)

# Visual boundary logged after every message. With three subscriptions interleaving
# in one log stream, this brackets each request; the trailing newline leaves a blank
# line between messages and the subscription id / message id stay greppable.
_END_RULE = "─" * 72


def _log_message_end(subscription_id: str, message_id: str, outcome: str) -> None:
    logger.info(
        "end [%s] msg=%s → %s\n%s\n", subscription_id, message_id, outcome, _END_RULE
    )


__all__ = ["ProcessOutcome", "Processor", "PubSubConsumer", "Subscription"]


@dataclass(frozen=True)
class Subscription:
    """One subscription and the processor that handles its messages."""

    subscription_id: str
    process: Processor


class PubSubConsumer:
    """Streams one or more subscriptions and dispatches each to its processor."""

    def __init__(
        self,
        *,
        project_id: str,
        max_messages: int,
        subscriptions: list[Subscription],
        on_close: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        if not project_id:
            raise ValueError("project_id is required")
        if not subscriptions:
            raise ValueError("at least one subscription is required")
        if any(not s.subscription_id for s in subscriptions):
            raise ValueError("every subscription needs a subscription_id")
        self._project_id = project_id
        self._max_messages = max_messages
        self._subscriptions = subscriptions
        self._on_close = on_close
        self._subscriber = pubsub_v1.SubscriberClient()
        # One event loop in its own thread services the async work for every
        # subscription; Pub/Sub callbacks (on worker threads) marshal onto it.
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever, name="pubsub-loop", daemon=True
        )

    def run(self) -> None:
        """Block, consuming from all subscriptions until interrupted."""
        self._loop_thread.start()
        flow_control = pubsub_v1.types.FlowControl(max_messages=self._max_messages)

        futures = []
        for sub in self._subscriptions:
            path = self._subscriber.subscription_path(self._project_id, sub.subscription_id)
            futures.append(
                self._subscriber.subscribe(
                    path,
                    callback=self._make_callback(sub.subscription_id, sub.process),
                    flow_control=flow_control,
                )
            )
            logger.info("Listening on %s", path)

        try:
            # Each future pulls in the background (the client auto-reconnects
            # transient stream errors); blocking on them keeps the process alive.
            # A permanent stream error or KeyboardInterrupt drops to finally.
            for future in futures:
                future.result()
        except KeyboardInterrupt:
            logger.info("Shutting down consumer")
        finally:
            for future in futures:
                future.cancel()
            for future in futures:
                try:
                    future.result(timeout=5)
                except Exception:  # noqa: BLE001 - best-effort drain
                    pass
            if self._on_close is not None:
                close = asyncio.run_coroutine_threadsafe(self._on_close(), self._loop)
                try:
                    close.result(timeout=10)
                except Exception:  # noqa: BLE001 - best-effort shutdown
                    logger.exception("on_close hook failed")
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._subscriber.close()

    def _make_callback(self, subscription_id: str, process: Processor):
        def _on_message(message: pubsub_v1.subscriber.message.Message) -> None:
            # Block this callback thread on the async processor — that's what
            # gives us flow control (at most `max_messages` in flight per sub).
            future = asyncio.run_coroutine_threadsafe(process(message.data), self._loop)
            try:
                outcome = future.result()
            except Exception:  # noqa: BLE001 - retryable; let Pub/Sub redeliver
                logger.exception("Processing failed for message %s", message.message_id)
                message.nack()
                _log_message_end(subscription_id, message.message_id, "NACK")
                return
            logger.info("message %s -> %s", message.message_id, outcome.name)
            message.ack()
            _log_message_end(subscription_id, message.message_id, outcome.name)

        return _on_message