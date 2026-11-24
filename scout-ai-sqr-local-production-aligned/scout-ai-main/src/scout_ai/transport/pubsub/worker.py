from __future__ import annotations

import argparse
import logging

from scout_ai.config import get_settings
from scout_ai.transport.container import AppContainer
from scout_ai.transport.flows import (
    _build_draft_processor,
    _build_kpi_compute_processor,
    _build_regenerate_processor,
)
from scout_ai.transport.pubsub.consumer import PubSubConsumer, Subscription

logger = logging.getLogger(__name__)


# flow name -> (resolve its subscription id from settings, build its processor)
_FLOWS = {
    "draft": (lambda s: s.pubsub_subscription, _build_draft_processor),
    "regenerate": (
        lambda s: s.pubsub_regenerate_subscription,
        _build_regenerate_processor,
    ),
    "kpi_compute": (
        lambda s: s.pubsub_kpi_recompute_subscription,
        _build_kpi_compute_processor,
    ),
}


# --------------------------------------------------------------------------- #
# Entrypoints
# --------------------------------------------------------------------------- #
def run(flows: list[str] | None = None) -> None:
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    selected = flows or list(_FLOWS)
    container = AppContainer.build(settings)
    subscriptions = [
        Subscription(
            subscription_id=_FLOWS[name][0](settings),
            process=_FLOWS[name][1](container),
        )
        for name in selected
    ]
    logger.info("Starting worker with flows: %s", ", ".join(selected))
    PubSubConsumer(
        project_id=settings.pubsub_project_id,
        max_messages=settings.pubsub_max_messages,
        subscriptions=subscriptions,
        on_close=container.dispose,
    ).run()


def main() -> None:  # console-script / -m entrypoint
    parser = argparse.ArgumentParser(description="scout-ai Pub/Sub worker")
    parser.add_argument(
        "--flow",
        action="append",
        choices=sorted(_FLOWS),
        help="Flow(s) to run (repeatable). Defaults to all flows.",
    )
    args = parser.parse_args()
    run(args.flow)


if __name__ == "__main__":
    main()
