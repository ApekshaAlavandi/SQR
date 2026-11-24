from __future__ import annotations


class DomainError(Exception):
    """Base class for all domain errors."""


class InvalidGcsUriError(DomainError):
    """Raised when a string cannot be parsed as a `gs://bucket/object` URI."""


class ClipNotFoundError(DomainError):
    """Raised when a clip cannot be located by its identifier."""

    def __init__(self, clip_id: str) -> None:
        super().__init__(f"Clip '{clip_id}' was not found")
        self.clip_id = clip_id


class ClipReviewNotFoundError(DomainError):
    """Raised when no ``clip_reviews`` row exists for a clip."""

    def __init__(self, clip_id: str) -> None:
        super().__init__(f"Clip review for '{clip_id}' was not found")
        self.clip_id = clip_id


class PlayerNotFoundError(DomainError):
    """Raised when no ``players`` row exists for ``(player_id, club_id)``.

    Not retryable: the player is genuinely absent, so the consumer should drop the
    message rather than nack it.
    """

    def __init__(self, player_id: str) -> None:
        super().__init__(f"Player '{player_id}' was not found")
        self.player_id = player_id


class SportCatalogNotFoundError(DomainError):
    """Raised when no sport catalog resolves for ``(sport_id, club_id)``.

    Not retryable: the club_sports/sport_catalog join returned nothing, so the
    consumer should drop the message.
    """

    def __init__(self, sport_id: str) -> None:
        super().__init__(f"No sport catalog for club_sport '{sport_id}'")
        self.sport_id = sport_id


class VideoNotReadyError(DomainError):
    """Raised when the downsized video for a clip is not yet available.

    This is *retryable*: the upstream downsizing may still be in progress, so the
    consumer should nack the message and let Pub/Sub redeliver later.
    """

    def __init__(self, clip_id: str) -> None:
        super().__init__(f"Downsized video for clip '{clip_id}' is not ready yet")
        self.clip_id = clip_id


class LocalVideoNotFoundError(DomainError):
    """Raised in local mode when a clip's video file is missing from the media directory.

    Local mode maps ``gs://<bucket>/<key>`` to ``<SCOUT_LOCAL_MEDIA_DIR>/<bucket>/<key>``.
    """

    def __init__(self, uri: str, path: str) -> None:
        super().__init__(
            f"Local video for {uri} not found at {path}. "
            "Place the file there (see SCOUT_LOCAL_MEDIA_DIR)."
        )
        self.uri = uri
        self.path = path
