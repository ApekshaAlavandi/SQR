"""Local mode: run scout-ai's flows from a command line with only Postgres as a dependency.

Replaces the two cloud edges and nothing else:

* Pub/Sub  -> the ``scout-ai-local`` CLI hands the same JSON payloads to the same flow
  processors (``scout_ai.transport.flows``) that the Pub/Sub worker uses.
* GCS      -> ``gs://<bucket>/<key>`` is read from ``<SCOUT_LOCAL_MEDIA_DIR>/<bucket>/<key>``.
* Gemini   -> optional; defaults to an offline stub (``SCOUT_AI_BACKEND=stub``).
"""
