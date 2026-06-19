"""coordpy.adk.artifacts — named, versioned binary artifacts.

An ``Artifact`` is named, versioned binary data (a report, an image, a
note). The filename encodes scope: a plain name (``"summary.md"``) is
session-scoped; a ``user:`` prefix (``"user:profile.json"``) is shared
across that user's sessions. ``InMemoryArtifactService`` is the default;
saves are also surfaced as ARTIFACT capsules by the Runner's CapsuleTrail,
so every artifact is content-addressed and on-disk-verifiable.
"""

from __future__ import annotations

import dataclasses

USER_PREFIX = "user:"


@dataclasses.dataclass(frozen=True)
class Artifact:
    """A versioned binary blob with a MIME type."""

    name: str
    data: bytes
    mime_type: str = "application/octet-stream"
    version: int = 0

    @property
    def size(self) -> int:
        return len(self.data)


class BaseArtifactService:
    def save_artifact(self, *, app_name: str, user_id: str, session_id: str,
                      filename: str, data: bytes,
                      mime_type: str = "application/octet-stream") -> int:
        raise NotImplementedError

    def load_artifact(self, *, app_name: str, user_id: str, session_id: str,
                      filename: str, version: int | None = None) -> Artifact | None:
        raise NotImplementedError

    def list_artifact_keys(self, *, app_name: str, user_id: str,
                           session_id: str) -> list[str]:
        raise NotImplementedError


class InMemoryArtifactService(BaseArtifactService):
    """Process-local artifact store with per-name version history."""

    def __init__(self) -> None:
        # key -> ordered list of Artifact versions (version == index).
        self._store: dict[tuple, list[Artifact]] = {}

    def _key(self, app_name: str, user_id: str, session_id: str,
             filename: str) -> tuple:
        if filename.startswith(USER_PREFIX):
            return (app_name, user_id, "user", filename)
        return (app_name, user_id, session_id, filename)

    def save_artifact(self, *, app_name: str, user_id: str, session_id: str,
                      filename: str, data: bytes,
                      mime_type: str = "application/octet-stream") -> int:
        key = self._key(app_name, user_id, session_id, filename)
        versions = self._store.setdefault(key, [])
        version = len(versions)
        versions.append(Artifact(name=filename, data=bytes(data),
                                 mime_type=mime_type, version=version))
        return version

    def load_artifact(self, *, app_name: str, user_id: str, session_id: str,
                      filename: str, version: int | None = None) -> Artifact | None:
        key = self._key(app_name, user_id, session_id, filename)
        versions = self._store.get(key)
        if not versions:
            return None
        if version is None:
            return versions[-1]
        if 0 <= version < len(versions):
            return versions[version]
        return None

    def list_artifact_keys(self, *, app_name: str, user_id: str,
                           session_id: str) -> list[str]:
        out: list[str] = []
        for (a, u, scope, filename) in self._store:
            if a != app_name or u != user_id:
                continue
            if scope == "user" or scope == session_id:
                out.append(filename)
        return sorted(set(out))
