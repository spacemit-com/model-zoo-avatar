# Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Two-model SDK. Native backends are loaded lazily; no model downloads on import."""

from .pipeline import AvatarService, RenderResult

_NATIVE = {
    "RuntimeOptions",
    "RenderOptions",
    "FeatureBatch",
    "FeatherHubertExtractor",
    "UnetLipGenerator",
    "PreparedFrame",
    "runtime_version",
}


def __getattr__(name):
    if name == "__version__":
        from importlib.metadata import version, PackageNotFoundError

        try:
            return version("spacemit-avatar")
        except PackageNotFoundError:
            from pathlib import Path

            source = Path(__file__).resolve().parents[2] / "VERSION"
            return source.read_text().strip() if source.exists() else "0+unknown"
    if name in _NATIVE:
        from . import _spacemit_avatar

        return getattr(_spacemit_avatar, name)
    raise AttributeError(name)


__all__ = ["AvatarService", "RenderResult", *sorted(_NATIVE)]
