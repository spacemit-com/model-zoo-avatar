# Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Offline file rendering through the avatar_worker extract and render stages."""

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading

import yaml


@dataclass(frozen=True)
class RenderResult:
    output_dir: Path
    frames: int
    extract_metrics: dict
    render_metrics: dict


class AvatarService:
    """Run an offline render in two native workers with caller-supplied assets."""

    def __init__(self, config, worker=None):
        self.config = Path(config).expanduser().resolve()
        with self.config.open(encoding="utf-8") as stream:
            self.options = yaml.safe_load(stream)
        if not isinstance(self.options, dict) or self.options.get("schema") != 1:
            raise ValueError("Expected config schema=1")
        for stage in ("hubert", "unet"):
            section = self.options.get(stage)
            if not isinstance(section, dict) or not isinstance(section.get("model"), str):
                raise ValueError(f"Missing {stage} model configuration")
            timeout = float(section.get("timeout_s", 3600))
            if not math.isfinite(timeout) or not 0 < timeout <= 86400:
                raise ValueError("Worker timeout must be finite, positive, and at most 86400 seconds")
        bundled_worker = Path(__file__).resolve().parent / "bin" / "avatar_worker"
        selected = worker or os.environ.get("AVATAR_WORKER")
        if not selected:
            selected = str(bundled_worker) if bundled_worker.is_file() else "avatar_worker"
        self.worker = shutil.which(os.path.expanduser(selected))
        if not self.worker:
            raise FileNotFoundError(f"Native worker not found or executable: {selected}")
        self.worker = str(Path(self.worker).resolve())
        self._state_lock = threading.Lock()
        self._child = None
        self._active = False
        self._cancelled = False
        self._closed = False

    def _run(self, stage, args):
        with self._state_lock:
            if self._cancelled or self._closed:
                raise RuntimeError("Rendering cancelled or service closed")
            child = subprocess.Popen(
                [self.worker, *args, "--config", str(self.config)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            self._child = child
        try:
            out, err = child.communicate(timeout=float(self.options[stage].get("timeout_s", 3600)))
        except subprocess.TimeoutExpired as exc:
            child.terminate()
            try:
                child.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.communicate()
            raise TimeoutError(f"{stage} worker timed out") from exc
        finally:
            # A caller interrupt must also reap the worker before releasing the slot.
            if child.poll() is None:
                child.kill()
                child.communicate()
            with self._state_lock:
                self._child = None
        with self._state_lock:
            if self._cancelled:
                raise RuntimeError("Rendering cancelled")
        if child.returncode:
            raise RuntimeError(f"{stage} worker exit {child.returncode}: {err[-4000:]}")
        lines = [line for line in out.splitlines() if line.startswith("{")]
        if not lines:
            raise RuntimeError("Native worker returned no result")
        result = json.loads(lines[-1])
        if not result.get("ok"):
            raise RuntimeError("Native worker rejected request")
        return result

    def render_file(self, audio_path, output_dir, max_frames=0):
        with self._state_lock:
            if self._closed:
                raise RuntimeError("Service closed")
            if self._active:
                raise RuntimeError("Service already rendering")
            self._active = True
            self._cancelled = False
        try:
            audio = Path(audio_path).expanduser().resolve()
            out = Path(output_dir).expanduser().resolve()
            if not audio.is_file():
                raise FileNotFoundError(audio)
            if not isinstance(max_frames, int) or max_frames < 0:
                raise ValueError("max_frames must be nonnegative")
            if out.exists() and (not out.is_dir() or any(out.iterdir())):
                raise FileExistsError("Output directory must be empty")
            with tempfile.TemporaryDirectory(prefix="avatar-") as temp:
                features = Path(temp) / "audio.ftf"
                first = self._run("hubert", ["extract", "--audio", str(audio), "--output", str(features)])
                if first["frames"] < 1:
                    raise ValueError("Audio is too short to produce feature windows")
                second = self._run(
                    "unet",
                    ["render", "--features", str(features), "--output", str(out), "--max-frames", str(max_frames)],
                )
                expected = min(first["frames"], max_frames) if max_frames else first["frames"]
                if second["frames"] != expected:
                    raise RuntimeError("Native frame count mismatch")
                return RenderResult(out, second["frames"], first, second)
        finally:
            with self._state_lock:
                self._active = False

    def _cancel_locked(self):
        self._cancelled = True
        if self._child is not None and self._child.poll() is None:
            self._child.terminate()

    def cancel(self):
        with self._state_lock:
            self._cancel_locked()

    def close(self):
        with self._state_lock:
            self._closed = True
            self._cancel_locked()
            child = self._child
        if child is not None:
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
