#!/usr/bin/env python3
# Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Render a persona speaking a WAV file into 25 fps PNG frames and an MP4 video.

Models, assets and test audio default to the downloader cache. Output defaults
to output/<persona>-python-frames/ and output/<persona>-python.mp4.
"""

import argparse
import shutil
import subprocess
from pathlib import Path

from spacemit_avatar import AvatarService

COMPONENT = Path(__file__).resolve().parents[2]


def encode_video(frames, audio, video):
    """Mux the frames with the audio. Returns False when ffmpeg is not installed."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-n", "-framerate", "25",
         "-i", str(Path(frames) / "%08d.png"), "-i", str(audio),
         "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(video)],
        check=True,
    )  # fmt: skip
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--persona", choices=["feifan", "zihan"], default="feifan")
    parser.add_argument("--audio", type=Path, help="16 kHz mono PCM16 WAV; defaults to <persona>_test.wav")
    parser.add_argument("--output", type=Path, help="empty frame directory")
    parser.add_argument("--video", type=Path, help="MP4 path; defaults to <persona>-python.mp4 beside the frames")
    parser.add_argument("--no-video", action="store_true", help="only write PNG frames")
    parser.add_argument("--max-frames", type=int, default=0, help="stop after N frames; 0 renders the whole file")
    parser.add_argument("--config", type=Path, help="defaults to configs/<persona>.yaml")
    parser.add_argument("--worker", help="avatar_worker path; defaults to the one bundled with the package")
    args = parser.parse_args()

    config = args.config or COMPONENT / "configs" / f"{args.persona}.yaml"
    audio = args.audio or Path.home() / ".cache/assets/avatar/audio" / f"{args.persona}_test.wav"
    output = args.output or Path("output") / f"{args.persona}-python-frames"
    video = args.video or output.parent / f"{args.persona}-python.mp4"
    if not args.no_video and video.exists():
        raise SystemExit(f"Video already exists: {video}")

    with AvatarService(config, args.worker) as engine:
        result = engine.render_file(audio, output, args.max_frames)
    print(f"{result.frames} frames saved to {output}")
    if args.no_video:
        return
    if encode_video(output, audio, video):
        print(f"Video saved to {video}")
    else:
        print(f"ffmpeg not found; install ffmpeg to also get {video}")


if __name__ == "__main__":
    main()
