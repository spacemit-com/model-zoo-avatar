#!/usr/bin/env python3
# Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Render a persona speaking a WAV file into 25 fps PNG frames."""

import argparse

from spacemit_avatar import AvatarService


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="configs/feifan.yaml or configs/zihan.yaml")
    parser.add_argument("--audio", required=True, help="16 kHz mono PCM16 WAV")
    parser.add_argument("--output", required=True, help="empty output directory")
    parser.add_argument("--max-frames", type=int, default=0, help="stop after N frames; 0 renders the whole file")
    parser.add_argument("--worker", help="avatar_worker path; defaults to the one bundled with the package")
    args = parser.parse_args()
    with AvatarService(args.config, args.worker) as engine:
        result = engine.render_file(args.audio, args.output, args.max_frames)
    print(f"{result.frames} frames saved to {result.output_dir} at 25fps")


if __name__ == "__main__":
    main()
