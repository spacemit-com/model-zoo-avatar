#!/usr/bin/env python3
# Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Require observed EP execution and no CPU Resize events in an ORT JSON profile."""

import argparse
import json
from pathlib import Path


def inspect(path):
    events = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(events, list):
        raise ValueError("Expected ORT event list")
    cpu = sum(
        1
        for e in events
        if e.get("args", {}).get("provider") == "CPUExecutionProvider" and e.get("args", {}).get("op_name") == "Resize"
    )
    ep = sum(1 for e in events if e.get("args", {}).get("provider") == "SpaceMITExecutionProvider")
    return {"ok": cpu == 0 and ep > 0, "cpu_resize_events": cpu, "ep_events": ep}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = inspect(args.profile)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))
    raise SystemExit(0 if result["ok"] else 2)


if __name__ == "__main__":
    main()
