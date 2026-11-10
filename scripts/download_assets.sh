#!/usr/bin/env bash
#
# Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
# SPDX-License-Identifier: Apache-2.0
#
# Download the pinned avatar assets; arguments go to download_resources.py.

set -euo pipefail
exec python3 "$(cd -- "$(dirname -- "$0")" && pwd)/download_resources.py" assets "$@"
