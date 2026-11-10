#!/usr/bin/env python3
# Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Download only the pinned two-model resources; never extract arbitrary tar paths."""

import argparse
import contextlib
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
import wave
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for data in iter(lambda: f.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def valid(path, item):
    return path.is_file() and path.stat().st_size == item["size"] and digest(path) == item["sha256"]


def target(root, relative):
    p = PurePosixPath(relative)
    if p.is_absolute() or not p.parts or any(x in ("..", ".") for x in p.parts) or "\\" in relative:
        raise ValueError("Unsafe resource path")
    result = root.joinpath(*p.parts)
    if not result.resolve().is_relative_to(root.resolve()):
        raise ValueError("Resource path escapes destination")
    return result


def download(item, cache, offline=False):
    sha = item["sha256"]
    if not re.fullmatch("[0-9a-f]{64}", sha):
        raise ValueError("Invalid SHA256")
    cache.mkdir(parents=True, exist_ok=True)
    dest = cache / sha
    with (cache / (sha + ".lock")).open("a+b") as lock:
        if os.name == "nt":
            import msvcrt

            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return download_locked(item, dest, offline)


def download_locked(item, dest, offline):
    if dest.exists():
        if valid(dest, item):
            return dest
        raise ValueError("Cached file checksum mismatch: " + str(dest))
    part = dest.with_suffix(".part")
    if valid(part, item):
        os.replace(part, dest)
        return dest
    if offline:
        raise FileNotFoundError("Verified download is not cached: " + dest.name)
    start = part.stat().st_size if part.exists() else 0
    if start >= item["size"]:
        start = 0
    req = urllib.request.Request(item["url"], headers={"Range": f"bytes={start}-"} if start else {})
    with urllib.request.urlopen(req, timeout=60) as response:
        resumed = start and response.status == 206
        if resumed and not response.headers.get("Content-Range", "").startswith(f"bytes {start}-"):
            raise ValueError("Invalid resume range")
        size = start if resumed else 0
        with part.open("ab" if resumed else "wb") as f:
            for block in iter(lambda: response.read(1024 * 1024), b""):
                size += len(block)
                if size > item["size"]:
                    raise ValueError("Download exceeds declared size")
                f.write(block)
    if not valid(part, item):
        raise ValueError("Downloaded content failed size/SHA256 validation")
    os.replace(part, dest)
    return dest


@contextlib.contextmanager
def open_archive(path, kind):
    if kind == "tar.zst":
        try:
            import zstandard
        except ImportError:
            exe = shutil.which("zstd")
            if not exe:
                raise RuntimeError("Install zstandard Python module or zstd command")
            proc = subprocess.Popen([exe, "-dc", "--", str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                with tarfile.open(fileobj=proc.stdout, mode="r|") as t:
                    yield t
                while proc.stdout.read(1024 * 1024):
                    pass
                error = proc.stderr.read().decode(errors="replace")
                if proc.wait():
                    raise ValueError("zstd failed: " + error)
            finally:
                if proc.poll() is None:
                    proc.terminate()
                proc.stdout.close()
                proc.stderr.close()
                proc.wait()
        else:
            with path.open("rb") as f, zstandard.ZstdDecompressor().stream_reader(f) as reader, tarfile.open(
                fileobj=reader, mode="r|"
            ) as t:
                yield t
    elif kind == "tar.gz":
        with tarfile.open(path, mode="r|gz") as t:
            yield t
    else:
        raise ValueError("Unsupported archive type")


def install(archive, kind, entries, root, force=False):
    desired = {}
    seen = set()
    for e in entries:
        p = target(root, e["dest"])
        if p.exists():
            if valid(p, e):
                continue
            if not force:
                raise FileExistsError("Existing resource differs; retained: " + str(p) + " (use --force to replace)")
        if e["member"] in desired:
            raise ValueError("Duplicate manifest member")
        desired[e["member"]] = (e, p)
    if not desired:
        return
    with open_archive(archive, kind) as stream:
        for member in stream:
            if member.name not in desired:
                continue
            if member.name in seen:
                raise ValueError("Duplicate selected archive member")
            seen.add(member.name)
            entry, dest = desired[member.name]
            if not member.isfile() or member.size != entry["size"]:
                raise ValueError("Selected member is not a regular file of the declared size")
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.resolve().is_relative_to(root.resolve()):
                raise ValueError("Destination changed while extracting")
            fd, temp = tempfile.mkstemp(prefix=dest.name + ".", suffix=".part", dir=dest.parent)
            try:
                source = stream.extractfile(member)
                with os.fdopen(fd, "wb") as out:
                    shutil.copyfileobj(source, out, 1024 * 1024)
                if not valid(Path(temp), entry):
                    raise ValueError("Archive member checksum mismatch: " + member.name)
                if force:
                    os.replace(temp, dest)
                else:
                    try:
                        os.link(temp, dest)
                    except FileExistsError:
                        if not valid(dest, entry):
                            raise
            finally:
                if Path(temp).exists():
                    Path(temp).unlink()
    if seen != set(desired):
        raise ValueError("Selected members missing from archive: " + str(sorted(set(desired) - seen)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["models", "assets"])
    parser.add_argument("--persona", choices=["feifan", "zihan", "all"], default="feifan")
    parser.add_argument("--model", choices=["hubert", "unet", "all"], default="all")
    parser.add_argument("--frames", type=int, default=32, help="Asset frame limit; 0 downloads all paired frames")
    parser.add_argument("--manifest", type=Path, default=ROOT / "config/resources.json")
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache/avatar/downloads")
    parser.add_argument("--models-dir", type=Path)
    parser.add_argument("--assets-dir", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    data = json.loads(args.manifest.read_text(encoding="utf-8"))
    if data.get("schema") != 1 or args.frames < 0:
        raise ValueError("Invalid manifest schema or frame limit")
    root = (
        (args.models_dir or Path.home() / ".cache/models/avatar")
        if args.kind == "models"
        else (args.assets_dir or Path.home() / ".cache/assets/avatar")
    )
    root = root.expanduser().resolve()
    entries = []
    if args.kind == "models":
        if args.model in ("hubert", "all"):
            entries.append(data["models"]["hubert"])
        if args.model in ("unet", "all"):
            for persona in ["feifan", "zihan"] if args.persona == "all" else [args.persona]:
                entries.append(data["models"]["unet_" + persona])
    else:
        for persona in ["feifan", "zihan"] if args.persona == "all" else [args.persona]:
            for e in data["assets"][persona]:
                if not args.frames or int(PurePosixPath(e["member"]).stem) < args.frames:
                    entries.append(e)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "kind": args.kind,
                    "root": str(root),
                    "selected_files": len(entries),
                    "archives": sorted({e["archive"] for e in entries}),
                    "audio": list(data.get("audio", {})) if args.kind == "assets" else [],
                },
                indent=2,
            )
        )
        return
    root.mkdir(parents=True, exist_ok=True)
    for archive_id in sorted({e["archive"] for e in entries}):
        selected = [e for e in entries if e["archive"] == archive_id]
        # Fully installed resources work offline even after archive cache removal.
        if all(valid(target(root, e["dest"]), e) for e in selected):
            continue
        spec = data["archives"][archive_id]
        cached = download(spec, args.cache_dir.expanduser(), args.offline)
        install(cached, spec["kind"], selected, root, args.force)
    if args.kind == "assets":
        for name, item in data.get("audio", {}).items():
            dest = target(root, "audio/" + name)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if valid(dest, item):
                continue
            if dest.exists() and not args.force:
                raise FileExistsError("Existing audio differs; retained: " + str(dest))
            if item.get("generated") == "silence_16k_1s":
                fd, tmp = tempfile.mkstemp(dir=dest.parent, suffix=".wav")
                os.close(fd)
                try:
                    with wave.open(tmp, "wb") as w:
                        w.setparams((1, 2, 16000, 0, "NONE", ""))
                        w.writeframes(bytes(32000))
                    if not valid(Path(tmp), item):
                        raise ValueError("Generated silence checksum mismatch")
                    os.replace(tmp, dest)
                finally:
                    if Path(tmp).exists():
                        Path(tmp).unlink()
            else:
                cached = download(item, args.cache_dir.expanduser(), args.offline)
                fd, tmp = tempfile.mkstemp(dir=dest.parent, suffix=".part")
                os.close(fd)
                try:
                    shutil.copyfile(cached, tmp)
                    os.replace(tmp, dest)
                finally:
                    if Path(tmp).exists():
                        Path(tmp).unlink()
    print(json.dumps({"ok": True, "kind": args.kind, "files": len(entries), "root": str(root)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
