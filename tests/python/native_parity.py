# Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Real graph parity with the extracted Python feature server and CPU UNet reference.

Only the reference subprocess imports Python ONNX Runtime in UNet mode.
"""

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
import numpy as np


def native():
    try:
        import _spacemit_avatar as n
    except ModuleNotFoundError as e:
        if e.name != "_spacemit_avatar":
            raise
        from spacemit_avatar import _spacemit_avatar as n
    return n


def features(args):
    import onnxruntime as ort

    spec = importlib.util.spec_from_file_location(
        "reference", Path(__file__).parents[1] / "reference/feature_reference.py"
    )
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(args.model, sess_options=opts, providers=["CPUExecutionProvider"])
    n = native()
    options = n.RuntimeOptions()
    extractor = n.FeatherHubertExtractor(args.model, options)
    results = []
    rng = np.random.default_rng(1234)
    if args.wav:
        with wave.open(args.wav) as w:
            if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (16000, 1, 2):
                raise ValueError("Expected 16k PCM16 mono")
            speech = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").copy()
    else:
        t = np.arange(32000) / 16000
        speech = (8000 * np.sin(2 * np.pi * 220 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t))).astype(np.int16)
    for kind, pcm in [
        ("speech", speech),
        ("silence", np.zeros(32000, np.int16)),
        ("noise", rng.integers(-1000, 1000, 32000, dtype=np.int16)),
    ]:
        for chunk in [1600, 12000, len(pcm)]:
            extractor.reset()
            baseline = ref.FeatherFeatureSession(session)
            old = []
            new = []
            ids = []
            for at in range(0, len(pcm), chunk):
                block = pcm[at : at + chunk]
                final = at + chunk >= len(pcm)
                r, _ = baseline.push(block.tobytes(), final)
                old.append(r.reshape(-1, 16, 32, 32))
                c = extractor.push_pcm(block, final)
                new.append(c.windows)
                ids.extend(c.frame_ids)
            x = np.concatenate(old)
            y = np.concatenate(new)
            assert x.shape == y.shape, (kind, chunk, x.shape, y.shape)
            assert ids == list(range(len(ids)))
            error = float(np.max(np.abs(x - y))) if x.size else 0
            rmse = float(np.sqrt(np.mean((x - y) ** 2))) if x.size else 0
            np.testing.assert_allclose(y, x, atol=2e-5, rtol=1e-5)
            if kind == "silence":
                assert not np.any(y)
            results.append({"audio": kind, "chunk_samples": chunk, "frames": len(ids), "max_abs": error, "rmse": rmse})
    # Ownership/dtype and finalized-state behavior, beyond the original protocol.
    try:
        extractor.push_pcm(np.zeros(32, np.float32))
        raise AssertionError("float PCM accepted")
    except ValueError:
        pass
    try:
        extractor.push_pcm(np.zeros(32, np.int16))
        raise AssertionError("post-final push accepted")
    except RuntimeError:
        pass
    extractor.reset()
    pcm = np.ndarray(speech.shape, dtype=np.int16, buffer=bytearray(speech.nbytes + 1), offset=1)
    pcm[:] = speech
    unaligned = extractor.push_pcm(pcm, True).windows
    extractor.reset()
    np.testing.assert_array_equal(unaligned, extractor.push_pcm(speech, True).windows)
    return {"ok": True, "mode": "feature", "cases": results}


def reference_unet(args):
    import onnxruntime as ort

    data = np.load(args.inputs)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(args.model, sess_options=opts, providers=["CPUExecutionProvider"])
    np.save(args.reference, session.run(["output"], {"input": data["image"], "audio": data["audio"]})[0])


def unet(args):
    import cv2

    n = native()
    if args.dataset:
        dataset = Path(args.dataset)
        bgr = cv2.imread(str(dataset / "full_body_img/0.jpg"))
        points = np.loadtxt(dataset / "landmarks/0.lms", dtype=np.float32)
    else:
        y, x = np.indices((240, 240))
        # Fine texture separates the direct and residual compositors; a smooth
        # gradient would make them agree within rounding.
        texture = np.random.default_rng(7).integers(0, 64, (240, 240, 3))
        bgr = np.clip(np.stack((x, y, (x + y) // 2), axis=-1) * 3 // 4 + texture, 0, 255).astype(np.uint8)
        points = np.full((110, 2), 120, dtype=np.float32)
        points[[1, 31, 52]] = [[30, 120], [210, 120], [120, 30]]
        points[90:] = np.column_stack((np.arange(100, 140, 2), 150 + np.arange(20) % 3))
    frame = n.PreparedFrame(bgr, points)
    x, y, w, h = frame.face_rect
    face = cv2.resize(bgr[y : y + h, x : x + w], (168, 168), interpolation=cv2.INTER_AREA)
    inner = face[4:164, 4:164]
    masked = inner.copy()
    masked[5:150, 5:155] = 0
    expected = np.ascontiguousarray(
        np.concatenate([inner.transpose(2, 0, 1), masked.transpose(2, 0, 1)], axis=0).astype(np.float32)[None] / 255
    )
    np.testing.assert_array_equal(frame.image_tensor, expected)
    audio = np.random.default_rng(123).normal(0, 0.15, (1, 16, 32, 32)).astype(np.float32)
    options = n.RuntimeOptions()
    options.provider = args.provider
    options.ep_threads = 2
    if args.profile:
        options.profile_prefix = str(Path(args.profile).resolve())
    with tempfile.TemporaryDirectory() as t:
        inputs = Path(t) / "input.npz"
        reference = Path(t) / "reference.npy"
        np.savez(inputs, image=expected, audio=audio)
        subprocess.run(
            [
                sys.executable,
                __file__,
                "--mode",
                "reference-unet",
                "--model",
                args.model,
                "--inputs",
                str(inputs),
                "--reference",
                str(reference),
            ],
            check=True,
        )
        baseline = np.load(reference)
        model = n.UnetLipGenerator(args.model, options)
        output = model.infer(expected, audio)
        assert output.shape == (1, 3, 160, 160) and np.isfinite(output).all()
        diff = output - baseline
        rmse = float(np.sqrt(np.mean(diff**2)))
        max_abs = float(np.max(np.abs(diff)))
        if args.provider == "cpu":
            np.testing.assert_allclose(output, baseline, atol=1e-5, rtol=1e-5)
        else:
            assert rmse < 0.01, (rmse, max_abs)
        result = frame.compose(output)
        assert result.shape == bgr.shape
        outside = np.ones(bgr.shape[:2], bool)
        outside[y : y + h, x : x + w] = False
        np.testing.assert_array_equal(result[outside], bgr[outside])
        # Independent NumPy/OpenCV reference for the default native-alpha compositor. Like
        # bianbu-avatar, native paste-back replaces the face directly instead of adding a residual.
        generated = face.copy()
        generated[4:164, 4:164] = np.clip(
            np.floor(np.clip(output[0].transpose(1, 2, 0), 0, 1) * 255 + 0.5), 0, 255
        ).astype(np.uint8)
        gen = cv2.resize(generated, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        dst = bgr[y : y + h, x : x + w].astype(np.float32)
        margin = np.clip(int(np.floor(min(w, h) * 0.1 + 0.5)), 2, 64)
        xx = np.minimum(np.arange(w, dtype=np.float32) + 0.5, w - 0.5 - np.arange(w, dtype=np.float32))
        yy = np.minimum(np.arange(h, dtype=np.float32) + 0.5, h - 0.5 - np.arange(h, dtype=np.float32))
        alpha = np.clip(yy / margin, 0, 1)[:, None] * np.clip(xx / margin, 0, 1)[None, :]
        expected_image = bgr.copy()
        expected_image[y : y + h, x : x + w] = np.clip(np.rint(dst + alpha[:, :, None] * (gen - dst)), 0, 255).astype(
            np.uint8
        )
        pixel = int(np.abs(result.astype(np.int16) - expected_image.astype(np.int16)).max())
        assert pixel <= 1, pixel
        try:
            model.infer(expected[:, :, :, :159].copy(), audio)
            raise AssertionError("Bad shape accepted")
        except ValueError:
            pass
        return {
            "ok": True,
            "mode": "unet",
            "provider": args.provider,
            "max_abs": max_abs,
            "rmse": rmse,
            "preprocess_exact": True,
            "composite_max_pixel_difference": pixel,
            "outside_face_exact": True,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["feature", "unet", "reference-unet"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--wav")
    parser.add_argument("--dataset")
    parser.add_argument("--provider", choices=["cpu", "spacemit"], default="cpu")
    parser.add_argument("--profile")
    parser.add_argument("--output")
    parser.add_argument("--inputs")
    parser.add_argument("--reference")
    args = parser.parse_args()
    if args.mode == "reference-unet":
        reference_unet(args)
        return
    result = features(args) if args.mode == "feature" else unet(args)
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
