from __future__ import annotations

import argparse
import os
import socketserver
import struct
import time

import numpy as np
import onnxruntime as ort


TOKENS_PER_FRAME = 2
FEATURE_CHANNELS = 2
FEATURE_HEIGHT = 32
FEATURE_WIDTH = 32
WINDOW_FRAMES = 8
LEFT_CONTEXT = 4
RIGHT_CONTEXT = 4
SAMPLES_PER_FRAME = 640
INFERENCE_AUDIO_SECONDS = 1.25
MAX_AUDIO_SECONDS = max(
    0.5,
    float(os.getenv("FEATHER_FEATURE_MAX_AUDIO_SECONDS", str(INFERENCE_AUDIO_SECONDS))),
)
# Run one inference per retained one-second context.  Smaller chunks preserve
# correctness but repeat the same ONNX work many times and make feature
# extraction slower than real time on K3.
FEATURE_INPUT_CHUNK_SAMPLES = 12000
# Digitally silent audio must never be normalized.  The rolling-context
# normalization divides by sqrt(variance + 1e-7), so a silent context -- whose
# variance is the noise floor -- is scaled up to full range; the model reads that
# as loud speech and opens the mouth, and the renderer's zero-audio release then
# closes it again.  The booth sees that as "the mouth opens and closes after the
# character stops talking".  All-zero features are the closed-mouth reference, so
# anything below this standard deviation is emitted as zeros instead.  1e-3 is
# about -60 dBFS: far below the quietest speech the TTS produces, far above the
# 1/32768 LSB of a digital zero.
SILENCE_STD = float(os.getenv("FEATHER_FEATURE_SILENCE_STD", "1e-3"))
REQUEST_HEADER = struct.Struct("!BI")
RESPONSE_HEADER = struct.Struct("!If")
FLAG_FINAL = 1
FLAG_RESET = 2


class FeatherFeatureSession:
    def __init__(self, ort_session: ort.InferenceSession):
        self.ort_session = ort_session
        self.input_name = ort_session.get_inputs()[0].name
        self.reset()

    def reset(self) -> None:
        self.audio = np.empty(0, dtype=np.float32)
        self.emitted_global_frames = 0
        self.total_samples = 0
        self.silence_reported = False

    def push(self, pcm_bytes: bytes, final: bool) -> tuple[np.ndarray, float]:
        """Append PCM and emit every frame exactly once in global time.

        TTS providers may return a several-second WAV in one protocol message.
        The model only needs a one-second sliding context, but the old
        implementation used a local frame counter and silently discarded the
        portion of a large message before the retained context.  Feed large
        messages through the same incremental path as 100-ms audio and keep a
        global frame cursor so no audio/feature interval is skipped.
        """
        started = time.perf_counter()
        outputs: list[np.ndarray] = []
        if pcm_bytes:
            step = FEATURE_INPUT_CHUNK_SAMPLES * 2
            for offset in range(0, len(pcm_bytes), step):
                chunk = pcm_bytes[offset:offset + step]
                chunk_final = final and offset + len(chunk) >= len(pcm_bytes)
                windows = self._push_chunk(chunk, chunk_final)
                if windows.size:
                    outputs.append(windows)
        else:
            windows = self._push_chunk(b"", final)
            if windows.size:
                outputs.append(windows)
        if outputs:
            result = np.concatenate(outputs, axis=0)
        else:
            result = np.empty((0, WINDOW_FRAMES, FEATURE_CHANNELS, FEATURE_HEIGHT, FEATURE_WIDTH), np.float32)
        return result, (time.perf_counter() - started) * 1000.0

    def _push_chunk(self, pcm_bytes: bytes, final: bool) -> np.ndarray:
        if pcm_bytes:
            pcm = np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float32) / 32768.0
            self.audio = np.concatenate((self.audio, pcm))
            self.total_samples += pcm.size
            max_samples = int(MAX_AUDIO_SECONDS * 16000)
            if self.audio.size > max_samples:
                self.audio = self.audio[-max_samples:].copy()
        if self.audio.size < 400:
            return np.empty((0, WINDOW_FRAMES, FEATURE_CHANNELS, FEATURE_HEIGHT, FEATURE_WIDTH), np.float32)

        base_sample = max(0, self.total_samples - self.audio.size)
        variance = float(self.audio.var())
        # See SILENCE_STD: a silent context must not be scaled to full range.
        silent = np.sqrt(variance) < SILENCE_STD
        # ``self.audio`` is already the retained one-second context.  Do not
        # slice it again: doing so shifts the global frame origin forward by
        # another 24 frames and makes every subsequent update look like a
        # duplicate/overlong window.
        speech_audio = self.audio
        speech = (speech_audio - float(speech_audio.mean())) / np.sqrt(variance + 1e-7)
        hidden = self.ort_session.run(
            None,
            {self.input_name: speech.astype(np.float32, copy=False)[None]},
        )[0][0]
        if hidden.shape[0] % 2 == 1:
            hidden = hidden[:-1]
        if hidden.shape[0] < TOKENS_PER_FRAME:
            return np.empty((0, WINDOW_FRAMES, FEATURE_CHANNELS, FEATURE_HEIGHT, FEATURE_WIDTH), np.float32)

        frame_features = hidden.reshape(-1, FEATURE_CHANNELS, FEATURE_HEIGHT, FEATURE_WIDTH)
        if silent:
            # Keep the model's frame count so every tick still lines up with the
            # audio, but emit the closed-mouth reference instead of the features
            # of amplified noise.
            frame_features = np.zeros_like(frame_features)
            if not self.silence_reported:
                self.silence_reported = True
                print(
                    f"FEATURE silence guard: std<{SILENCE_STD:g} emitted zeros "
                    f"(context {self.audio.size} samples)",
                    flush=True,
                )
        base_frame = base_sample // SAMPLES_PER_FRAME
        available_end = base_frame + frame_features.shape[0]
        ready_end = available_end if final else max(base_frame, available_end - RIGHT_CONTEXT)
        available_global_end = (self.total_samples + SAMPLES_PER_FRAME - 1) // SAMPLES_PER_FRAME if final else self.total_samples // SAMPLES_PER_FRAME
        ready_global_end = min(ready_end, available_global_end)
        emit_start = max(self.emitted_global_frames, base_frame)
        if ready_global_end <= emit_start:
            return np.empty((0, WINDOW_FRAMES, FEATURE_CHANNELS, FEATURE_HEIGHT, FEATURE_WIDTH), np.float32)

        padded = np.pad(frame_features, ((LEFT_CONTEXT, RIGHT_CONTEXT), (0, 0), (0, 0), (0, 0)))
        windows = np.stack(
            [padded[index:index + WINDOW_FRAMES] for index in range(emit_start - base_frame, ready_global_end - base_frame)],
            axis=0,
        ).astype(np.float32, copy=False)
        self.emitted_global_frames = ready_global_end
        return windows


class FeatureRequestHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        session = FeatherFeatureSession(self.server.ort_session)
        connection_id = f"{self.client_address[0]}:{self.client_address[1]}"
        frame_seq = 0
        while True:
            header = self._read_exact(REQUEST_HEADER.size)
            if not header:
                return
            frame_seq += 1
            try:
                flags, payload_size = REQUEST_HEADER.unpack(header)
                if flags & ~(FLAG_FINAL | FLAG_RESET):
                    raise ValueError(f"invalid flags=0x{flags:02x}")
                if payload_size > 8 * 1024 * 1024:
                    raise ValueError(f"PCM packet too large: {payload_size}")
                if payload_size % 2:
                    raise ValueError(f"PCM16 payload has odd byte length: {payload_size}")
                payload = self._read_exact(payload_size)
                if payload is None:
                    return
                if flags & FLAG_RESET:
                    session.reset()
                windows, elapsed_ms = session.push(payload, bool(flags & FLAG_FINAL))
                self.request.sendall(RESPONSE_HEADER.pack(windows.shape[0], elapsed_ms))
                if windows.size:
                    self.request.sendall(windows.tobytes(order="C"))
            except ValueError as error:
                # A HTTP request accidentally sent to the binary feature port,
                # or a stale client using an incompatible framing, must only
                # terminate this connection.  Keep the service alive and log
                # enough bytes to identify the bad protocol at the source.
                print(
                    f"FEATURE protocol error connection={connection_id} frame={frame_seq} "
                    f"header={header.hex()} error={error}", flush=True,
                )
                return

    def _read_exact(self, size: int) -> bytes | None:
        chunks = []
        remaining = size
        while remaining:
            chunk = self.request.recv(remaining)
            if not chunk:
                return None if chunks else b""
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)


class FeatureServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> None:
    parser = argparse.ArgumentParser(description="K3 FeatherHuBERT ONNX feature server")
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8892)
    # The production K3 layout reserves four general-purpose cores for
    # FeatherHuBERT (the renderer's EP workers use the other eight).  Starting
    # eight intra-op workers on those four cores oversubscribes the CPU and
    # creates the 200--500 ms inference spikes that starve live video.
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    options = ort.SessionOptions()
    options.intra_op_num_threads = max(1, args.threads)
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    started = time.perf_counter()
    ort_session = ort.InferenceSession(
        args.model,
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    server = FeatureServer((args.host, args.port), FeatureRequestHandler)
    server.ort_session = ort_session
    print(
        f"READY host={args.host} port={args.port} model={args.model} "
        f"load_ms={(time.perf_counter() - started) * 1000.0:.1f}",
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
