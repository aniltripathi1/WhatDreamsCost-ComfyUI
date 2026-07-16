"""Video assembly helpers for the Kling engine — all via PyAV (already a repo dep).

- ``file_to_b64`` / ``first_frame_b64``: prepare start images for image2video.
- ``assemble``: re-encode + concatenate downloaded clips into one mp4, optionally
  muxing a custom audio track over the whole thing.
- ``decode_to_frames`` / ``decode_audio``: for the output node's optional IMAGE/AUDIO.

Audio muxing is best-effort: any failure degrades to a video-only result (a warning
is logged) rather than aborting the run — the video is the essential deliverable.
"""

import base64
import io as _io
import logging
import os

import av
import numpy as np
import torch
from PIL import Image

log = logging.getLogger(__name__)


def file_to_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def first_frame_b64(path):
    """Base64 JPEG (no data-URI prefix) of a video's first frame, or None."""
    try:
        with av.open(path) as c:
            for frame in c.decode(video=0):
                img = Image.fromarray(frame.to_ndarray(format="rgb24"))
                buf = _io.BytesIO()
                img.save(buf, format="JPEG", quality=92)
                return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception as e:
        log.warning("[KlingDirector] Could not read first frame of %s: %s", path, e)
    return None


def _probe(path):
    with av.open(path) as c:
        vs = c.streams.video[0]
        w = int(vs.codec_context.width)
        h = int(vs.codec_context.height)
        fps = float(vs.average_rate) if vs.average_rate else 24.0
    return w, h, fps


def assemble(clip_paths, out_path, audio=None):
    """Concatenate `clip_paths` (re-encoded to a uniform h264 stream) into `out_path`.

    `audio` (optional): {"waveform": tensor[1,2,S] or [2,S], "sample_rate": int} — muxed
    as a single AAC track spanning the whole video (used for the custom-audio case).
    Returns the output path.
    """
    clip_paths = [p for p in clip_paths if p and os.path.exists(p)]
    if not clip_paths:
        raise ValueError("No clips available to assemble.")

    w, h, fps = _probe(clip_paths[0])
    w -= w % 2
    h -= h % 2
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    out = av.open(out_path, mode="w")
    vstream = out.add_stream("libx264", rate=int(round(fps)) or 24)
    vstream.width = w
    vstream.height = h
    vstream.pix_fmt = "yuv420p"
    vstream.options = {"crf": "18", "preset": "medium"}

    idx = 0
    for p in clip_paths:
        with av.open(p) as c:
            for frame in c.decode(video=0):
                arr = frame.to_ndarray(format="rgb24")
                if arr.shape[1] != w or arr.shape[0] != h:
                    arr = np.asarray(Image.fromarray(arr).resize((w, h), Image.BILINEAR))
                vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(arr), format="rgb24")
                vf.pts = idx
                idx += 1
                for pkt in vstream.encode(vf):
                    out.mux(pkt)
    for pkt in vstream.encode(None):
        out.mux(pkt)

    if audio is not None:
        try:
            _mux_audio_stream(out, audio)
        except Exception as e:
            log.warning("[KlingDirector] Audio mux failed, writing video-only: %s", e)

    out.close()
    return out_path


def _mux_audio_stream(out_container, audio):
    wav = audio.get("waveform")
    sr = int(audio.get("sample_rate", 44100))
    if wav is None:
        return
    if isinstance(wav, torch.Tensor):
        wav = wav.detach().cpu().numpy()
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim == 3:
        wav = wav[0]
    if wav.ndim == 1:
        wav = np.stack([wav, wav], axis=0)
    if wav.shape[0] != 2:
        wav = wav[:2] if wav.shape[0] > 2 else np.repeat(wav[:1], 2, axis=0)

    astream = out_container.add_stream("aac", rate=sr)
    try:
        astream.layout = "stereo"
    except Exception:
        pass

    chunk = 1024
    total = wav.shape[1]
    pts = 0
    for start in range(0, total, chunk):
        block = np.ascontiguousarray(wav[:, start:start + chunk])
        frame = av.AudioFrame.from_ndarray(block, format="fltp", layout="stereo")
        frame.sample_rate = sr
        frame.pts = pts
        pts += block.shape[1]
        for pkt in astream.encode(frame):
            out_container.mux(pkt)
    for pkt in astream.encode(None):
        out_container.mux(pkt)


def decode_to_frames(path, max_frames=4000):
    """Decode a video to an [N,H,W,3] float32 IMAGE tensor + fps. Guarded by max_frames."""
    frames = []
    fps = 24.0
    with av.open(path) as c:
        vs = c.streams.video[0]
        fps = float(vs.average_rate) if vs.average_rate else 24.0
        for frame in c.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) >= max_frames:
                log.warning("[KlingDirector] decode_to_frames hit the %d-frame cap; truncating.", max_frames)
                break
    if not frames:
        return torch.zeros((1, 64, 64, 3), dtype=torch.float32), fps
    arr = np.stack(frames).astype(np.float32) / 255.0
    return torch.from_numpy(arr), fps


def decode_audio(path, target_sr=44100):
    """Best-effort decode of a video's audio to ComfyUI AUDIO dict, or silent stereo on failure."""
    try:
        with av.open(path) as c:
            if not c.streams.audio:
                raise ValueError("no audio stream")
            resampler = av.AudioResampler(format="fltp", layout="stereo", rate=target_sr)
            chunks = []
            astream = c.streams.audio[0]
            for frame in c.decode(astream):
                for rf in resampler.resample(frame):
                    chunks.append(rf.to_ndarray())
            if chunks:
                data = np.concatenate(chunks, axis=1)  # [2, S]
                wf = torch.from_numpy(np.ascontiguousarray(data)).unsqueeze(0).float()
                return {"waveform": wf, "sample_rate": target_sr}
    except Exception as e:
        log.debug("[KlingDirector] decode_audio fell back to silence: %s", e)
    return {"waveform": torch.zeros((1, 2, 1), dtype=torch.float32), "sample_rate": target_sr}
