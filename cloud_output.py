"""Kling Video Output — the single output node.

Takes the `KLING_VIDEO` produced by LTX Director (Kling engine), previews it inline,
and (optionally) decodes it to an IMAGE frame batch + AUDIO for downstream nodes.
The video file already lives under ComfyUI's output directory, so preview is served
via the standard /view endpoint.
"""

import os

import torch
import folder_paths

from . import video_utils


class KlingVideoOutput:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "kling_video": ("KLING_VIDEO",),
            },
            "optional": {
                "decode_to_frames": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Also output IMAGE frames + AUDIO. Uses a lot of RAM — short clips only.",
                }),
                "filename_prefix": ("STRING", {"default": "KlingDirector"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("frames", "audio", "video_path")
    FUNCTION = "output"
    CATEGORY = "WhatDreamsCost"
    OUTPUT_NODE = True

    def output(self, kling_video, decode_to_frames=False, filename_prefix="KlingDirector"):
        info = kling_video or {}
        path = info.get("video_path")
        if not path or not os.path.exists(path):
            raise RuntimeError(
                "Kling Video Output: no video received. The upstream LTX Director (Kling) "
                "generation likely failed — check the console for the error."
            )

        frame_rate = float(info.get("frame_rate", 24.0))

        # Reference the file (already under the output dir) for the inline /view preview.
        out_dir = folder_paths.get_output_directory()
        try:
            rel = os.path.relpath(path, out_dir)
            subfolder = os.path.dirname(rel).replace(os.sep, "/")
            filename = os.path.basename(rel)
            inside_output = not rel.startswith("..")
        except Exception:
            inside_output = False
            subfolder, filename = "", os.path.basename(path)

        ui = {}
        if inside_output:
            ui = {"kling_video": [{
                "filename": filename,
                "subfolder": subfolder,
                "type": "output",
                "format": "video/mp4",
                "frame_rate": frame_rate,
            }]}

        frames = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        audio = info.get("audio") or {"waveform": torch.zeros((1, 2, 1), dtype=torch.float32), "sample_rate": 44100}

        if decode_to_frames:
            frames, _ = video_utils.decode_to_frames(path)
            if info.get("audio") is None:
                audio = video_utils.decode_audio(path)

        return {"ui": ui, "result": (frames, audio, path)}
