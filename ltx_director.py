import logging
import asyncio
import json
import base64
import io as _io
import math

import numpy as np
import torch
import torch.nn.functional as F
import av
from PIL import Image

import os
import platform
import folder_paths
import comfy.model_management
from server import PromptServer
from aiohttp import web

from comfy_api.latest import io

from . import kling_client
from . import video_utils

log = logging.getLogger(__name__)

# Setup global event loop exception handler to silence ConnectionResetError (WinError 10054/10053) on Windows
try:
    loop = None
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        try:
            loop = asyncio.get_event_loop_policy().get_event_loop()
        except Exception:
            pass

    if loop is not None:
        old_handler = loop.get_exception_handler()
        
        def silence_connection_reset_handler(loop, context):
            exception = context.get('exception')
            if (isinstance(exception, (ConnectionResetError, ConnectionAbortedError)) or 
                (isinstance(exception, OSError) and getattr(exception, 'winerror', None) in (10054, 10053))):
                # Suppress WinError 10054 and WinError 10053 tracebacks in logging
                return
            if old_handler:
                old_handler(loop, context)
            else:
                loop.default_exception_handler(context)
                
        loop.set_exception_handler(silence_connection_reset_handler)
except Exception:
    pass

# Custom socket carrying the generated Kling result to the output node.
KlingVideo = io.Custom("KLING_VIDEO")

# --- File Check Endpoint for Deduplication ---
@PromptServer.instance.routes.get("/ltx_director_check_file")
async def ltx_director_check_file(request):
    filename = request.query.get("filename", "")
    file_size = request.query.get("size", "")
    if not filename:
        return web.json_response({"exists": False})

    upload_dir = folder_paths.get_input_directory()
    temp_dir = os.path.join(upload_dir, "whatdreamscost")
    
    # 1. Check if the exact filename exists in whatdreamscost or root input dir
    possible_paths = [
        os.path.join(temp_dir, filename),
        os.path.join(upload_dir, filename)
    ]
    
    found_path = None
    for p in possible_paths:
        if os.path.exists(p) and os.path.isfile(p):
            if file_size:
                try:
                    if os.path.getsize(p) == int(file_size):
                        found_path = p
                        break
                except ValueError:
                    found_path = p
                    break
            else:
                found_path = p
                break
                
    if found_path:
        rel_name = os.path.relpath(found_path, upload_dir).replace('\\', '/')
        return web.json_response({"exists": True, "name": rel_name})

    # 2. Suffix search if exact match not found
    base_name = os.path.basename(filename)
    suffix = f"_{base_name}"
    try:
        for search_dir in [temp_dir, upload_dir]:
            if os.path.exists(search_dir):
                for f_name in os.listdir(search_dir):
                    if f_name.endswith(suffix) or f_name == base_name:
                        pot_path = os.path.join(search_dir, f_name)
                        if os.path.isfile(pot_path):
                            if file_size:
                                try:
                                    if os.path.getsize(pot_path) == int(file_size):
                                        rel_name = os.path.relpath(pot_path, upload_dir).replace('\\', '/')
                                        return web.json_response({"exists": True, "name": rel_name})
                                except ValueError:
                                    pass
                            else:
                                rel_name = os.path.relpath(pot_path, upload_dir).replace('\\', '/')
                                return web.json_response({"exists": True, "name": rel_name})
    except Exception as e:
        log.warning(f"[LTXDirector] Error listing input directory: {e}")

    return web.json_response({"exists": False})


def read_wav_peaks(wav_path):
    import wave
    peaks = []
    with wave.open(wav_path, 'rb') as w:
        n_frames = w.getnframes()
        if n_frames > 0:
            frames_bytes = w.readframes(n_frames)
            samples = np.frombuffer(frames_bytes, dtype=np.int16)
            num_peaks = 200
            step = max(1, len(samples) // num_peaks)
            for i in range(num_peaks):
                chunk = samples[i * step : (i + 1) * step]
                if len(chunk) > 0:
                    max_val = np.max(np.abs(chunk)) / 32767.0
                    peaks.append(float(max_val))
                else:
                    peaks.append(0.0)
        else:
            peaks = [0.0] * 200
    return peaks


def extract_audio_from_video(video_path):
    import wave
    try:
        base, _ = os.path.splitext(video_path)
        output_wav = base + "_extracted_audio.wav"
        
        # Check if already exists, is not empty, and has the correct 44100Hz sample rate
        if os.path.exists(output_wav) and os.path.getsize(output_wav) > 44:
            try:
                with wave.open(output_wav, 'rb') as w_check:
                    if w_check.getframerate() == 44100:
                        peaks = read_wav_peaks(output_wav)
                        input_dir = folder_paths.get_input_directory()
                        rel_output = os.path.relpath(output_wav, input_dir).replace('\\', '/')
                        return rel_output, peaks
            except Exception:
                pass

        # Decode the video using PyAV
        with av.open(video_path) as container:
            if not container.streams.audio:
                return None, None
            stream = container.streams.audio[0]
            
            # Setup resampler to 44100Hz, Mono, signed 16-bit integer (s16)
            resampler = av.AudioResampler(
                format='s16',
                layout='mono',
                rate=44100,
            )
            
            audio_bytes = bytearray()
            
            for frame in container.decode(stream):
                for resampled_frame in resampler.resample(frame):
                    arr = resampled_frame.to_ndarray()
                    audio_bytes.extend(arr.tobytes())
                    
            # Flush resampler
            for resampled_frame in resampler.resample(None):
                arr = resampled_frame.to_ndarray()
                audio_bytes.extend(arr.tobytes())
                
            if not audio_bytes:
                return None, None
                
            # Write WAV file
            with wave.open(output_wav, 'wb') as w:
                w.setnchannels(1)
                w.setsampwidth(2) # 16-bit
                w.setframerate(44100)
                w.writeframes(audio_bytes)
                
        # Calculate peaks
        peaks = []
        samples = np.frombuffer(audio_bytes, dtype=np.int16)
        num_peaks = 200
        step = max(1, len(samples) // num_peaks)
        for i in range(num_peaks):
            chunk = samples[i * step : (i + 1) * step]
            if len(chunk) > 0:
                max_val = np.max(np.abs(chunk)) / 32767.0
                peaks.append(float(max_val))
            else:
                peaks.append(0.0)
                
        input_dir = folder_paths.get_input_directory()
        rel_output = os.path.relpath(output_wav, input_dir).replace('\\', '/')
        return rel_output, peaks
    except Exception as e:
        print(f"[LTXDirector] Server audio extraction failed: {e}")
        return None, None


def get_audio_peaks(audio_path):
    import wave
    # If it is already a WAV file, read peaks directly
    _, ext = os.path.splitext(audio_path)
    if ext.lower() == ".wav":
        try:
            return read_wav_peaks(audio_path)
        except Exception:
            pass # fallback to PyAV
            
    # Use PyAV to decode and resample the audio file
    try:
        with av.open(audio_path) as container:
            if not container.streams.audio:
                return None
            stream = container.streams.audio[0]
            resampler = av.AudioResampler(
                format='s16',
                layout='mono',
                rate=8000,
            )
            audio_bytes = bytearray()
            for frame in container.decode(stream):
                for resampled_frame in resampler.resample(frame):
                    arr = resampled_frame.to_ndarray()
                    audio_bytes.extend(arr.tobytes())
            for resampled_frame in resampler.resample(None):
                arr = resampled_frame.to_ndarray()
                audio_bytes.extend(arr.tobytes())
                
            if not audio_bytes:
                return None
                
            peaks = []
            samples = np.frombuffer(audio_bytes, dtype=np.int16)
            num_peaks = 200
            step = max(1, len(samples) // num_peaks)
            for i in range(num_peaks):
                chunk = samples[i * step : (i + 1) * step]
                if len(chunk) > 0:
                    max_val = np.max(np.abs(chunk)) / 32767.0
                    peaks.append(float(max_val))
                else:
                    peaks.append(0.0)
            return peaks
    except Exception as e:
        print(f"[LTXDirector] Failed to get audio peaks via PyAV: {e}")
        return None


@PromptServer.instance.routes.get("/ltx_director_get_audio")
async def ltx_director_get_audio(request):
    filename = request.query.get("filename")
    if not filename:
        return web.json_response({"error": "Missing filename"}, status=400)

    upload_dir = folder_paths.get_input_directory()
    
    clean_filename = filename.replace('\\', '/')
    file_path = os.path.join(upload_dir, clean_filename)
    if not os.path.exists(file_path):
        basename = os.path.basename(clean_filename)
        temp_path = os.path.join(upload_dir, "whatdreamscost", basename)
        if os.path.exists(temp_path):
            file_path = temp_path
        else:
            file_path = os.path.join(upload_dir, basename)
        
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        return web.json_response({"error": "File not found"}, status=404)

    _, ext = os.path.splitext(file_path)
    is_audio = ext.lower() in [".wav", ".mp3", ".ogg", ".flac", ".m4a"]
    
    if is_audio:
        peaks = None
        try:
            peaks = get_audio_peaks(file_path)
        except Exception as e:
            print(f"[LTXDirector] Failed to get audio peaks for audio file: {e}")
            
        rel_path = os.path.relpath(file_path, upload_dir).replace('\\', '/')
        return web.json_response({
            "audio_file": rel_path,
            "peaks": peaks
        })

    audio_file, peaks = None, None
    try:
        loop = asyncio.get_event_loop()
        audio_file, peaks = await loop.run_in_executor(None, extract_audio_from_video, file_path)
    except Exception as e:
        print(f"[LTXDirector] Error extracting audio: {e}")

    return web.json_response({
        "audio_file": audio_file,
        "peaks": peaks
    })


@PromptServer.instance.routes.get("/ltx_director_open_folder")
async def ltx_director_open_folder(request):
    upload_dir = os.path.join(folder_paths.get_input_directory(), "whatdreamscost")
    os.makedirs(upload_dir, exist_ok=True)
    try:
        if hasattr(os, "startfile"):
            os.startfile(upload_dir)
        else:
            import webbrowser
            webbrowser.open(os.path.abspath(upload_dir))
        return web.json_response({"success": True})
    except Exception as e:
        print(f"[LTXDirector] Failed to open workspace folder: {e}")
        return web.json_response({"success": False, "error": str(e)}, status=500)


def _read_and_write_file_chunk(file, file_path, mode):
    chunk_bytes = file.file.read()
    with open(file_path, mode) as f:
        f.write(chunk_bytes)


# --- LTX Director Chunked Video Upload Endpoint ---
# Bypasses the 413 Payload Too Large error for large video files.
# This endpoint is self-contained and independent of any other node.
@PromptServer.instance.routes.post("/ltx_director_upload_chunk")
async def ltx_director_upload_chunk(request):
    post = await request.post()
    file = post.get("file")
    filename = post.get("filename")
    chunk_index = int(post.get("chunk_index"))
    total_chunks = int(post.get("total_chunks"))

    upload_dir = os.path.join(folder_paths.get_input_directory(), "whatdreamscost")
    os.makedirs(upload_dir, exist_ok=True)

    # Sanitize filename to prevent path traversal attacks (e.g. ../../etc/passwd)
    filename = os.path.basename(filename)
    file_path = os.path.join(upload_dir, filename)

    # Belt-and-suspenders: confirm the resolved path is still inside the upload directory
    if not os.path.realpath(file_path).startswith(os.path.realpath(upload_dir)):
        return web.json_response({"error": "Invalid filename"}, status=400)

    # Append chunk to file (write fresh on first chunk, append on subsequent)
    mode = "ab" if chunk_index > 0 else "wb"
    
    # Offload the blocking read/write disk I/O to a thread executor
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _read_and_write_file_chunk, file, file_path, mode)

    if chunk_index == total_chunks - 1:
        audio_file, peaks = None, None
        try:
            audio_file, peaks = await loop.run_in_executor(None, extract_audio_from_video, file_path)
        except Exception as e:
            print(f"[LTXDirector] Error in final chunk audio extraction: {e}")
            
        return web.json_response({
            "name": f"whatdreamscost/{filename}",
            "audio_file": audio_file,
            "peaks": peaks
        })
    return web.json_response({"status": "ok"})



def _load_image_tensor(seg: dict) -> torch.Tensor:
    """Decode an image from the ComfyUI input folder (if imageFile provided) or fallback to base64
    to a ComfyUI-style image tensor of shape [1, H, W, 3], float32 in [0, 1]."""
    if seg.get("imageFile"):
        file_path = os.path.join(folder_paths.get_input_directory(), seg["imageFile"])
        if os.path.exists(file_path):
            img = Image.open(file_path).convert("RGB")
            arr = np.array(img, dtype=np.float32) / 255.0
            return torch.from_numpy(arr).unsqueeze(0)

    b64_str = seg.get("imageB64", "")
    if not b64_str or b64_str.startswith("/view?"):
        return torch.zeros((1, 512, 512, 3), dtype=torch.float32)

    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    
    try:
        img_bytes = base64.b64decode(b64_str)
        img = Image.open(_io.BytesIO(img_bytes)).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)
    except:
        return torch.zeros((1, 512, 512, 3), dtype=torch.float32)

def _load_video_tensor(seg: dict, frame_rate: float) -> torch.Tensor:
    """Extracts a sequence of frames from a video file based on the segment's trim parameters,
    and returns them as an [N, H, W, 3] float32 tensor."""
    file_path = os.path.join(folder_paths.get_input_directory(), seg.get("imageFile", ""))
    
    if not os.path.exists(file_path):
        return torch.zeros((1, 512, 512, 3), dtype=torch.float32)

    trim_start_frames = float(seg.get("trimStart", 0))
    length_frames = float(seg.get("length", 1))
    start_sec = trim_start_frames / frame_rate
    
    frames = []
    try:
        with av.open(file_path) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            
            # Seek slightly before target to hit a keyframe
            if stream.time_base:
                seek_pts = int((max(0, start_sec - 0.5)) / float(stream.time_base))
            else:
                seek_pts = int((max(0, start_sec - 0.5)) * av.time_base)
            
            container.seek(seek_pts, stream=stream, backward=True)
            
            for frame in container.decode(stream):
                frame_time = frame.time
                if frame_time is None and frame.pts is not None and stream.time_base:
                    frame_time = float(frame.pts * stream.time_base)
                    
                if frame_time is None:
                    frame_time = 0.0
                    
                if frame_time < start_sec - 0.01:
                    continue
                    
                frames.append(frame.to_ndarray(format='rgb24'))
                
                if len(frames) >= int(length_frames):
                    break
    except Exception as e:
        log.warning(f"[PromptRelay] Video extract error: {e}")
        
    if not frames:
        return torch.zeros((1, 512, 512, 3), dtype=torch.float32)
        
    frames_np = np.array(frames, dtype=np.float32) / 255.0
    return torch.from_numpy(frames_np)

def _resize_image(tensor: torch.Tensor, target_w: int, target_h: int, method: str, divisible_by: int) -> torch.Tensor:
    """Resize an [N, H, W, 3] float32 tensor to target dimensions using the given method,
    then snap the final dimensions to be divisible by `divisible_by`."""
    
    def snap(val, div):
        return max(div, (val // div) * div)

    tw = snap(target_w, divisible_by)
    th = snap(target_h, divisible_by)

    N, H, W, C = tensor.shape
    if H == th and W == tw:
        return tensor

    t_nchw = tensor.permute(0, 3, 1, 2)
    
    if method == "stretch to fit":
        resized = F.interpolate(t_nchw, size=(th, tw), mode="bilinear", align_corners=False)
        
    elif method == "maintain aspect ratio":
        ratio = min(tw / W, th / H)
        new_w = snap(int(W * ratio), divisible_by)
        new_h = snap(int(H * ratio), divisible_by)
        resized = F.interpolate(t_nchw, size=(new_h, new_w), mode="bilinear", align_corners=False)
        
    elif method == "pad" or method == "pad green":
        ratio = min(tw / W, th / H)
        new_w = snap(int(W * ratio), divisible_by)
        new_h = snap(int(H * ratio), divisible_by)
        inner = F.interpolate(t_nchw, size=(new_h, new_w), mode="bilinear", align_corners=False)
        
        pad_l = (tw - new_w) // 2
        pad_t = (th - new_h) // 2
        
        if method == "pad green":
            resized = torch.zeros((N, C, th, tw), dtype=t_nchw.dtype, device=t_nchw.device)
            # #66FF00 is roughly R: 102/255, G: 255/255, B: 0
            resized[:, 0, :, :] = 102 / 255.0
            resized[:, 1, :, :] = 1.0
            resized[:, 2, :, :] = 0.0
            resized[:, :, pad_t:pad_t+new_h, pad_l:pad_l+new_w] = inner
        else:
            resized = F.pad(inner, (pad_l, tw - new_w - pad_l, pad_t, th - new_h - pad_t), mode="constant", value=0)
        
    elif method == "crop":
        ratio = max(tw / W, th / H)
        new_w = int(W * ratio)
        new_h = int(H * ratio)
        inner = F.interpolate(t_nchw, size=(new_h, new_w), mode="bilinear", align_corners=False)
        
        left = (new_w - tw) // 2
        top = (new_h - th) // 2
        resized = inner[:, :, top:top+th, left:left+tw]
        
    else:
        resized = F.interpolate(t_nchw, size=(th, tw), mode="bilinear", align_corners=False)

    return resized.permute(0, 2, 3, 1)


def _compress_image(tensor: torch.Tensor, crf: int) -> torch.Tensor:
    """Apply H.264 compression artefacts to an [N, H, W, 3] float32 tensor (ComfyUI image format).
    crf=0 means no compression. Uses PyAV to encode/decode frames in-memory."""
    if crf == 0:
        return tensor
        
    N, H, W, C = tensor.shape
    
    # Dimensions must be even for H.264
    h = (H // 2) * 2
    w = (W // 2) * 2
    
    # uint8 [N, H, W, 3]
    tensor_bytes = (tensor[:, :h, :w, :] * 255.0).byte().cpu().numpy()
    
    try:
        buf = _io.BytesIO()
        container = av.open(buf, mode="w", format="mp4")
        stream = container.add_stream("libx264", rate=24)
        stream.width = w
        stream.height = h
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(crf), "preset": "ultrafast"}
        
        for i in range(N):
            frame = av.VideoFrame.from_ndarray(tensor_bytes[i], format="rgb24")
            for pkt in stream.encode(frame):
                container.mux(pkt)
                
        for pkt in stream.encode(None):
            container.mux(pkt)
            
        container.close()
        
        buf.seek(0)
        container_r = av.open(buf, mode="r")
        decoded = [frame_r.to_ndarray(format="rgb24") for frame_r in container_r.decode(video=0)]
        container_r.close()
        
        if not decoded:
            return tensor
            
        decoded_np = np.stack(decoded).astype(np.float32) / 255.0
        
        # Re-embed into original tensor shape (may have been cropped by even-rounding)
        out = tensor.clone()
        dec_N = min(N, len(decoded))
        out[:dec_N, :h, :w] = torch.from_numpy(decoded_np[:dec_N]).to(tensor.device, tensor.dtype)
        
        return out
        
    except Exception as e:
        log.warning("[PromptRelay] img_compression encode/decode failed: %s", e)
        return tensor


def _build_combined_audio(timeline_data_str: str, start_frame: int, duration_frames: int, frame_rate: float, override_audio: bool = False) -> dict:
    """Parses timeline JSON, loads/trims audio directly from memory using PyAV, 
    and aligns to a global timeline yielding ComfyUI's format.
    Output length explicitly mimics the timeline's duration_frames length."""
    target_sr = 44100
    total_samples = max(1, int(math.ceil(duration_frames / frame_rate * target_sr)))
    empty_audio = {"waveform": torch.zeros((1, 2, total_samples), dtype=torch.float32), "sample_rate": target_sr}

    if not timeline_data_str:
        return empty_audio

    try:
        data = json.loads(timeline_data_str)
        is_retake = data.get("retakeMode", False)
        if is_retake and data.get("retakeVideo"):
            retake_vid = data.get("retakeVideo")
            audio_segs = [{
                "videoFile": retake_vid.get("imageFile") or retake_vid.get("fileName"),
                "audioFile": retake_vid.get("imageFile") or retake_vid.get("fileName"),
                "start": 0,
                "length": retake_vid.get("videoDurationFrames", duration_frames),
                "trimStart": 0
            }]
            override_audio = True
        elif override_audio:
            audio_segs = data.get("motionSegments", [])
        else:
            audio_segs = data.get("audioSegments", [])
    except Exception:
        return empty_audio

    if not audio_segs:
        return empty_audio

    out_waveform = torch.zeros((2, total_samples), dtype=torch.float32)

    for seg in audio_segs:
        buffer = None
        file_key = "videoFile" if override_audio else "audioFile"
        if seg.get(file_key):
            file_path = os.path.join(folder_paths.get_input_directory(), seg[file_key])
            if not os.path.exists(file_path):
                # Try fallback under whatdreamscost subfolder
                basename = os.path.basename(seg[file_key])
                fallback_path = os.path.join(folder_paths.get_input_directory(), "whatdreamscost", basename)
                if os.path.exists(fallback_path):
                    file_path = fallback_path

            if os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    buffer = _io.BytesIO(f.read())
        
        if not override_audio and not buffer and seg.get("audioB64"):
            b64 = seg.get("audioB64")
            if "," in b64:
                b64 = b64.split(",", 1)[1]
            try:
                audio_bytes = base64.b64decode(b64)
                buffer = _io.BytesIO(audio_bytes)
            except:
                pass
                
        if not buffer:
            continue

        try:
            clip_frames = []
            
            # Use PyAV to decode directly from memory buffer
            with av.open(buffer) as container:
                if not container.streams.audio:
                    continue
                stream = container.streams.audio[0]
                
                # Setup resampler to ensure output is 44.1kHz, Stereo, Float32 Planar
                resampler = av.AudioResampler(
                    format='fltp',
                    layout='stereo',
                    rate=target_sr,
                )
                
                for frame in container.decode(stream):
                    for resampled_frame in resampler.resample(frame):
                        # to_ndarray() on fltp gives shape (channels, samples)
                        arr = resampled_frame.to_ndarray()
                        clip_frames.append(torch.from_numpy(arr))
                
                # Flush the resampler to get any remaining samples
                for resampled_frame in resampler.resample(None):
                    arr = resampled_frame.to_ndarray()
                    clip_frames.append(torch.from_numpy(arr))

            if not clip_frames:
                continue

            # Concatenate all frame blocks along the samples dimension (dim 1)
            waveform = torch.cat(clip_frames, dim=1) # Shape: [2, total_clip_samples]

            # Calculate interactive trim boundaries
            trim_start_frames = float(seg.get("trimStart", 0))
            length_frames = float(seg.get("length", 1))
            start_frames = float(seg.get("start", 0))
            
            if start_frames + length_frames <= start_frame:
                continue
                
            offset = max(0, start_frame - start_frames)
            trim_start_frames += offset
            length_frames = max(1, length_frames - offset)
            start_frames = max(0, start_frames - start_frame)

            start_sample_src = int(trim_start_frames / frame_rate * target_sr)
            length_samples = int(length_frames / frame_rate * target_sr)
            end_sample_src = start_sample_src + length_samples

            if start_sample_src < 0: start_sample_src = 0
            if end_sample_src > waveform.shape[1]:
                end_sample_src = waveform.shape[1]

            actual_length = end_sample_src - start_sample_src
            if actual_length <= 0: continue

            # Extract the correct segment of the audio
            clip_waveform = waveform[:, start_sample_src:end_sample_src]

            # Position onto the timeline
            start_sample_dst = int(start_frames / frame_rate * target_sr)
            
            if start_sample_dst >= out_waveform.shape[1]:
                continue
                
            end_sample_dst = start_sample_dst + actual_length

            # Clip any trailing overflow so we don't index past the timeline bounds
            if end_sample_dst > out_waveform.shape[1]:
                actual_length = out_waveform.shape[1] - start_sample_dst
                clip_waveform = clip_waveform[:, :actual_length]
                end_sample_dst = start_sample_dst + actual_length
                
            if actual_length <= 0:
                continue

            # Additive composite (allows clips overlapping to sum together naturally)
            out_waveform[:, start_sample_dst:end_sample_dst] += clip_waveform

        except Exception as e:
            log.warning("[PromptRelay] Audio process error for segment %s: %s", seg.get("fileName"), e)
            continue

    return {"waveform": out_waveform.unsqueeze(0), "sample_rate": target_sr}


def _snap_kling_duration(seconds: float) -> int:
    """Kling clips are 5 or 10 seconds; snap a requested length to the nearest."""
    try:
        return 10 if float(seconds) > 7.5 else 5
    except Exception:
        return 5


def _resolve_input_path(rel_or_name: str):
    """Resolve an imageFile/videoFile (input-relative) to an absolute path, checking the
    ComfyUI input dir and the whatdreamscost/ subfolder (where uploads land)."""
    if not rel_or_name:
        return None
    base = folder_paths.get_input_directory()
    candidates = [
        os.path.join(base, rel_or_name),
        os.path.join(base, "whatdreamscost", os.path.basename(rel_or_name)),
    ]
    for p in candidates:
        if os.path.exists(p) and os.path.isfile(p):
            return p
    return None


def _segment_start_image_b64(seg: dict):
    """Get a base64 (no data-URI prefix) start image for a timeline segment, or None.
    Images send their file bytes directly; video segments send their first frame."""
    seg_type = seg.get("type", "image")
    path = _resolve_input_path(seg.get("imageFile") or seg.get("videoFile") or "")
    if seg_type == "video" and path:
        return video_utils.first_frame_b64(path)
    if path:
        try:
            return video_utils.file_to_b64(path)
        except Exception:
            pass
    b64 = seg.get("imageB64", "")
    if b64 and not b64.startswith("/view?"):
        return b64.split(",", 1)[1] if "," in b64 else b64
    return None


class LTXDirector(io.ComfyNode):
    """WYSIWYG timeline variant — segments and lengths come from a visual editor in the node UI."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXDirector",
            display_name="LTX Director",
            category="WhatDreamsCost",
            description=(
                "Storyboard timeline that generates video with the Kling API (klingapi.com). "
                "Each timeline segment becomes a Kling shot (image2video if it has an image, else "
                "text2video); shots are downloaded and stitched, then sent to Kling Video Output. "
                "Enter a single Kling API key below."
            ),
            inputs=[
                io.String.Input(
                    "global_prompt", multiline=True, default="", force_input=True, optional=True,
                    tooltip="Conditions the entire video. Anchors persistent characters, objects, and scene context.",
                ),
                io.Float.Input(
                    "start_second", default=0.0, min=0.0, max=1000.0, step=0.01,
                    tooltip="Start time in seconds of the timeline generation.",
                ),
                io.Float.Input(
                    "end_second", default=5.0, min=0.0, max=1000.0, step=0.01,
                    tooltip="End time in seconds of the timeline generation.",
                ),
                io.Float.Input(
                    "duration_seconds", default=5.0, min=0.1, max=1000.0, step=0.01,
                    tooltip="Total timeline duration in seconds (computed/synced from frames).",
                ),
                io.Int.Input(
                    "start_frame", default=0, min=0, max=10000, step=1,
                    tooltip="Start frame of the timeline generation.",
                ),
                io.Int.Input(
                    "end_frame", default=120, min=1, max=10000, step=1,
                    tooltip="End frame of the timeline generation.",
                ),
                io.Int.Input(
                    "duration_frames", default=120, min=1, max=10000, step=1,
                    tooltip="Total timeline length in pixel-space frames. Used by the editor for visual scale only.",
                ),
                io.String.Input(
                    "timeline_data", default="",
                    tooltip="JSON state of the timeline editor (auto-managed; do not edit by hand).",
                ),
                io.Boolean.Input(
                    "use_custom_audio", default=False, optional=True,
                    tooltip="Toggle between using timeline audio (ON) and generating audio from scratch (OFF).",
                ),
                io.Boolean.Input(
                    "use_custom_motion", default=True, optional=True,
                    tooltip="Toggle between using timeline motion guidance (ON) and ignoring motion video segments (OFF).",
                ),
                io.Boolean.Input(
                    "inpaint_audio", default=True, optional=True,
                    tooltip="Toggle whether empty gaps in the audio track are inpainted with generated audio.",
                ),
                io.String.Input(
                    "local_prompts", multiline=True, default="",
                    tooltip="Auto-populated from the timeline editor.",
                ),
                io.String.Input(
                    "segment_lengths", default="",
                    tooltip="Auto-populated from the timeline editor (pixel-space frame counts).",
                ),
                io.Float.Input(
                    "epsilon", default=0.001, min=0.0001, max=0.99, step=0.0001,
                    tooltip="Penalty decay parameter. Values below ~0.1 all produce sharp boundaries (paper default 0.001). For softer transitions, try 0.5 or higher.",
                ),
                io.Float.Input(
                    "frame_rate", default=24, min=1, max=240, step=1, optional=True,
                    tooltip="Frames per second — only affects how time is displayed in the timeline editor when time_units is set to 'seconds'.",
                ),
                io.Combo.Input(
                    "display_mode", options=["frames", "seconds"], default="seconds", optional=True,
                    tooltip="Display the ruler, segment ranges, length input, and total in frames or seconds. Internal storage is always pixel-space frames.",
                ),
                io.String.Input(
                    "guide_strength", default="",
                    tooltip="Auto-populated from the timeline editor (comma-separated guide strengths for image segments).",
                ),
                io.Int.Input(
                    "custom_width", default=0, min=0, max=8192, step=1, optional=True,
                    tooltip="Target output width for all image segments. Set to 0 to use the original image width.",
                ),
                io.Int.Input(
                    "custom_height", default=0, min=0, max=8192, step=1, optional=True,
                    tooltip="Target output height for all image segments. Set to 0 to use the original image height.",
                ),
                io.Combo.Input(
                    "resize_method",
                    options=["maintain aspect ratio", "stretch to fit", "pad", "pad green", "crop"],
                    default="maintain aspect ratio",
                    optional=True,
                    tooltip="How to resize image segments to fit the target dimensions.",
                ),
                io.Int.Input(
                    "divisible_by", default=32, min=1, max=256, step=1, optional=True,
                    tooltip="Snap the final output image dimensions to be divisible by this number (e.g. 32 for LTX).",
                ),
                io.Int.Input(
                    "img_compression", default=18, min=0, max=100, step=1, optional=True,
                    tooltip="H.264 CRF compression to apply to each guide image. 0 = no compression, higher = more artefacts.",
                ),
                io.Boolean.Input(
                    "override_audio", default=False, optional=True,
                    tooltip="Use the audio from an imported video segment instead of the audio track.",
                ),
                # --- Kling engine (appended last to keep the JS positional widget order intact) ---
                io.String.Input(
                    "kling_api_key", default="", optional=True,
                    tooltip="Your klingapi.com key (Bearer). Saved in the workflow — blank it before sharing. "
                            "Env var KLING_API_KEY is used if this is left empty.",
                ),
                io.Combo.Input(
                    "model_name", options=kling_client.MODEL_NAMES, default=kling_client.MODEL_NAMES[0],
                    optional=True, tooltip="Kling model to generate with.",
                ),
                io.Combo.Input(
                    "mode", options=kling_client.MODES, default="standard", optional=True,
                    tooltip="Kling quality mode.",
                ),
                io.Combo.Input(
                    "aspect_ratio", options=["16:9", "9:16", "1:1"], default="16:9", optional=True,
                    tooltip="Output aspect ratio (text-to-video; image-to-video follows the image).",
                ),
                io.String.Input(
                    "negative_prompt", multiline=True,
                    default="blurry, distorted, deformed, extra limbs, warping, low quality, jpeg artifacts",
                    optional=True, tooltip="Things to avoid in every shot.",
                ),
                io.Float.Input(
                    "cfg_scale", default=0.5, min=0.0, max=1.0, step=0.05, optional=True,
                    tooltip="Prompt relevance (higher = follow the prompt more closely).",
                ),
                io.String.Input(
                    "base_url", default=kling_client.DEFAULT_BASE_URL, optional=True,
                    tooltip="Gateway base URL. Default is klingapi.com.",
                ),
            ],
            outputs=[
                KlingVideo.Output(display_name="kling_video"),
            ],
        )

    @classmethod
    def execute(cls, global_prompt="", start_second=0.0, end_second=5.0, duration_seconds=5.0,
                start_frame=0, end_frame=120, duration_frames=120, timeline_data="",
                use_custom_audio=False, use_custom_motion=True, inpaint_audio=True,
                local_prompts="", segment_lengths="", epsilon=1e-3, frame_rate=24,
                display_mode="seconds", guide_strength="", custom_width=0, custom_height=0,
                resize_method="maintain aspect ratio", divisible_by=32, img_compression=0,
                override_audio=False, kling_api_key="", model_name=None, mode="standard",
                aspect_ratio="16:9", negative_prompt="", cfg_scale=0.5, base_url=None) -> io.NodeOutput:
        """Kling engine: turn the timeline into klingapi.com calls, then stitch the clips."""

        key = (kling_api_key or "").strip() or os.environ.get("KLING_API_KEY", "").strip()
        if not key:
            raise RuntimeError(
                "LTX Director (Kling): no API key. Enter it in the kling_api_key field "
                "or set the KLING_API_KEY environment variable."
            )
        base_url = (base_url or kling_client.DEFAULT_BASE_URL).strip()
        model_name = model_name or kling_client.MODEL_NAMES[0]
        fr = float(frame_rate) or 24.0

        try:
            tdata = json.loads(timeline_data) if timeline_data else {}
        except Exception as e:
            log.error("[LTXDirector/Kling] timeline_data parse error: %s", e)
            tdata = {}

        if not global_prompt:
            global_prompt = tdata.get("global_prompt", "") or ""

        # --- Build an ordered shot list from the timeline main track ---
        win_start = int(start_frame)
        win_end = int(start_frame) + int(duration_frames)
        segs = [
            s for s in tdata.get("segments", [])
            if s.get("type", "image") in ("image", "video")
            and int(s.get("start", 0)) < win_end
            and int(s.get("start", 0)) + int(s.get("length", 1)) > win_start
        ]
        segs.sort(key=lambda s: int(s.get("start", 0)))

        shots = []  # each: {"prompt", "image" (b64 or None), "duration"}
        if segs:
            for s in segs:
                p = (s.get("prompt") or global_prompt or "").strip() or "video"
                dur = _snap_kling_duration(int(s.get("length", 1)) / fr)
                shots.append({"prompt": p, "image": _segment_start_image_b64(s), "duration": dur})
        else:
            prompts = [p.strip() for p in (local_prompts or "").split("|") if p.strip()]
            lengths = [x.strip() for x in (segment_lengths or "").split(",") if x.strip()]
            if not prompts and global_prompt.strip():
                prompts = [global_prompt.strip()]
            if not prompts:
                raise RuntimeError(
                    "LTX Director (Kling): nothing to generate — add a prompt or an "
                    "image/video segment to the timeline."
                )
            for i, p in enumerate(prompts):
                try:
                    dur = _snap_kling_duration(float(lengths[i]) / fr) if i < len(lengths) else 5
                except Exception:
                    dur = 5
                gp = global_prompt.strip()
                full = f"{gp}, {p}" if (gp and gp not in p) else p
                shots.append({"prompt": full, "image": None, "duration": dur})

        log.info("[LTXDirector/Kling] Generating %d shot(s) via %s @ %s", len(shots), model_name, base_url)

        # --- Generate each shot (submit -> poll -> download) ---
        run_root = os.path.join(folder_paths.get_output_directory(), "kling_director")
        os.makedirs(run_root, exist_ok=True)
        try:
            run_id = "run_%d" % (len(os.listdir(run_root)) + 1)
        except Exception:
            run_id = "run"
        run_dir = os.path.join(run_root, run_id)
        os.makedirs(run_dir, exist_ok=True)

        clip_paths = []
        for i, shot in enumerate(shots):
            try:
                if shot["image"]:
                    task_id = kling_client.submit_image2video(
                        base_url, key, model_name, shot["prompt"], shot["image"],
                        duration=shot["duration"], mode=mode,
                        negative_prompt=negative_prompt, cfg_scale=cfg_scale,
                    )
                else:
                    task_id = kling_client.submit_text2video(
                        base_url, key, model_name, shot["prompt"],
                        duration=shot["duration"], aspect_ratio=aspect_ratio, mode=mode,
                        negative_prompt=negative_prompt, cfg_scale=cfg_scale,
                    )
                log.info("[LTXDirector/Kling] shot %d/%d submitted (task %s); polling…",
                         i + 1, len(shots), task_id)

                def _cb(status, waited, _i=i):
                    log.info("[LTXDirector/Kling] shot %d/%d status=%s (%ss)",
                             _i + 1, len(shots), status, waited)

                video_url = kling_client.poll(base_url, key, task_id, on_status=_cb)
                dest = os.path.join(run_dir, "shot%02d.mp4" % i)
                kling_client.download(video_url, dest)
                clip_paths.append(dest)
                log.info("[LTXDirector/Kling] shot %d/%d downloaded -> %s", i + 1, len(shots), dest)
            except kling_client.KlingError as e:
                raise RuntimeError(f"LTX Director (Kling): shot {i + 1}/{len(shots)} failed — {e}") from e

        if not clip_paths:
            raise RuntimeError("LTX Director (Kling): no clips were generated.")

        # --- Custom audio track (only when the user provided one) ---
        audio = None
        if use_custom_audio and tdata.get("audioSegments"):
            try:
                audio = _build_combined_audio(
                    timeline_data, int(start_frame), int(duration_frames), fr, override_audio
                )
            except Exception as e:
                log.warning("[LTXDirector/Kling] combined audio build failed: %s", e)

        # --- Stitch ---
        if len(clip_paths) == 1 and audio is None:
            final_path = clip_paths[0]
        else:
            final_path = os.path.join(run_dir, "final.mp4")
            video_utils.assemble(clip_paths, final_path, audio=audio)

        return io.NodeOutput({"video_path": final_path, "frame_rate": fr, "audio": audio})
