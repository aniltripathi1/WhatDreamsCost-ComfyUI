"""Minimal client for the klingapi.com gateway (single Bearer key).

Endpoints (confirmed against klingapi.com docs):
  POST {base}/v1/videos/text2video
  POST {base}/v1/videos/image2video
  GET  {base}/v1/videos/{task_id}
Auth: header ``Authorization: Bearer <key>``.

Response shapes vary slightly between gateway versions, so task-id / status /
video-url extraction is deliberately defensive (searches common field names and
nesting). If klingapi.com changes field names, only this file needs edits.
"""

import base64
import json
import logging
import os
import time

log = logging.getLogger(__name__)

try:
    import requests  # ComfyUI ships this
    _HAS_REQUESTS = True
except Exception:  # pragma: no cover - stdlib fallback
    _HAS_REQUESTS = False
    import urllib.request
    import urllib.error

DEFAULT_BASE_URL = "https://api.klingapi.com"

# Model ids exposed by the gateway (docs). standard/professional are the mode values.
MODEL_NAMES = ["kling-v2.6-std", "kling-v2.6-pro", "kling-v2.5-turbo", "kling-video-o1"]
MODES = ["standard", "professional"]


class KlingError(Exception):
    """Any failure talking to the gateway, phrased for the user."""


def _headers(key):
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _post(url, key, payload):
    body_bytes = json.dumps(payload).encode("utf-8")
    if _HAS_REQUESTS:
        r = requests.post(url, headers=_headers(key), data=body_bytes, timeout=90)
        if r.status_code in (401, 403):
            raise KlingError(f"Authorization failed (HTTP {r.status_code}). Check your Kling API key.")
        try:
            data = r.json()
        except Exception:
            raise KlingError(f"Unexpected non-JSON response (HTTP {r.status_code}): {r.text[:300]}")
        if r.status_code >= 400:
            raise KlingError(f"Gateway error (HTTP {r.status_code}): {_msg(data)}")
        return data
    req = urllib.request.Request(url, data=body_bytes, headers=_headers(key), method="POST")
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:  # pragma: no cover
        detail = e.read().decode("utf-8", "replace")[:300]
        if e.code in (401, 403):
            raise KlingError(f"Authorization failed (HTTP {e.code}). Check your Kling API key.")
        raise KlingError(f"Gateway error (HTTP {e.code}): {detail}")


def _get(url, key):
    if _HAS_REQUESTS:
        r = requests.get(url, headers=_headers(key), timeout=60)
        try:
            return r.json()
        except Exception:
            raise KlingError(f"Unexpected non-JSON response (HTTP {r.status_code}): {r.text[:300]}")
    req = urllib.request.Request(url, headers=_headers(key), method="GET")
    with urllib.request.urlopen(req, timeout=60) as resp:  # pragma: no cover
        return json.loads(resp.read().decode("utf-8"))


def _msg(body):
    if isinstance(body, dict):
        return body.get("message") or body.get("msg") or json.dumps(body)[:300]
    return str(body)[:300]


def _data(body):
    if isinstance(body, dict) and isinstance(body.get("data"), dict):
        return body["data"]
    return body if isinstance(body, dict) else {}


def _extract_task_id(body):
    data = _data(body)
    for src in (data, body if isinstance(body, dict) else {}):
        for k in ("task_id", "taskId", "id"):
            if src.get(k):
                return str(src[k])
    raise KlingError(f"Could not find a task id in the gateway response: {json.dumps(body)[:300]}")


def _extract_status_and_url(body):
    data = _data(body)
    status = None
    for k in ("task_status", "status", "state"):
        v = data.get(k)
        if v:
            status = str(v).lower()
            break

    found = {"url": None}

    def _walk(o):
        if found["url"]:
            return
        if isinstance(o, dict):
            for k, v in o.items():
                if found["url"]:
                    return
                if k in ("url", "video_url", "videoUrl", "resource_url") and isinstance(v, str) and v.startswith("http"):
                    found["url"] = v
                    return
                _walk(v)
        elif isinstance(o, list):
            for v in o:
                _walk(v)

    _walk(data)
    return status, found["url"]


def submit_text2video(base_url, key, model, prompt, duration=5, aspect_ratio="16:9",
                      mode="standard", negative_prompt="", cfg_scale=0.5):
    url = base_url.rstrip("/") + "/v1/videos/text2video"
    payload = {"model": model, "prompt": prompt, "duration": int(duration),
               "aspect_ratio": aspect_ratio, "mode": mode}
    if negative_prompt:
        payload["negative_prompt"] = negative_prompt
    if cfg_scale is not None:
        payload["cfg_scale"] = float(cfg_scale)
    return _extract_task_id(_post(url, key, payload))


def submit_image2video(base_url, key, model, prompt, image_b64, image_tail_b64=None,
                       duration=5, mode="standard", negative_prompt="", cfg_scale=0.5):
    url = base_url.rstrip("/") + "/v1/videos/image2video"
    payload = {"model": model, "prompt": prompt, "image": image_b64,
               "duration": int(duration), "mode": mode}
    if image_tail_b64:
        payload["image_tail"] = image_tail_b64
    if negative_prompt:
        payload["negative_prompt"] = negative_prompt
    if cfg_scale is not None:
        payload["cfg_scale"] = float(cfg_scale)
    return _extract_task_id(_post(url, key, payload))


_DONE = ("succeed", "success", "succeeded", "completed", "done")
_FAIL = ("failed", "fail", "error")


def poll(base_url, key, task_id, timeout_s=600, interval_s=5, on_status=None):
    """Poll a task until a video URL appears. Raises KlingError on failure/timeout."""
    url = base_url.rstrip("/") + f"/v1/videos/{task_id}"
    waited = 0
    while True:
        body = _get(url, key)
        status, video_url = _extract_status_and_url(body)
        if on_status:
            on_status(status, waited)
        if status in _FAIL:
            raise KlingError(f"Kling task {task_id} failed: {_msg(body)}")
        if video_url and status not in _FAIL:
            return video_url
        if status in _DONE and not video_url:
            raise KlingError(f"Kling task {task_id} reported done but returned no video URL: {json.dumps(body)[:300]}")
        if waited >= timeout_s:
            raise KlingError(f"Kling task {task_id} timed out after {timeout_s}s (last status={status}).")
        time.sleep(interval_s)
        waited += interval_s


def download(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if _HAS_REQUESTS:
        with requests.get(url, stream=True, timeout=300) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    if chunk:
                        f.write(chunk)
    else:  # pragma: no cover
        with urllib.request.urlopen(url, timeout=300) as resp, open(dest, "wb") as f:
            f.write(resp.read())
    return dest


def encode_b64(raw_bytes):
    return base64.b64encode(raw_bytes).decode("ascii")
