#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
# cloud_infer_jpeg_v2.py — Client for the JPEG-based proxy
#
# Two functions:
#   cloud_infer_jpeg(server_ip, frame)        → benchmarking only (fast)
#   cloud_infer_jpeg_full(server_ip, frame)   → full object detection
#
# للاستخدام في collect_training_data.py:
#   from cloud_infer_jpeg_v2 import cloud_infer_jpeg
#   cloud_ms = cloud_infer_jpeg(SERVER_IP, frame)
#
# للاستخدام في main_pipeline.py:
#   from cloud_infer_jpeg_v2 import cloud_infer_jpeg_full
#   detections, cloud_ms = cloud_infer_jpeg_full(SERVER_IP, frame)
# ─────────────────────────────────────────────────────────────────────────────

import cv2
import time
import requests

PROXY_PORT       = 5000
JPEG_QUALITY     = 50
REQUEST_TIMEOUT  = 10


def _encode_jpeg(frame):
    """Encode frame to JPEG. Returns (bytes, encode_ms) or (None, -1)."""
    t0 = time.time()
    try:
        ret, jpeg_buffer = cv2.imencode(
            '.jpg', frame,
            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        )
        if not ret:
            return None, -1
        return jpeg_buffer.tobytes(), (time.time() - t0) * 1000.0
    except Exception:
        return None, -1


def cloud_infer_jpeg(server_ip, frame, port=PROXY_PORT, verbose=False):
    """
    Fast benchmark-only inference. Skips NMS for cleanest latency measurement.
    
    Use this for: data collection, RF training (collect_training_data.py)
    
    Args:
        server_ip: IP of the Flask proxy server
        frame:     BGR numpy frame
    
    Returns:
        latency_ms (float), or -1.0 if failed
    """
    jpeg_bytes, encode_ms = _encode_jpeg(frame)
    if jpeg_bytes is None:
        return -1.0
    
    url = f"http://{server_ip}:{port}/infer_fast"
    t0 = time.time()
    try:
        response = requests.post(
            url, data=jpeg_bytes,
            headers={'Content-Type': 'application/octet-stream'},
            timeout=REQUEST_TIMEOUT
        )
    except Exception as e:
        if verbose:
            print(f"[cloud-jpeg] failed: {e}")
        return -1.0
    
    if response.status_code != 200:
        if verbose:
            print(f"[cloud-jpeg] HTTP {response.status_code}")
        return -1.0
    
    total_ms = (time.time() - t0) * 1000.0
    
    if verbose:
        try:
            data = response.json()
            t = data.get("timings_ms", {})
            print(f"[cloud-jpeg-fast] payload={t.get('payload_kb', 0):.1f}KB  "
                  f"triton={t.get('triton_ms', 0):.1f}ms  "
                  f"server_total={t.get('total_ms', 0):.1f}ms  "
                  f"round_trip={total_ms:.1f}ms  "
                  f"anchors={data.get('anchor_count', '?')}")
        except Exception:
            pass
    
    return total_ms


def cloud_infer_jpeg_full(server_ip, frame, port=PROXY_PORT, verbose=False):
    """
    Full object detection with NMS and real bounding boxes.
    
    Use this for: real object detection (main_pipeline.py production use)
    
    Args:
        server_ip: IP of the Flask proxy server
        frame:     BGR numpy frame
    
    Returns:
        (detections, latency_ms)
            detections: list of dicts [{"x1","y1","x2","y2","conf","class_id","class_name"}, ...]
                        empty list if no detections
                        None if request failed
            latency_ms: float, -1.0 if failed
    """
    jpeg_bytes, encode_ms = _encode_jpeg(frame)
    if jpeg_bytes is None:
        return None, -1.0
    
    url = f"http://{server_ip}:{port}/infer"
    t0 = time.time()
    try:
        response = requests.post(
            url, data=jpeg_bytes,
            headers={'Content-Type': 'application/octet-stream'},
            timeout=REQUEST_TIMEOUT
        )
    except Exception as e:
        if verbose:
            print(f"[cloud-jpeg-full] failed: {e}")
        return None, -1.0
    
    if response.status_code != 200:
        if verbose:
            print(f"[cloud-jpeg-full] HTTP {response.status_code}")
        return None, -1.0
    
    total_ms = (time.time() - t0) * 1000.0
    
    try:
        data = response.json()
        detections = data.get("detections", [])
    except Exception:
        return None, -1.0
    
    if verbose:
        t = data.get("timings_ms", {})
        print(f"[cloud-jpeg-full] {len(detections)} detections  "
              f"triton={t.get('triton_ms', 0):.1f}ms  "
              f"nms={t.get('postprocess_ms', 0):.1f}ms  "
              f"round_trip={total_ms:.1f}ms")
        for d in detections[:5]:
            print(f"    {d['class_name']:15s} conf={d['conf']:.2f}  "
                  f"box=({d['x1']},{d['y1']})-({d['x2']},{d['y2']})")
    
    return detections, total_ms


def test_connection(server_ip, port=PROXY_PORT):
    url = f"http://{server_ip}:{port}/health"
    try:
        response = requests.get(url, timeout=3)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Proxy reachable at {server_ip}:{port}")
            print(f"   Triton ready: {data.get('triton_ready')}")
            print(f"   Model ready:  {data.get('model_ready')}")
            return True
        else:
            print(f"❌ Proxy returned {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot reach proxy: {e}")
        return False


# ─── Standalone test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import numpy as np
    
    if len(sys.argv) < 2:
        print("Usage: python3 cloud_infer_jpeg_v2.py <server_ip> [image_path]")
        print("Example 1: python3 cloud_infer_jpeg_v2.py 10.0.20.10")
        print("Example 2: python3 cloud_infer_jpeg_v2.py 10.0.20.10 bus.jpg")
        sys.exit(1)
    
    SERVER = sys.argv[1]
    
    # Load image: from file if given, else use a black frame
    if len(sys.argv) >= 3:
        frame = cv2.imread(sys.argv[2])
        if frame is None:
            print(f"Cannot read {sys.argv[2]}")
            sys.exit(1)
        print(f"\nLoaded: {sys.argv[2]} ({frame.shape[1]}x{frame.shape[0]})")
    else:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        print("\nUsing black frame (1280x720) — no detections expected")
    
    print(f"\n=== Testing connection ===")
    if not test_connection(SERVER):
        sys.exit(1)
    
    print(f"\n=== /infer_fast (benchmark mode) ===")
    for i in range(5):
        ms = cloud_infer_jpeg(SERVER, frame, verbose=True)
        if ms < 0:
            print(f"  Run {i+1}: FAILED")
            break
    
    print(f"\n=== /infer (full detection) ===")
    for i in range(3):
        detections, ms = cloud_infer_jpeg_full(SERVER, frame, verbose=True)
        if ms < 0:
            print(f"  Run {i+1}: FAILED")
            break
    
    print("\nDone.")
