#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
# cloud_infer_jpeg.py — JPEG-based cloud inference for the Jetson side
#
# يستبدل الدالة cloud_infer() القديمة في collect_training_data.py.
# بدل ما يبعث 4.9 MB raw tensor، يبعث JPEG ~50-100 KB.
# 
# للاستخدام في collect_training_data.py:
#   1. حط هذا الملف في نفس مجلد collect_training_data.py
#   2. في الـ collector، استبدل:
#         cloud_ms = cloud_infer(client, tensor) if cloud_ok else -1.0
#      بـ:
#         from cloud_infer_jpeg import cloud_infer_jpeg
#         cloud_ms = cloud_infer_jpeg(SERVER_IP, frame) if cloud_ok else -1.0
#   3. الـ collector ما يحتاجش الـ `preprocess(frame)` للـ cloud — يبعث الـ frame مباشرة
# ─────────────────────────────────────────────────────────────────────────────

import cv2
import time
import requests   # pip install requests

# ─── CONFIG ──────────────────────────────────────────────────────────────────
PROXY_PORT       = 5000               # نفس البورت في server_proxy.py
JPEG_QUALITY     = 85                 # 85 يعطي توازن بين الـ size والـ quality
REQUEST_TIMEOUT  = 10                 # ثواني — تجاوز الحد يعتبر fail
# ─────────────────────────────────────────────────────────────────────────────


def cloud_infer_jpeg(server_ip, frame, port=PROXY_PORT, verbose=False):
    """
    Cloud inference via JPEG-based proxy server.
    
    Args:
        server_ip: IP الـ server اللي يخدم عليه server_proxy.py
        frame:     BGR numpy frame (مش tensor!)
        port:      port الـ Flask proxy (default 5000)
        verbose:   لو True، يطبع breakdown للـ timing
    
    Returns:
        latency_ms (float): الـ round-trip بالـ milliseconds
        أو -1.0 لو فشل
    """
    # ── 1. Encode JPEG على الـ Jetson ────────────────────────────────────
    # هذا overhead صغير على الـ Jetson (~5-10 ms)
    # المكسب: 50x أصغر في الـ network transfer
    t_encode = time.time()
    try:
        ret, jpeg_buffer = cv2.imencode(
            '.jpg', frame, 
            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        )
        if not ret:
            if verbose:
                print("[cloud-jpeg] encode failed")
            return -1.0
        jpeg_bytes = jpeg_buffer.tobytes()
    except Exception as e:
        if verbose:
            print(f"[cloud-jpeg] encode exception: {e}")
        return -1.0
    
    encode_ms = (time.time() - t_encode) * 1000
    payload_kb = len(jpeg_bytes) / 1024
    
    if verbose:
        print(f"[cloud-jpeg] JPEG encoded: {payload_kb:.1f} KB in {encode_ms:.1f}ms")
    
    # ── 2. POST للـ proxy server ─────────────────────────────────────────
    url = f"http://{server_ip}:{port}/infer"
    
    t0 = time.time()
    try:
        response = requests.post(
            url,
            data=jpeg_bytes,
            headers={'Content-Type': 'application/octet-stream'},
            timeout=REQUEST_TIMEOUT
        )
    except requests.exceptions.Timeout:
        if verbose:
            print(f"[cloud-jpeg] timeout after {REQUEST_TIMEOUT}s")
        return -1.0
    except requests.exceptions.ConnectionError as e:
        if verbose:
            print(f"[cloud-jpeg] connection failed: {e}")
        return -1.0
    except Exception as e:
        if verbose:
            print(f"[cloud-jpeg] request failed: {e}")
        return -1.0
    
    total_ms = (time.time() - t0) * 1000
    
    # ── 3. Validate response ─────────────────────────────────────────────
    if response.status_code != 200:
        if verbose:
            print(f"[cloud-jpeg] server returned {response.status_code}: {response.text[:200]}")
        return -1.0
    
    if verbose:
        try:
            data = response.json()
            timings = data.get("timings_ms", {})
            print(f"[cloud-jpeg] success: payload={payload_kb:.1f}KB "
                  f"server_total={timings.get('total_ms', '?'):.1f}ms "
                  f"triton={timings.get('triton_ms', '?'):.1f}ms "
                  f"detections={data.get('num_detections', '?')}")
        except Exception:
            pass
    
    return total_ms


def test_connection(server_ip, port=PROXY_PORT):
    """Test if the proxy server is reachable."""
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


# ─── Standalone smoke test ───────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import numpy as np
    
    if len(sys.argv) < 2:
        print("Usage: python3 cloud_infer_jpeg.py <server_ip>")
        print("Example: python3 cloud_infer_jpeg.py 10.0.20.10")
        sys.exit(1)
    
    SERVER = sys.argv[1]
    
    print(f"\n=== Testing connection to {SERVER}:{PROXY_PORT} ===")
    if not test_connection(SERVER):
        sys.exit(1)
    
    # دير 5 inferences مع dummy frame
    print(f"\n=== Running 5 test inferences ===")
    dummy_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    
    for i in range(5):
        ms = cloud_infer_jpeg(SERVER, dummy_frame, verbose=True)
        if ms > 0:
            print(f"  Run {i+1}: {ms:.1f} ms")
        else:
            print(f"  Run {i+1}: FAILED")
    
    print("\nDone.")
