#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
# server_proxy.py — JPEG-based inference proxy for NeuroEdgeFlow
#
# يخدم كـ middle layer بين الـ Jetson و Triton:
#   1. الـ Jetson يبعث JPEG bytes (~50-100 KB بدل 4.9 MB tensor)
#   2. هاد الـ server يفك الـ JPEG ويعمل preprocessing
#   3. ينده Triton عبر gRPC على localhost (سريع)
#   4. يرجع النتيجة كـ JSON
#
# ركب على السيرفر:
#   pip install flask tritonclient[grpc] opencv-python-headless numpy
#
# شغل:
#   python3 server_proxy.py
#   # Server يستمع على port 5000
#
# اختبر من الـ Jetson:
#   curl http://<server-ip>:5000/health
# ─────────────────────────────────────────────────────────────────────────────

from flask import Flask, request, jsonify
import cv2
import numpy as np
import tritonclient.grpc as grpc
import time
import logging
import sys

# ─── CONFIG ──────────────────────────────────────────────────────────────────
TRITON_URL    = "localhost:8001"      # Triton على نفس السيرفر
MODEL_NAME    = "yolov5su"
INPUT_NAME    = "images"
OUTPUT_NAME   = "output0"
INPUT_SIZE    = 640
LISTEN_PORT   = 5000

# Logging
logging.basicConfig(
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO
)
log = logging.getLogger(__name__)

app = Flask(__name__)

# Triton client — connected once at startup
triton_client = None

def init_triton():
    """Initialize Triton client at startup."""
    global triton_client
    try:
        triton_client = grpc.InferenceServerClient(url=TRITON_URL, verbose=False)
        if not triton_client.is_server_ready():
            log.error(f"Triton server not ready at {TRITON_URL}")
            sys.exit(1)
        log.info(f"✅ Connected to Triton at {TRITON_URL}")
        log.info(f"✅ Model '{MODEL_NAME}' ready")
    except Exception as e:
        log.error(f"Failed to connect to Triton: {e}")
        sys.exit(1)


def preprocess(frame, size=INPUT_SIZE):
    """Same preprocessing as the original Jetson collector."""
    img = cv2.resize(frame, (size, size))
    img = img[:, :, ::-1].astype(np.float32) / 255.0   # BGR -> RGB, normalize
    img = np.transpose(img, (2, 0, 1))                 # HWC -> CHW
    img = np.expand_dims(img, axis=0)                  # add batch dim
    return np.ascontiguousarray(img)


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    try:
        ready = triton_client.is_server_ready()
        model_ready = triton_client.is_model_ready(MODEL_NAME)
        return jsonify({
            "status": "ok",
            "triton_ready": ready,
            "model_ready": model_ready
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route('/infer', methods=['POST'])
def infer():
    """
    Main inference endpoint.
    
    Input:  raw JPEG bytes في الـ request body
    Output: JSON بـ detections + timing breakdown
    """
    timings = {}
    t_start = time.time()
    
    # ── 1. Decode JPEG ──────────────────────────────────────────────────
    t0 = time.time()
    try:
        jpeg_bytes = np.frombuffer(request.data, dtype=np.uint8)
        if len(jpeg_bytes) == 0:
            return jsonify({"error": "empty payload"}), 400
        
        frame = cv2.imdecode(jpeg_bytes, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "invalid JPEG"}), 400
    except Exception as e:
        return jsonify({"error": f"decode failed: {e}"}), 400
    
    timings["decode_ms"] = (time.time() - t0) * 1000.0
    timings["payload_kb"] = len(jpeg_bytes) / 1024.0
    
    # ── 2. Preprocess ───────────────────────────────────────────────────
    t0 = time.time()
    try:
        tensor = preprocess(frame)
    except Exception as e:
        return jsonify({"error": f"preprocess failed: {e}"}), 500
    
    timings["preprocess_ms"] = (time.time() - t0) * 1000.0
    
    # ── 3. Triton inference ─────────────────────────────────────────────
    t0 = time.time()
    try:
        inp = grpc.InferInput(INPUT_NAME, tensor.shape, "FP32")
        inp.set_data_from_numpy(tensor)
        out = grpc.InferRequestedOutput(OUTPUT_NAME)
        
        result = triton_client.infer(
            model_name=MODEL_NAME,
            inputs=[inp],
            outputs=[out],
            compression_algorithm=None   # localhost, no need to compress
        )
        output = result.as_numpy(OUTPUT_NAME)
    except Exception as e:
        log.error(f"Triton inference failed: {e}")
        return jsonify({"error": f"inference failed: {e}"}), 500
    
    timings["triton_ms"] = (time.time() - t0) * 1000.0
    
    # ── 4. Count detections (basic NMS would go here for full impl) ─────
    # For benchmarking purposes, we just count non-zero confidence anchors
    # Output shape: (1, 84, 8400) — 84 = 4 box + 80 class probs
    # Anchor count for any class above 0.5 confidence
    try:
        class_scores = output[0, 4:, :]   # (80, 8400)
        max_scores = class_scores.max(axis=0)   # (8400,)
        num_detections = int((max_scores > 0.5).sum())
    except Exception:
        num_detections = -1
    
    timings["total_ms"] = (time.time() - t_start) * 1000.0
    
    return jsonify({
        "num_detections": num_detections,
        "output_shape": list(output.shape),
        "timings_ms": timings
    }), 200


@app.route('/stats', methods=['GET'])
def stats():
    """Triton model statistics."""
    try:
        s = triton_client.get_inference_statistics(model_name=MODEL_NAME)
        return jsonify({"stats": str(s)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── Main ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    log.info("="*60)
    log.info("  NeuroEdgeFlow Server Proxy — JPEG-based inference")
    log.info("="*60)
    log.info(f"  Triton URL:    {TRITON_URL}")
    log.info(f"  Model:         {MODEL_NAME}")
    log.info(f"  Input:         {INPUT_NAME} ({INPUT_SIZE}x{INPUT_SIZE})")
    log.info(f"  Output:        {OUTPUT_NAME}")
    log.info(f"  Listen port:   {LISTEN_PORT}")
    log.info("="*60)
    
    init_triton()
    
    # threaded=True عشان يقدر يتعامل مع multiple Jetsons في نفس الوقت
    # debug=False عشان مايعملش restart مع كل تعديل (production-like)
    app.run(host='0.0.0.0', port=LISTEN_PORT, threaded=True, debug=False)
