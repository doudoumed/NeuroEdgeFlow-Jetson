#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
# server_proxy_v2.py — JPEG-based inference proxy with FULL object detection
#
# Improvements over v1:
#   - Returns real bounding boxes after NMS (not just anchor counts)
#   - Returns class names from COCO
#   - Two endpoints:
#       /infer        → full detection (slower, for production)
#       /infer_fast   → benchmark only (faster, for latency data collection)
#   - Better timing breakdown
#
# ركب على السيرفر:
#   pip install flask "tritonclient[grpc]" opencv-python-headless numpy
#
# شغل:
#   python3 server_proxy_v2.py
# ─────────────────────────────────────────────────────────────────────────────

from flask import Flask, request, jsonify
import cv2
import numpy as np
import tritonclient.grpc as grpc
import time
import logging
import sys

# ─── CONFIG ──────────────────────────────────────────────────────────────────
TRITON_URL     = "localhost:8001"
MODEL_NAME     = "yolov5su"
INPUT_NAME     = "images"
OUTPUT_NAME    = "output0"
INPUT_SIZE     = 640
LISTEN_PORT    = 5000
CONF_THRESHOLD = 0.25
IOU_THRESHOLD  = 0.45

# COCO class names (80 classes)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO
)
log = logging.getLogger(__name__)

app = Flask(__name__)
triton_client = None


# ─── Triton init ─────────────────────────────────────────────────────────────
def init_triton():
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


# ─── Preprocessing ───────────────────────────────────────────────────────────
def preprocess(frame, size=INPUT_SIZE):
    img = cv2.resize(frame, (size, size))
    img = img[:, :, ::-1].astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return np.ascontiguousarray(img)


# ─── Postprocessing (NMS + box extraction) ───────────────────────────────────
def xywh2xyxy(boxes, orig_w, orig_h, input_size=INPUT_SIZE):
    """Convert center-format boxes to corner format in original image space."""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = (cx - w / 2) / input_size * orig_w
    y1 = (cy - h / 2) / input_size * orig_h
    x2 = (cx + w / 2) / input_size * orig_w
    y2 = (cy + h / 2) / input_size * orig_h
    return np.stack([x1, y1, x2, y2], axis=1)


def nms(boxes, scores, iou_thresh=IOU_THRESHOLD):
    """Pure-numpy non-maximum suppression."""
    if len(boxes) == 0:
        return []
    
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[np.where(iou <= iou_thresh)[0] + 1]
    
    return keep


def postprocess(output, orig_w, orig_h):
    """
    Decode YOLOv5su output to detections.
    
    Args:
        output: numpy array, shape (1, 84, 8400)
        orig_w, orig_h: original image dimensions for scaling
    
    Returns:
        list of dicts: [{"x1","y1","x2","y2","conf","class_id","class_name"}, ...]
    """
    # Transpose to (8400, 84)
    pred = output[0].T
    
    # Split: first 4 cols are box coords, next 80 are class probs
    boxes_xywh = pred[:, 0:4]
    class_scores = pred[:, 4:]
    
    # Best class per anchor
    class_ids = np.argmax(class_scores, axis=1)
    confs = class_scores[np.arange(len(class_ids)), class_ids]
    
    # Filter by confidence
    mask = confs > CONF_THRESHOLD
    boxes_xywh = boxes_xywh[mask]
    confs = confs[mask]
    class_ids = class_ids[mask]
    
    if len(boxes_xywh) == 0:
        return []
    
    # Convert to xyxy + scale to original image
    boxes_xyxy = xywh2xyxy(boxes_xywh, orig_w, orig_h)
    
    # NMS
    keep = nms(boxes_xyxy, confs)
    
    # Build detection list
    detections = []
    for i in keep:
        cls_id = int(class_ids[i])
        cls_name = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f"class_{cls_id}"
        detections.append({
            "x1": int(boxes_xyxy[i][0]),
            "y1": int(boxes_xyxy[i][1]),
            "x2": int(boxes_xyxy[i][2]),
            "y2": int(boxes_xyxy[i][3]),
            "conf": float(confs[i]),
            "class_id": cls_id,
            "class_name": cls_name
        })
    
    return detections


# ─── Triton inference helper ─────────────────────────────────────────────────
def run_triton(tensor):
    """Returns: (output array, triton_ms)."""
    inp = grpc.InferInput(INPUT_NAME, tensor.shape, "FP32")
    inp.set_data_from_numpy(tensor)
    out = grpc.InferRequestedOutput(OUTPUT_NAME)
    
    t0 = time.time()
    result = triton_client.infer(
        model_name=MODEL_NAME,
        inputs=[inp],
        outputs=[out],
        compression_algorithm=None
    )
    triton_ms = (time.time() - t0) * 1000.0
    
    return result.as_numpy(OUTPUT_NAME), triton_ms


# ─── Endpoints ───────────────────────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    try:
        return jsonify({
            "status": "ok",
            "triton_ready": triton_client.is_server_ready(),
            "model_ready": triton_client.is_model_ready(MODEL_NAME)
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route('/infer', methods=['POST'])
def infer_full():
    """
    Full inference with NMS and real bounding boxes.
    For production use (real object detection).
    """
    t_start = time.time()
    timings = {}
    
    # Decode JPEG
    t0 = time.time()
    jpeg_bytes = np.frombuffer(request.data, dtype=np.uint8)
    if len(jpeg_bytes) == 0:
        return jsonify({"error": "empty payload"}), 400
    
    frame = cv2.imdecode(jpeg_bytes, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "invalid JPEG"}), 400
    
    orig_h, orig_w = frame.shape[:2]
    timings["decode_ms"] = (time.time() - t0) * 1000.0
    timings["payload_kb"] = len(jpeg_bytes) / 1024.0
    
    # Preprocess
    t0 = time.time()
    tensor = preprocess(frame)
    timings["preprocess_ms"] = (time.time() - t0) * 1000.0
    
    # Triton inference
    try:
        output, triton_ms = run_triton(tensor)
        timings["triton_ms"] = triton_ms
    except Exception as e:
        return jsonify({"error": f"inference failed: {e}"}), 500
    
    # Postprocess (NMS + box decoding)
    t0 = time.time()
    detections = postprocess(output, orig_w, orig_h)
    timings["postprocess_ms"] = (time.time() - t0) * 1000.0
    
    timings["total_ms"] = (time.time() - t_start) * 1000.0
    
    return jsonify({
        "detections": detections,
        "num_detections": len(detections),
        "image_size": {"width": orig_w, "height": orig_h},
        "timings_ms": timings
    }), 200


@app.route('/infer_fast', methods=['POST'])
def infer_fast():
    """
    Benchmarking-only endpoint. Skips NMS to give the cleanest latency
    measurement for the inference pipeline itself. Returns detection count
    based on raw anchor confidence (NOT after NMS).
    
    Use this for the collector / RF training data collection.
    """
    t_start = time.time()
    timings = {}
    
    # Decode JPEG
    t0 = time.time()
    jpeg_bytes = np.frombuffer(request.data, dtype=np.uint8)
    if len(jpeg_bytes) == 0:
        return jsonify({"error": "empty payload"}), 400
    
    frame = cv2.imdecode(jpeg_bytes, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "invalid JPEG"}), 400
    
    timings["decode_ms"] = (time.time() - t0) * 1000.0
    timings["payload_kb"] = len(jpeg_bytes) / 1024.0
    
    # Preprocess + inference
    t0 = time.time()
    tensor = preprocess(frame)
    timings["preprocess_ms"] = (time.time() - t0) * 1000.0
    
    try:
        output, triton_ms = run_triton(tensor)
        timings["triton_ms"] = triton_ms
    except Exception as e:
        return jsonify({"error": f"inference failed: {e}"}), 500
    
    # Quick anchor count (no NMS)
    try:
        class_scores = output[0, 4:, :]
        max_scores = class_scores.max(axis=0)
        anchor_count = int((max_scores > CONF_THRESHOLD).sum())
    except Exception:
        anchor_count = -1
    
    timings["total_ms"] = (time.time() - t_start) * 1000.0
    
    return jsonify({
        "anchor_count": anchor_count,
        "timings_ms": timings
    }), 200


@app.route('/stats', methods=['GET'])
def stats():
    try:
        s = triton_client.get_inference_statistics(model_name=MODEL_NAME)
        return jsonify({"stats": str(s)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── Main ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    log.info("="*60)
    log.info("  NeuroEdgeFlow Server Proxy v2 — Full Object Detection")
    log.info("="*60)
    log.info(f"  Triton URL:        {TRITON_URL}")
    log.info(f"  Model:             {MODEL_NAME}")
    log.info(f"  Input/Output:      {INPUT_NAME} / {OUTPUT_NAME}")
    log.info(f"  Conf threshold:    {CONF_THRESHOLD}")
    log.info(f"  IoU threshold:     {IOU_THRESHOLD}")
    log.info(f"  Listen port:       {LISTEN_PORT}")
    log.info(f"  Endpoints:")
    log.info(f"    POST /infer        → full detection (NMS + boxes + classes)")
    log.info(f"    POST /infer_fast   → benchmark only (no NMS)")
    log.info(f"    GET  /health       → health check")
    log.info(f"    GET  /stats        → Triton statistics")
    log.info("="*60)
    
    init_triton()
    app.run(host='0.0.0.0', port=LISTEN_PORT, threaded=True, debug=False)
