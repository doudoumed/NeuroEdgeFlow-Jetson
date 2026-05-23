#!/usr/bin/env python3.8
# ─────────────────────────────────────────────────────────────────────────────
# NeuroEdgeFlow — Sprint 3 — cloud_client_video.py
# Always-Cloud video inference. Jetson sends a small JPEG per frame to the
# Triton "yolo_ensemble" (decode + preprocess + TensorRT inference on server).
# Only the JPEG (~20 KB) goes up — no 4.9 MB raw tensor.
#
# Run:  python3.8 cloud_client_video.py
# ─────────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np
import tritonclient.grpc as grpcclient
import time
import sys

# ─── CONFIG ──────────────────────────────────────────────────────────────────
TRITON_URL   = "10.0.20.10:8001"
MODEL_NAME   = "yolo_ensemble"        # ensemble: preprocess -> yolov5su
INPUT_NAME   = "jpeg_bytes"
OUTPUT_NAME  = "output0"

INPUT_SIZE   = 640
CONF_THRESH  = 0.3
IOU_THRESH   = 0.45
JPEG_QUALITY = 70                     # sprint default — 22 KB frames

VIDEO_IN     = "/home/nvidia/test_video.mp4"
VIDEO_OUT    = "/home/nvidia/result_video.mp4"
CSV_OUT      = "/home/nvidia/metrics_sprint3.csv"

COMPRESSION  = "gzip"

COCO_CLASSES = [
    "person","bicycle","car","motorbike","aeroplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
    "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
    "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl",
    "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
    "donut","cake","chair","sofa","pottedplant","bed","diningtable","toilet",
    "tvmonitor","laptop","mouse","remote","keyboard","cell phone","microwave",
    "oven","toaster","sink","refrigerator","book","clock","vase","scissors",
    "teddy bear","hair drier","toothbrush"
]

# ─── POSTPROCESSING ──────────────────────────────────────────────────────────
def xywh2xyxy(boxes, orig_w, orig_h):
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = (cx - w / 2) / INPUT_SIZE * orig_w
    y1 = (cy - h / 2) / INPUT_SIZE * orig_h
    x2 = (cx + w / 2) / INPUT_SIZE * orig_w
    y2 = (cy + h / 2) / INPUT_SIZE * orig_h
    return np.stack([x1, y1, x2, y2], axis=1)

def nms(boxes, scores, iou_thresh):
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
    pred = output[0].T                 # (8400, 84)
    boxes_xywh   = pred[:, 0:4]
    class_scores = pred[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    confs     = class_scores[np.arange(len(class_ids)), class_ids]

    mask = confs > CONF_THRESH
    boxes_xywh = boxes_xywh[mask]
    confs      = confs[mask]
    class_ids  = class_ids[mask]
    if len(boxes_xywh) == 0:
        return []

    boxes_xyxy = xywh2xyxy(boxes_xywh, orig_w, orig_h)
    keep = nms(boxes_xyxy, confs, IOU_THRESH)
    return [(int(boxes_xyxy[i][0]), int(boxes_xyxy[i][1]),
             int(boxes_xyxy[i][2]), int(boxes_xyxy[i][3]),
             float(confs[i]), int(class_ids[i])) for i in keep]

# ─── DISPLAY ─────────────────────────────────────────────────────────────────
def draw(frame, detections, grpc_ms, fps, frame_kb):
    for (x1, y1, x2, y2, conf, cls_id) in detections:
        name = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else "class_%d" % cls_id
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "%s %.2f" % (name, conf), (x1, max(y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    lines = [
        "FPS: %.1f" % fps,
        "gRPC RTT: %.0f ms" % grpc_ms,
        "Frame: %.1f KB" % frame_kb,
        "Detections: %d" % len(detections),
    ]
    cv2.rectangle(frame, (5, 5), (260, 110), (0, 0, 0), -1)
    for i, t in enumerate(lines):
        cv2.putText(frame, t, (10, 30 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame

# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(VIDEO_IN)
    if not cap.isOpened():
        print("[ERROR] Could not open video %s" % VIDEO_IN)
        sys.exit(1)

    in_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("[OK] Video: %dx%d, %d frames" % (in_w, in_h, total))

    channel_args = [
        ("grpc.max_send_message_length", 100 * 1024 * 1024),
        ("grpc.max_receive_message_length", 100 * 1024 * 1024),
    ]
    try:
        client = grpcclient.InferenceServerClient(
            url=TRITON_URL, verbose=False, channel_args=channel_args)
        if not client.is_server_ready():
            print("[ERROR] Triton not ready.")
            sys.exit(1)
        print("[OK] Connected to Triton at %s" % TRITON_URL)
    except Exception as e:
        print("[ERROR] Connection failed: %s" % e)
        sys.exit(1)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(VIDEO_OUT, fourcc, 10.0, (in_w, in_h))

    csv = open(CSV_OUT, "w")
    csv.write("frame,timestamp,grpc_ms,frame_kb,detections,fps\n")

    enc_params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    frame_idx  = 0
    t_start    = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        orig_h, orig_w = frame.shape[:2]

        # ── JPEG encode on Jetson (small payload) ──
        ok_enc, jpeg = cv2.imencode(".jpg", frame, enc_params)
        if not ok_enc:
            continue
        jpeg_bytes = jpeg.tobytes()
        frame_kb   = len(jpeg_bytes) / 1024.0

        # ── Send JPEG to the ensemble ──
        jpeg_np = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        inputs  = [grpcclient.InferInput(INPUT_NAME, [jpeg_np.shape[0]], "UINT8")]
        inputs[0].set_data_from_numpy(jpeg_np)
        outputs = [grpcclient.InferRequestedOutput(OUTPUT_NAME)]

        t0 = time.time()
        try:
            result = client.infer(
                model_name=MODEL_NAME, inputs=inputs, outputs=outputs,
                compression_algorithm=COMPRESSION)
        except Exception as e:
            print("[WARN] frame %d failed: %s" % (frame_idx, e))
            continue
        grpc_ms = (time.time() - t0) * 1000.0

        detections = postprocess(result.as_numpy(OUTPUT_NAME), orig_w, orig_h)
        fps = 1000.0 / grpc_ms if grpc_ms > 0 else 0.0

        out_frame = draw(frame, detections, grpc_ms, fps, frame_kb)
        writer.write(out_frame)

        csv.write("%d,%.3f,%.1f,%.1f,%d,%.2f\n"
                  % (frame_idx, time.time() - t_start, grpc_ms,
                     frame_kb, len(detections), fps))

        if frame_idx % 10 == 0 or frame_idx == 1:
            print("Frame %4d/%d  RTT %.0f ms  FPS %.1f  %d det  %.1f KB"
                  % (frame_idx, total, grpc_ms, fps,
                     len(detections), frame_kb))

    cap.release()
    writer.release()
    csv.close()

    elapsed = time.time() - t_start
    print("\n[DONE] %d frames in %.1f s  (avg %.2f FPS overall)"
          % (frame_idx, elapsed, frame_idx / elapsed if elapsed else 0))
    print("Video saved: %s" % VIDEO_OUT)
    print("Metrics CSV: %s" % CSV_OUT)

if __name__ == "__main__":
    main()
