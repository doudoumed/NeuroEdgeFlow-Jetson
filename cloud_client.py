import cv2
import numpy as np
import tritonclient.grpc as grpcclient
import time
import os

# ─── CONFIG ────────────────────────────────────────────────────────────────
TRITON_URL   = "10.0.20.66:8001"
MODEL_NAME   = "yolov5su"
INPUT_NAME   = "input"
OUTPUT_NAME  = "output"

INPUT_SIZE   = 640
CONF_THRESH  = 0.3  # Lowered slightly to see more detections
IOU_THRESH   = 0.45

IMAGE_PATH   = "bus.jpg"

COCO_CLASSES = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet",
    "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]

# ─── PREPROCESSING ─────────────────────────────────────────────────────────
def preprocess(frame):
    img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    img = img[:, :, ::-1].astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return np.ascontiguousarray(img)

# ─── POSTPROCESSING ────────────────────────────────────────────────────────
def xywh2xyxy(boxes, orig_w, orig_h):
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = (cx - w / 2) / 640 * orig_w
    y1 = (cy - h / 2) / 640 * orig_h
    x2 = (cx + w / 2) / 640 * orig_w
    y2 = (cy + h / 2) / 640 * orig_h
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
        iou = (w * h) / (areas[i] + areas[order[1:]] - w * h)
        order = order[np.where(iou <= iou_thresh)[0] + 1]
    return keep

def postprocess(output, orig_w, orig_h):
    # Shape is (25200, 6) -> [cx, cy, w, h, conf, cls]
    pred = output[0]  
    boxes_xywh = pred[:, 0:4]
    confs = pred[:, 4]
    class_ids = pred[:, 5].astype(int)
    
    mask = confs > CONF_THRESH
    boxes_xywh = boxes_xywh[mask]
    confs = confs[mask]
    class_ids = class_ids[mask]
    
    if len(boxes_xywh) == 0:
        return []
        
    boxes_xyxy = xywh2xyxy(boxes_xywh, orig_w, orig_h)
    keep = nms(boxes_xyxy, confs, IOU_THRESH)
    
    return [(int(boxes_xyxy[i][0]), int(boxes_xyxy[i][1]),
             int(boxes_xyxy[i][2]), int(boxes_xyxy[i][3]),
             float(confs[i]), int(class_ids[i])) for i in keep]

# ─── DISPLAY ───────────────────────────────────────────────────────────────
def draw(frame, detections, grpc_ms):
    for (x1, y1, x2, y2, conf, cls_id) in detections:
        cls_name = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else "class_%d" % cls_id
        label = "%s %.2f" % (cls_name, conf)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, max(y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Simulate an FPS measure based on the one inference
    fps_est = 1000.0 / grpc_ms if grpc_ms > 0 else 0
    overlay = f"gRPC RTT: {grpc_ms:.0f}ms | Est. FPS: {fps_est:.1f}"
    
    # Black background for text
    cv2.rectangle(frame, (5, 5), (400, 45), (0,0,0), -1)
    cv2.putText(frame, overlay, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    return frame

# ─── MAIN ──────────────────────────────────────────────────────────────────
def main():
    frame = cv2.imread(IMAGE_PATH)
    if frame is None:
        print(f"[ERROR] Could not load {IMAGE_PATH}")
        return

    orig_h, orig_w = frame.shape[:2]

    try:
        client = grpcclient.InferenceServerClient(url=TRITON_URL, verbose=False)
        if not client.is_server_ready():
            print("[ERROR] Triton server not ready.")
            return
    except Exception as e:
        print(f"[ERROR] Connection Failed: {e}")
        return

    tensor = preprocess(frame)
    inputs = [grpcclient.InferInput(INPUT_NAME, tensor.shape, "FP32")]
    inputs[0].set_data_from_numpy(tensor)
    outputs = [grpcclient.InferRequestedOutput(OUTPUT_NAME)]

    t0 = time.time()
    result = client.infer(model_name=MODEL_NAME, inputs=inputs, outputs=outputs, client_timeout=120)
    grpc_ms = (time.time() - t0) * 1000
    
    detections = postprocess(result.as_numpy(OUTPUT_NAME), orig_w, orig_h)

    # User's requested output format
    print(f"\n✅ Detected {len(detections)} objects:\n")
    for (x1, y1, x2, y2, conf, cls_id) in detections:
        cls_name = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f"class_{cls_id}"
        print(f"  🔹 {cls_name:<20} confidence: {conf*100:.2f}%  box: [{x1}, {y1}, {x2}, {y2}]")
    
    res_img = draw(frame, detections, grpc_ms)
    cv2.imwrite("result.jpg", res_img)
    
    print(f"\n📸 Result saved to result.jpg with the FPS and RTT\n")

if __name__ == "__main__":
    main()



