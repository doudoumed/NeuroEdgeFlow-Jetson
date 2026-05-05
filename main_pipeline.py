#!/usr/bin/env python3
# =============================================================================
# main_pipeline.py — NeuroEdgeFlow Sprint 5
# Unified Adaptive Pipeline
#
# Architecture (three threads, one queue):
#
#   Thread 1 — Frame Capture (CSI camera via GStreamer subprocess)
#              Continuously reads frames → FrameQueue (maxsize=2)
#              Never blocks on inference — drops oldest if queue full
#
#   Thread 2 — Engine Control Loop (AdaptiveEngine background thread)
#              Polls network conditions every 2 s
#              Updates engine._state (LOCAL or CLOUD) atomically
#              Does NOT touch the frame stream
#
#   Main thread — Inference Loop
#              Reads frames from FrameQueue (blocks max FRAME_TIMEOUT_S)
#              Reads engine.current_mode — instant string attribute read
#              Routes frame to cloud_infer() or edge_infer() accordingly
#              Draws detections, updates CSV, updates Grafana FPS
#
# "Without dropping frames" guarantee:
#   The engine control loop NEVER blocks the inference loop.
#   A mode switch is a single attribute write (engine._state = new_mode).
#   The inference loop reads engine.current_mode on every frame — instant.
#   At most ONE frame processes in the old mode after a switch decision.
#   Frame capture drops frames only if inference is slower than capture —
#   this is the same behaviour as Sprint 3 cloud_client.py (camera-limited).
#
# Python 3.6 compatible — Jetson TX2, JetPack R32.7.6
# Run: OPENBLAS_CORETYPE=ARMV8 python3 ~/main_pipeline.py
# =============================================================================

from __future__ import print_function

import csv
import logging
import os
import queue
import signal
import subprocess
import sys
import threading
import time

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Local modules
# ---------------------------------------------------------------------------
from adaptive_engine import (AdaptiveEngine, InferenceMode, HYSTERESIS_COUNT,
                              FALLBACK_ERROR_THRESHOLD, FALLBACK_RECOVERY_POLLS)

# edge_pipeline provides run_one_frame() for LOCAL mode
import edge_pipeline

# ---------------------------------------------------------------------------
# Triton gRPC client — Sprint 3 dependency
# ---------------------------------------------------------------------------
try:
    import tritonclient.grpc as grpcclient
    TRITON_AVAILABLE = True
except (ImportError, AttributeError):
    TRITON_AVAILABLE = False

# =============================================================================
# CONFIGURATION
# =============================================================================

LAPTOP_IP        = "192.168.1.72"    # update each session with sed
TRITON_URL       = LAPTOP_IP + ":8001"
TRITON_MODEL     = "yolov5su"

POLL_INTERVAL_S  = 2.0               # engine evaluation interval (Task 02)
HYSTERESIS       = HYSTERESIS_COUNT  # 3 (Task 04)

# Camera — CSI, Bayer BG10, nvarguscamerasrc
CAM_WIDTH        = 1280
CAM_HEIGHT       = 720
CAM_FPS          = 30

# Source — Sprint 5 Issue 1 Fix
#   0 = default camera
#   "path/to/video.mp4" = video file
#   "path/to/image.jpg" = image loop (Task 02 fallback)
VIDEO_SOURCE     = "/home/nvidia/yolov5/data/images/bus.jpg"

# Queue — maxsize=2 so capture stays 1 frame ahead of inference maximum.
# If inference is slow the oldest frame is dropped rather than blocking capture.
FRAME_QUEUE_SIZE = 2
FRAME_TIMEOUT_S  = 1.0    # max wait for a frame before logging a warning

# Inference
JPEG_QUALITY     = 70     # compression for gRPC frames (matches Sprint 3)
CONF_THRESHOLD   = 0.25
NMS_THRESHOLD    = 0.45
NUM_CLASSES      = 80

# Logging
LOG_FILE         = os.path.expanduser("~/main_pipeline.csv")
LOG_LEVEL        = logging.DEBUG

# Display — set to True if running via SSH/Headless
HEADLESS         = True

CSV_HEADER = [
    "timestamp", "frame_id", "mode",
    "encode_ms",        # JPEG encode time (CLOUD only, 0 for LOCAL)
    "inference_ms",     # gRPC round-trip (CLOUD) or TRT time (LOCAL)
    "decode_ms",        # JPEG decode time (CLOUD only, 0 for LOCAL)
    "total_ms",         # wall-clock frame time
    "num_detections",
    "fps",
    "pending_mode",     # engine hysteresis state
    "pending_count",
]

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s  [%(levelname)-8s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger("main_pipeline")


# =============================================================================
# CSV HELPER
# =============================================================================

def _init_csv(filepath):
    if not os.path.exists(filepath):
        try:
            with open(filepath, "w") as f:
                csv.writer(f).writerow(CSV_HEADER)
            log.info("CSV log created: %s", filepath)
        except IOError as exc:
            log.error("Cannot create CSV: %s", exc)


def _write_csv(filepath, row):
    try:
        with open(filepath, "a") as f:
            csv.DictWriter(f, fieldnames=CSV_HEADER).writerow(row)
    except IOError as exc:
        log.error("CSV write failed: %s", exc)


# =============================================================================
# FRAME CAPTURE THREAD
# =============================================================================

class FrameCapture(object):
    """
    Reads frames from the CSI camera via GStreamer subprocess in a background
    thread and puts them into a bounded queue.

    maxsize=FRAME_QUEUE_SIZE (2): if the inference loop falls behind,
    the oldest frame is discarded and replaced — capture never blocks.
    This is identical to the Sprint 3 cloud_client.py capture behaviour.
    """

    GST_PIPELINE = (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM),width={w},height={h},framerate={fps}/1 ! "
        "nvvidconv ! video/x-raw,format=BGRx ! "
        "videoconvert ! video/x-raw,format=BGR ! appsink drop=true"
    ).format(w=CAM_WIDTH, h=CAM_HEIGHT, fps=CAM_FPS)

    def __init__(self, frame_queue, source=VIDEO_SOURCE):
        self._queue   = frame_queue
        self._source  = source
        self._running = False
        self._thread  = None
        self._proc    = None
        self._frame_size = CAM_WIDTH * CAM_HEIGHT * 3

    def start(self):
        self._running = True
        self._thread  = threading.Thread(
            target=self._capture_loop, name="frame-capture"
        )
        self._thread.daemon = True
        self._thread.start()
        log.info("FrameCapture started")

    def stop(self):
        self._running = False
        if self._proc is not None:
            try:
                self._proc.terminate()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        log.info("FrameCapture stopped")

        self._running = False
        log.warning("FrameCapture loop ended")

    def _capture_loop(self):
        """
        Supports Camera (GStreamer), Video File, or Image loop.
        """
        if isinstance(self._source, str) and self._source.endswith((".jpg", ".png", ".jpeg")):
            self._image_loop()
        else:
            self._gstreamer_loop()

    def _image_loop(self):
        log.info("Starting IMAGE LOOP fallback: %s", self._source)
        img = cv2.imread(self._source)
        if img is None:
            log.error("Failed to read image source: %s", self._source)
            return
        
        # Sanity check for Image shape
        log.info("Sanity check — image shape: %s", img.shape)

        while self._running:
            # Resize internal to match Cam dimensions if needed
            frame = cv2.resize(img, (CAM_WIDTH, CAM_HEIGHT))
            
            if self._queue.full():
                try: self._queue.get_nowait()
                except queue.Empty: pass
            
            try: self._queue.put_nowait(frame.copy())
            except queue.Full: pass
            
            time.sleep(1.0 / CAM_FPS)

    def _gstreamer_loop(self):
        """
        Launches GStreamer as a subprocess, reads raw BGR bytes frame by frame.
        """
        env = os.environ.copy()
        env["OPENBLAS_CORETYPE"] = "ARMV8"

        # Determine if source is camera ID or path
        is_gstreamer = True
        if isinstance(self._source, int):
            # USB Camera
            pipeline = "v4l2src device=/dev/video{0} ! video/x-raw,width={1},height={2} ! videoconvert ! video/x-raw,format=BGR ! appsink".format(self._source, CAM_WIDTH, CAM_HEIGHT)
        elif self._source == "nvarguscamerasrc":
            pipeline = self.GST_PIPELINE
        else:
            # Video file native 
            pipeline = self._source
            is_gstreamer = False

        if is_gstreamer:
            reader_script = (
                "import cv2, sys, time\n"
                "cap = cv2.VideoCapture('{pipeline}', cv2.CAP_GSTREAMER)\n"
                "if not cap.isOpened():\n"
                "    sys.exit(1)\n"
            ).format(pipeline=pipeline)
        else:
            reader_script = (
                "import cv2, sys, time\n"
                "cap = cv2.VideoCapture('{pipeline}')\n"
                "if not cap.isOpened():\n"
                "    sys.exit(1)\n"
                "fps = cap.get(cv2.CAP_PROP_FPS)\n"
                "frame_time = 1.0 / fps if fps > 0 else 1.0/30.0\n"
            ).format(pipeline=pipeline)
        
        reader_script += (
            "while True:\n"
            "    ret, frame = cap.read()\n"
            "    if not ret:\n"
            "        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)\n" # Loop the video
            "        continue\n"
            "    frame = cv2.resize(frame, ({w}, {h}))\n"
            "    sys.stdout.buffer.write(frame.tobytes())\n"
            "    sys.stdout.buffer.flush()\n"
            "    if not {is_gstreamer}: time.sleep(max(0, frame_time - 0.01))\n"
        ).format(is_gstreamer=is_gstreamer, w=CAM_WIDTH, h=CAM_HEIGHT)

        self._proc = subprocess.Popen(
            ["python3", "-c", reader_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env
        )

        log.info("Camera subprocess started (Source: %s)", self._source)

        while self._running:
            raw = self._proc.stdout.read(self._frame_size)
            if len(raw) != self._frame_size:
                log.warning("Camera read short (%d bytes) — source might be exhausted",
                            len(raw))
                break

            frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                CAM_HEIGHT, CAM_WIDTH, 3
            )

            # Non-blocking put: discard oldest if queue is full
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass

            try:
                self._queue.put_nowait(frame.copy())
            except queue.Full:
                pass

        self._running = False


# =============================================================================
# CLOUD INFERENCE — Triton gRPC (Sprint 3)
# =============================================================================

def _init_triton():
    """Initialises Triton gRPC client. Returns client or None."""
    if not TRITON_AVAILABLE:
        log.warning("tritonclient not available — CLOUD inference stubbed")
        return None
    try:
        client = grpcclient.InferenceServerClient(url=TRITON_URL, verbose=False)
        if not client.is_server_live():
            log.warning("Triton server not live at %s", TRITON_URL)
            return None
        log.info("Triton client connected: %s", TRITON_URL)
        return client
    except Exception as exc:
        log.warning("Triton connect failed: %s", exc)
        return None


def cloud_infer(frame, triton_client):
    """
    Send one frame to Triton via gRPC, return (detections, encode_ms,
    inference_ms, decode_ms).

    Args:
        frame         — numpy BGR frame
        triton_client — grpcclient.InferenceServerClient or None (stub)

    Returns:
        detections   — list of (x1,y1,x2,y2,conf,class_id)
        encode_ms    — JPEG encode time
        inference_ms — gRPC round-trip time
        decode_ms    — JPEG decode time (0 — server handles decode)
    """
    # ── JPEG encode ──────────────────────────────────────────────────────────
    t0 = time.time()
    ret, buf = cv2.imencode(
        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    )
    if not ret:
        return [], 0.0, 0.0, 0.0
    encode_ms = (time.time() - t0) * 1000.0
    jpg_bytes = buf.tobytes()

    if triton_client is None:
        # Stub — no server available
        time.sleep(0.35)
        return [], encode_ms, 350.0, 0.0

    # ── gRPC send ─────────────────────────────────────────────────────────────
    t1 = time.time()
    try:
        inp = grpcclient.InferInput("input", [1, 3, 640, 640], "FP32")

        # Decode JPEG and preprocess to match Triton model input
        img = cv2.imdecode(np.frombuffer(jpg_bytes, np.uint8),
                           cv2.IMREAD_COLOR)
        img = cv2.resize(img, (640, 640))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[np.newaxis]
        inp.set_data_from_numpy(np.ascontiguousarray(img))

        out  = grpcclient.InferRequestedOutput("output")
        resp = triton_client.infer(
            model_name=TRITON_MODEL, inputs=[inp], outputs=[out]
        )
        inference_ms = (time.time() - t1) * 1000.0

        raw = resp.as_numpy("output")  # [1, 25200, 6] for yolov5su

        # ── NMS postprocess ──────────────────────────────────────────────────
        t2 = time.time()
        detections = _nms_postprocess(raw, frame.shape[0], frame.shape[1])
        decode_ms  = (time.time() - t2) * 1000.0

        return detections, encode_ms, inference_ms, decode_ms

    except Exception as exc:
        log.warning("cloud_infer error: %s", exc)
        inference_ms = (time.time() - t1) * 1000.0
        return [], encode_ms, inference_ms, 0.0


def _nms_postprocess(raw, orig_h, orig_w):
    """YOLOv5su output [1, 25200, 6] → NMS filtered detections.
    Each row: [cx, cy, w, h, obj_conf, class_id]  (no per-class probs — already argmaxed)
    """
    detections = []
    try:
        pred = raw[0]  # [25200, 6]
        boxes, scores, class_ids = [], [], []

        for row in pred:
            obj_conf = float(row[4])
            if obj_conf < CONF_THRESHOLD:
                continue
            class_id = int(row[5])
            score    = obj_conf
            if score < CONF_THRESHOLD:
                continue
            cx, cy, w, h = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            x1 = int((cx - w / 2) * orig_w / 640)
            y1 = int((cy - h / 2) * orig_h / 640)
            x2 = int((cx + w / 2) * orig_w / 640)
            y2 = int((cy + h / 2) * orig_h / 640)
            boxes.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(score)
            class_ids.append(class_id)

        if boxes:
            def py_nms(boxes, scores, iou_threshold):
                x1 = np.array([b[0] for b in boxes])
                y1 = np.array([b[1] for b in boxes])
                x2 = np.array([b[2] for b in boxes])
                y2 = np.array([b[3] for b in boxes])
                areas = (x2 - x1) * (y2 - y1)
                order = np.argsort(scores)[::-1]
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
                    ovr = inter / (areas[i] + areas[order[1:]] - inter)
                    inds = np.where(ovr <= iou_threshold)[0]
                    order = order[inds + 1]
                return keep

            indices = py_nms(boxes, scores, NMS_THRESHOLD)
            for i in indices:
                x, y, w, h = boxes[i]
                detections.append((x, y, x + w, y + h,
                                   scores[i], class_ids[i]))
    except Exception as exc:
        log.warning("NMS postprocess error: %s", exc)
    return detections


# =============================================================================
# DRAWING HELPER
# =============================================================================


MODE_COLORS = {
    InferenceMode.LOCAL: (0, 200, 0),    # green
    InferenceMode.CLOUD: (200, 100, 0),  # blue-ish
}

COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra",
    "giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove",
    "skateboard","surfboard","tennis racket","bottle","wine glass","cup",
    "fork","knife","spoon","bowl","banana","apple","sandwich","orange",
    "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
    "toothbrush"
]


def _draw(frame, detections, mode, fps, pending_mode, pending_count,
          fallback_locked=False):
    """Draw bounding boxes, mode label, FPS, hysteresis and fallback status."""
    color = (0, 0, 220) if fallback_locked else MODE_COLORS.get(mode, (255, 255, 255))

    for (x1, y1, x2, y2, conf, cls_id) in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = "{0} {1:.0f}%".format(
            COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else str(cls_id),
            conf * 100
        )
        cv2.putText(frame, label, (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Mode label top-left
    mode_label = "MODE: " + mode + (" [FALLBACK]" if fallback_locked else "")
    cv2.putText(frame, mode_label,
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    # FPS
    cv2.putText(frame, "FPS: {:.1f}".format(fps),
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    # Fallback locked indicator — bright red bar
    if fallback_locked:
        cv2.putText(frame, "!! FALLBACK LOCKED — waiting for recovery ({0}/{1})".format(
                        pending_count, FALLBACK_RECOVERY_POLLS),
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 220), 2)
    # Hysteresis pending indicator (only shown when not in fallback)
    elif pending_mode and pending_count > 0:
        hyst_label = "PENDING {0} ({1}/{2})".format(
            pending_mode, pending_count, HYSTERESIS
        )
        cv2.putText(frame, hyst_label,
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)

    return frame


# =============================================================================
# ADAPTIVE PIPELINE — main class
# =============================================================================

class AdaptivePipeline(object):
    """
    Integrates AdaptiveEngine with continuous frame capture and inference.

    The engine's 2s control loop runs in its own thread (inside AdaptiveEngine).
    This class owns the frame capture thread and the inference loop (main thread).

    No frames are dropped due to the engine evaluation — the engine never
    touches the frame stream. A mode switch takes effect on the next frame
    after engine._state is updated by the engine's background thread.
    """

    def __init__(self):
        self._frame_queue  = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
        self._capture      = FrameCapture(self._frame_queue)
        self._triton       = None
        self._frame_id     = 0
        self._running      = False
        self._fps_tracker  = _FPSTracker(window=30)

        # Fallback — Task 06: consecutive cloud error counter
        # Incremented each frame when cloud_infer() fails.
        # Calls engine.trigger_fallback() when it reaches FALLBACK_ERROR_THRESHOLD.
        self._cloud_error_count = 0
        self._dets_zero_streak   = 0

        # AdaptiveEngine — Task 02/04: decide() + hysteresis
        # PipelineManager is BYPASSED — main_pipeline.py owns the inference
        # routing directly. Engine is used only for its state machine.
        self._engine = AdaptiveEngine(
            poll_interval=POLL_INTERVAL_S,
            log_file=os.path.expanduser("~/adaptive_engine.csv"),
            hysteresis_count=HYSTERESIS
        )

        _init_csv(LOG_FILE)

    def start(self):
        log.info("=== AdaptivePipeline starting ===")

        # Start engine (control loop thread + network_monitor)
        self._engine.start()
        log.info("Engine started — initial mode: %s", self._engine.current_mode)

        # Connect to Triton (non-fatal if unavailable)
        self._triton = _init_triton()

        # Start frame capture thread
        self._capture.start()

        # Brief pause — let camera and network_monitor initialise
        time.sleep(2.0)

        self._running = True
        log.info("=== AdaptivePipeline running ===")
        self._inference_loop()

    def stop(self):
        log.info("=== AdaptivePipeline stopping ===")
        self._running = False
        self._capture.stop()
        self._engine.stop()
        cv2.destroyAllWindows()
        log.info("=== AdaptivePipeline stopped ===")

    # ── inference loop ────────────────────────────────────────────────────────

    def _inference_loop(self):
        """
        Main inference loop — runs in the calling (main) thread.

        Per-frame steps:
          1. Get frame from queue (blocks up to FRAME_TIMEOUT_S)
          2. Read engine.current_mode — instant attribute read, never blocks
          3. Route to cloud_infer() or edge_pipeline.run_one_frame()
          4. Draw detections + mode overlay
          5. Display (OpenCV window)
          6. Log to CSV
        """
        while self._running:
            # ── 1. GET FRAME ─────────────────────────────────────────────────
            try:
                frame = self._frame_queue.get(timeout=FRAME_TIMEOUT_S)
                # Notify engine of current queue depth (for XGBoost temporal features)
                self._engine.update_queue_depth(self._frame_queue.qsize())
            except queue.Empty:
                log.warning("Frame queue empty — camera may have stalled")
                continue

            t_frame_start = time.time()
            self._frame_id += 1
            ts = time.strftime("%Y-%m-%d %H:%M:%S")

            # ── 2. READ MODE — instant, never blocks ─────────────────────────
            mode            = self._engine.current_mode
            pending_mode    = self._engine.pending_mode
            pending_count   = self._engine.pending_count
            fallback_locked = self._engine.fallback_locked

            # ── 3. ROUTE TO INFERENCE ────────────────────────────────────────
            encode_ms = 0.0
            decode_ms = 0.0

            if mode == InferenceMode.CLOUD:
                detections, encode_ms, inference_ms, decode_ms = \
                    cloud_infer(frame, self._triton)

                # ── Fallback error tracking (Task 06) ────────────────────────
                # A cloud error is: encode succeeded (encode_ms > 0) but
                # inference returned no result in a plausible time, OR
                # encode itself failed (encode_ms == 0).
                cloud_error = (encode_ms == 0.0 or inference_ms > 8000.0)
                if cloud_error:
                    self._cloud_error_count += 1
                    log.warning(
                        "Cloud inference error %d/%d  "
                        "(encode_ms=%.1f  inference_ms=%.1f)",
                        self._cloud_error_count, FALLBACK_ERROR_THRESHOLD,
                        encode_ms, inference_ms
                    )
                    if self._cloud_error_count >= FALLBACK_ERROR_THRESHOLD:
                        try:
                            cond = {}
                            from network_monitor import get_current_conditions
                            cond = get_current_conditions()
                        except Exception:
                            pass
                        cond["consecutive_errors"] = self._cloud_error_count
                        self._engine.trigger_fallback(
                            "cloud_infer failed {0}x consecutively".format(
                                self._cloud_error_count),
                            conditions=cond
                        )
                        self._cloud_error_count = 0
                else:
                    # Successful cloud frame — reset error counter
                    self._cloud_error_count = 0

            else:
                # LOCAL or FALLBACK LOCKED — TensorRT via edge_pipeline
                detections, inference_ms = edge_pipeline.run_one_frame(frame)
                # Reset cloud error counter whenever not in CLOUD mode
                self._cloud_error_count = 0
                # FEEDBACK: Update engine with real-time local latency
                self._engine.update_local_latency(inference_ms)

            # ── 3.1 SANITY CHECK (Issue 2) ──────────────────────────────────
            if len(detections) == 0:
                self._dets_zero_streak += 1
                if self._dets_zero_streak >= 100:
                    if self._frame_id % 30 == 0:
                        log.warning("!! SANITY ALERT !! 100+ frames with 0 detections. "
                                    "Check if model engine is loaded properly.")
            else:
                self._dets_zero_streak = 0

            total_ms = (time.time() - t_frame_start) * 1000.0
            fps      = self._fps_tracker.update()

            # ── 4. DRAW ──────────────────────────────────────────────────────
            display = _draw(
                frame.copy(), detections, mode, fps,
                pending_mode, pending_count, fallback_locked
            )

            # ── 5. DISPLAY ───────────────────────────────────────────────────
            if not HEADLESS:
                cv2.imshow("NeuroEdgeFlow — Sprint 5", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self._running = False
                    break

            # ── 6. LOG ───────────────────────────────────────────────────────
            _write_csv(LOG_FILE, {
                "timestamp":     ts,
                "frame_id":      self._frame_id,
                "mode":          mode,
                "encode_ms":     "{:.2f}".format(encode_ms),
                "inference_ms":  "{:.2f}".format(inference_ms),
                "decode_ms":     "{:.2f}".format(decode_ms),
                "total_ms":      "{:.2f}".format(total_ms),
                "num_detections":len(detections),
                "fps":           "{:.2f}".format(fps),
                "pending_mode":  pending_mode if pending_mode else "",
                "pending_count": pending_count,
            })

            log.debug(
                "frame %d  mode=%-5s  inf=%.1f ms  fps=%.1f  dets=%d  "
                "pending=%s(%d/%d)",
                self._frame_id, mode, inference_ms, fps, len(detections),
                pending_mode or "-", pending_count, HYSTERESIS
            )


# =============================================================================
# FPS TRACKER
# =============================================================================

class _FPSTracker(object):
    """Rolling window FPS measurement."""

    def __init__(self, window=30):
        self._window    = window
        self._timestamps = []

    def update(self):
        now = time.time()
        self._timestamps.append(now)
        if len(self._timestamps) > self._window:
            self._timestamps.pop(0)
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed


# =============================================================================
# SIGNAL HANDLING
# =============================================================================

_pipeline_instance = None


def _signal_handler(signum, frame):
    log.info("Signal %d — shutting down", signum)
    if _pipeline_instance is not None:
        _pipeline_instance.stop()
    sys.exit(0)


# =============================================================================
# MAIN
# =============================================================================

def main():
    global _pipeline_instance

    signal.signal(signal.SIGINT,  _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    pipeline = AdaptivePipeline()
    _pipeline_instance = pipeline

    try:
        pipeline.start()
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
