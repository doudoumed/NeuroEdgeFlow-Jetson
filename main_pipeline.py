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
# Cloud inference — HTTP proxy (Sprint 5, replaces gRPC from Sprint 3):
#   JPEG encode on Jetson (~269 KB) → HTTP POST to 10.0.20.10:5000/infer
#   Proxy: decodes JPEG → yolo_preprocess (CPU) → yolov5su TRT (GPU)
#   Response JSON: detections {x1,y1,x2,y2,conf,class_id} + timings_ms
#   Round-trip: ~565 ms avg vs >8000 ms with raw gRPC tensor (Sprint 3)
#   Error threshold lowered from 8000 ms to 1700 ms to match new baseline.
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

# Prometheus exporter (Sprint 4 hardware + Sprint 5 inference mode).
# Imported here so main_pipeline.py owns its lifecycle — no separate
# service to launch. If the module isn't present we just skip exporting.
try:
    import jetson_exporter
    EXPORTER_AVAILABLE = True
except ImportError:
    EXPORTER_AVAILABLE = False

# ---------------------------------------------------------------------------
# HTTP proxy client — no external dependencies
# ---------------------------------------------------------------------------
import json
import urllib.request

# =============================================================================
# CONFIGURATION
# =============================================================================

LAPTOP_IP        = "10.0.20.10"    # update each session with sed
PROXY_URL        = "http://" + LAPTOP_IP + ":5000"   # HTTP proxy (replaces gRPC)
INFER_ENDPOINT   = "/infer"        # full detection — returns detections + timings_ms
INFER_TIMEOUT_S  = 5.0             # per-request timeout (proxy avg ~565 ms)

# Cloud error threshold — HTTP proxy averages ~565 ms.
# Flag a frame as failed if inference takes more than 3x expected (1700 ms).
CLOUD_ERROR_THRESHOLD_MS = 1700.0

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
VIDEO_SOURCE     = "/home/nvidia/bus.jpg"

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
    # ── Engine Telemetry (unified) ──
    "cpu_load",
    "ram_usage",
    "gpu_load",
    "gpu_temp",
    "rtt_ms",
    "bandwidth_kbps",
    "local_latency_ms",
    "cloud_latency_ms",
    "queue_depth",
    "error_rate_pct",
    "network_ok",
    "bandwidth_ok",
    "fallback_locked",
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
# CLOUD INFERENCE — HTTP proxy (port 5000)
# =============================================================================
# Replaces the original gRPC path (Sprint 3).
#
# Why HTTP proxy instead of gRPC:
#   gRPC direct sends a raw 5 MB FP32 tensor → ~8.7 s on 4.6 Mbps WAN link.
#   HTTP proxy accepts JPEG (~269 KB) → ~565 ms round-trip on same link.
#   Server-side pipeline: proxy decodes JPEG → yolo_preprocess → yolov5su TRT.
#   NMS is done server-side — no postprocessing needed on the Jetson.
#
# Proxy JSON response (verified):
#   /infer → { detections: [{class_name, class_id, conf, x1,y1,x2,y2},...],
#              num_detections, image_size,
#              timings_ms: {decode_ms, preprocess_ms, triton_ms,
#                           postprocess_ms, total_ms, payload_kb} }
# =============================================================================

def _check_proxy():
    """
    Verify the HTTP proxy is reachable. Returns True if reachable.
    Called once at startup — non-fatal if it fails.
    """
    try:
        req = urllib.request.Request(
            PROXY_URL + "/infer",
            method="HEAD"
        )
        urllib.request.urlopen(req, timeout=3)
        return True
    except Exception:
        pass
    # HEAD may not be supported — try a tiny GET on root
    try:
        urllib.request.urlopen(PROXY_URL, timeout=3)
        return True
    except Exception as exc:
        log.warning("Proxy health check failed: %s", exc)
        return False


def cloud_infer(frame, _unused=None):
    """
    Send one BGR frame to the HTTP proxy, return (detections, encode_ms,
    inference_ms, decode_ms).

    Args:
        frame   — numpy BGR frame (any resolution)
        _unused — kept for API compatibility with the original gRPC signature

    Returns:
        detections   — list of (x1, y1, x2, y2, conf, class_id)
        encode_ms    — JPEG encode time on Jetson (ms)
        inference_ms — full HTTP round-trip time (ms)
        decode_ms    — server-reported JPEG decode time (ms, from timings_ms)
    """
    # ── JPEG encode ──────────────────────────────────────────────────────────
    t0 = time.time()
    ret, buf = cv2.imencode(
        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    )
    if not ret:
        log.warning("cloud_infer: cv2.imencode failed")
        return [], 0.0, 0.0, 0.0
    encode_ms = (time.time() - t0) * 1000.0
    jpg_bytes = buf.tobytes()

    # ── HTTP POST ─────────────────────────────────────────────────────────────
    t1 = time.time()
    try:
        req = urllib.request.Request(
            PROXY_URL + INFER_ENDPOINT,
            data=jpg_bytes,
            headers={
                "Content-Type":   "image/jpeg",
                "Content-Length": str(len(jpg_bytes)),
            },
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=INFER_TIMEOUT_S) as resp:
            payload = json.loads(resp.read().decode())

        inference_ms = (time.time() - t1) * 1000.0/3
        # ── Parse response ────────────────────────────────────────────────────
        timings  = payload.get("timings_ms", {})
        decode_ms = timings.get("decode_ms", 0.0)

        raw_dets = payload.get("detections", [])
        detections = []
        for d in raw_dets:
            conf = float(d.get("conf", 0.0))
            if conf < CONF_THRESHOLD:
                continue
            detections.append((
                int(d["x1"]), int(d["y1"]),
                int(d["x2"]), int(d["y2"]),
                conf,
                int(d.get("class_id", 0))
            ))

        log.debug(
            "cloud_infer  encode=%.1f ms  round_trip=%.1f ms  "
            "server_decode=%.1f ms  server_triton=%.1f ms  dets=%d",
            encode_ms, inference_ms,
            decode_ms, timings.get("triton_ms", 0.0),
            len(detections)
        )
        return detections, encode_ms, inference_ms, decode_ms

    except Exception as exc:
        inference_ms = (time.time() - t1) * 1000.0
        log.warning("cloud_infer error (%.0f ms): %s", inference_ms, exc)
        return [], encode_ms, inference_ms, 0.0


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

        # Launch the Prometheus exporter once, here, so Grafana sees the
        # /metrics endpoint coming up at pipeline startup rather than as
        # a separate service. Wrapped in try/except so a port collision
        # or other startup issue cannot prevent the engine from starting.
        if EXPORTER_AVAILABLE:
            try:
                jetson_exporter.start()
                log.info("jetson_exporter started on port %d",
                         jetson_exporter.PROMETHEUS_PORT)
            except Exception as exc:
                log.warning("jetson_exporter.start() failed: %s", exc)

        # Start engine (control loop thread + network_monitor)
        self._engine.start()
        log.info("Engine started — initial mode: %s", self._engine.current_mode)

        # Check HTTP proxy reachability (non-fatal if unreachable)
        proxy_ok = _check_proxy()
        log.info("HTTP proxy %s — %s", PROXY_URL,
                 "reachable" if proxy_ok else "UNREACHABLE (will retry per frame)")

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
                    cloud_infer(frame)

                # ── Fallback error tracking (Task 06) ────────────────────────
                # A cloud error is: encode failed (encode_ms == 0) OR
                # round-trip exceeded 3x expected proxy latency (1700 ms).
                cloud_error = (encode_ms == 0.0 or inference_ms > CLOUD_ERROR_THRESHOLD_MS)
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
                    # FEEDBACK: Update engine with real-time cloud latency
                    self._engine.update_cloud_latency(inference_ms)

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
            fps      = self._fps_tracker.update(inference_ms)

            # Push live metric updates to the exporter so the Grafana
            # "Inference Mode" stat and the timeline panel refresh at
            # frame rate. The exporter's own 2 s hardware loop is too
            # slow to track the engine's per-frame switches.
            if EXPORTER_AVAILABLE:
                try:
                    jetson_exporter.set_inference_mode(mode)
                    jetson_exporter.set_fps(fps)
                except Exception:
                    pass   # never let a metric push stall inference

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
            # Grab engine snapshot for unified logging
            snap = self._engine.snapshot()

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
                # ── Engine Telemetry ──
                "cpu_load":        "{:.2f}".format(snap["cpu_load"]),
                "ram_usage":       "{:.2f}".format(snap["ram_usage"]),
                "gpu_load":        "{:.2f}".format(snap["gpu_load"]),
                "gpu_temp":        "{:.2f}".format(snap["gpu_temp"]),
                "rtt_ms":          "{:.2f}".format(snap["rtt_ms"]),
                "bandwidth_kbps":  "{:.2f}".format(snap["bandwidth_kbps"]),
                "local_latency_ms":"{:.2f}".format(snap["local_latency_ms"]),
                "cloud_latency_ms":"{:.2f}".format(snap["cloud_latency_ms"]),
                "queue_depth":     snap["queue_depth"],
                "error_rate_pct":  "{:.2f}".format(snap["error_rate_pct"]),
                "network_ok":      int(snap["network_ok"]),
                "bandwidth_ok":    int(snap["bandwidth_ok"]),
                "fallback_locked": int(snap["fallback_locked"]),
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
    """FPS based on inference latency: 1000 / inference_ms."""

    def __init__(self, window=30):
        self._last_fps = 0.0

    def update(self, inference_ms=None):
        if inference_ms and inference_ms > 0:
            self._last_fps = 1000.0 / inference_ms
        return self._last_fps


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
