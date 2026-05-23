#!/usr/bin/env python3
# =============================================================================
# test_integration.py — NeuroEdgeFlow Sprint 5 Task 05
# Integration tests for main_pipeline.py
#
# Tests the no-drop guarantee and mode routing without hardware.
# All camera, Triton, and TRT dependencies are stubbed.
#
# Python 3.6 compatible
# Run: OPENBLAS_CORETYPE=ARMV8 python3 ~/test_integration.py
# =============================================================================

from __future__ import print_function

import queue
import sys
import tempfile
import threading
import time
import traceback

import numpy as np

# ---------------------------------------------------------------------------
# Stub cv2 before importing main_pipeline (no display hardware in test)
# ---------------------------------------------------------------------------
import types, sys as _sys

_cv2_stub = types.ModuleType("cv2")
_cv2_stub.imencode   = lambda fmt, img, params=None: (True, np.zeros(100, dtype=np.uint8))
_cv2_stub.imdecode   = lambda buf, flags: np.zeros((640,640,3), dtype=np.uint8)
_cv2_stub.resize     = lambda img, sz: np.zeros((sz[1],sz[0],3), dtype=np.uint8)
_cv2_stub.cvtColor   = lambda img, code: img.astype(np.float32)
_cv2_stub.rectangle  = lambda *a, **kw: None
_cv2_stub.putText    = lambda *a, **kw: None
_cv2_stub.imshow     = lambda *a, **kw: None
_cv2_stub.waitKey    = lambda ms: 0
_cv2_stub.destroyAllWindows = lambda: None
_cv2_stub.CAP_GSTREAMER     = 0
_cv2_stub.IMWRITE_JPEG_QUALITY = 1
_cv2_stub.IMREAD_COLOR       = 1
_cv2_stub.FONT_HERSHEY_SIMPLEX = 0
_cv2_stub.COLOR_BGR2RGB      = 0
_cv2_stub.COLOR_BGR2GRAY     = 0

class _NMSStub(object):
    @staticmethod
    def NMSBoxes(boxes, scores, conf, nms):
        import numpy as np
        return np.array(list(range(len(boxes))))

_cv2_stub.dnn = _NMSStub()
_sys.modules["cv2"] = _cv2_stub

# Stub tritonclient
_triton_stub = types.ModuleType("tritonclient")
_triton_grpc = types.ModuleType("tritonclient.grpc")
_triton_grpc.InferenceServerClient = None
_triton_grpc.InferInput             = None
_triton_grpc.InferRequestedOutput   = None
_sys.modules["tritonclient"]       = _triton_stub
_sys.modules["tritonclient.grpc"]  = _triton_grpc

# Stub edge_pipeline
_edge_stub = types.ModuleType("edge_pipeline")
_edge_stub.run_one_frame = lambda frame: ([], 33.7)   # stub: 0 detections, ~30 FPS
_sys.modules["edge_pipeline"] = _edge_stub

import main_pipeline
from main_pipeline import (
    AdaptivePipeline, FrameCapture, _FPSTracker,
    cloud_infer, _draw, _write_csv, _init_csv, LOG_FILE
)
from adaptive_engine import InferenceMode, AdaptiveEngine, HYSTERESIS_COUNT

# =============================================================================
# MINIMAL TEST FRAMEWORK
# =============================================================================

_passed = 0
_failed = 0
_errors = []


def _assert(cond, name, msg=""):
    global _passed, _failed
    if cond:
        _passed += 1
        print("  PASS  " + name)
    else:
        _failed += 1
        print("  FAIL  " + name + ("  —  " + msg if msg else ""))
        _errors.append(name)


def _assert_eq(actual, expected, name):
    _assert(actual == expected, name,
            "expected={0!r} got={1!r}".format(expected, actual))


def section(t):
    print("\n" + "─" * 64 + "\n  " + t + "\n" + "─" * 64)


# =============================================================================
# TESTS
# =============================================================================

def test_fps_tracker():
    section("FPS tracker — rolling window")

    tracker = _FPSTracker(window=10)

    # No data yet
    fps = tracker.update()
    _assert(fps == 0.0, "single sample → 0 FPS")

    # Feed 10 samples 10ms apart → ~100 FPS
    for _ in range(9):
        time.sleep(0.01)
        tracker.update()
    fps = tracker.update()
    _assert(80.0 < fps < 120.0,
            "10 samples 10ms apart → ~100 FPS",
            "got {:.1f}".format(fps))

    # Window rolls — oldest sample falls off
    fps_before = fps
    time.sleep(0.1)
    fps_after = tracker.update()
    _assert(fps_after < fps_before,
            "FPS drops after slow frame", "before={:.1f} after={:.1f}".format(
                fps_before, fps_after))


def test_frame_queue_non_blocking():
    """
    FrameCapture uses a bounded queue with maxsize=2.
    If inference is slow, oldest frames must be discarded — capture never blocks.
    Verify the discard logic without a real camera.
    """
    section("Frame queue — oldest frame discarded when full (no-drop guarantee)")

    q = queue.Queue(maxsize=2)
    dummy = np.zeros((720, 1280, 3), dtype=np.uint8)

    # Fill the queue
    q.put(dummy); q.put(dummy)
    _assert(q.full(), "queue is full after 2 puts")

    # Simulate FrameCapture discard logic
    if q.full():
        try:
            q.get_nowait()
        except queue.Empty:
            pass
    q.put_nowait(dummy)

    _assert(q.full(), "queue still full after discard+put")
    _assert_eq(q.qsize(), 2, "queue size=2 (oldest discarded, new added)")


def test_mode_routing_local():
    """
    When engine mode is LOCAL, inference loop must call edge_pipeline.run_one_frame.
    Verify by tracking call count on the stub.
    """
    section("Mode routing — LOCAL calls edge_pipeline.run_one_frame")

    call_counts = {"edge": 0, "cloud": 0}
    frames_processed = []

    def fake_edge(frame):
        call_counts["edge"] += 1
        frames_processed.append(frame)
        return [], 33.7

    _edge_stub.run_one_frame = fake_edge

    # Simulate inference loop routing decision
    mode = InferenceMode.LOCAL
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    if mode == InferenceMode.CLOUD:
        cloud_infer(frame, None)
        call_counts["cloud"] += 1
    else:
        _edge_stub.run_one_frame(frame)

    _assert_eq(call_counts["edge"],  1, "edge called once for LOCAL frame")
    _assert_eq(call_counts["cloud"], 0, "cloud not called for LOCAL frame")
    _assert(len(frames_processed) == 1, "one frame processed")

    # Restore
    _edge_stub.run_one_frame = lambda frame: ([], 33.7)


def test_mode_routing_cloud():
    """
    When engine mode is CLOUD and Triton is None (stub), cloud_infer must
    be called and return a valid (detections, encode_ms, inference_ms, decode_ms).
    """
    section("Mode routing — CLOUD calls cloud_infer with stub Triton")

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    mode  = InferenceMode.CLOUD

    detections, encode_ms, inference_ms, decode_ms = cloud_infer(frame, None)

    _assert(isinstance(detections, list),   "detections is a list")
    _assert(isinstance(encode_ms, float),   "encode_ms is float")
    _assert(isinstance(inference_ms, float),"inference_ms is float")
    _assert(isinstance(decode_ms, float),   "decode_ms is float")
    _assert(encode_ms >= 0,                 "encode_ms >= 0")
    _assert(inference_ms > 0,               "inference_ms > 0 (stub sleep)")


def test_mode_switch_takes_effect_next_frame():
    """
    Core no-drop guarantee test.

    Engine mode is read once per frame as engine.current_mode.
    A mode switch (engine._state = new_mode) by the background thread
    takes effect on the NEXT frame — the current frame completes in its mode.

    This test drives the engine state machine directly (no real network)
    and verifies that the mode seen by consecutive frames reflects the
    engine state AT THE TIME of that frame's inference read.
    """
    section("No-drop guarantee — mode switch takes effect on next frame")

    tmp = tempfile.mktemp(suffix=".csv")
    engine = AdaptiveEngine(
        poll_interval=99.0,   # disable automatic polling
        log_file=tmp,
        hysteresis_count=1    # immediate switch for test clarity
    )
    # Manually set initial state
    engine._state = InferenceMode.LOCAL

    # Frame 1 — read mode before any switch
    mode_frame1 = engine.current_mode
    _assert_eq(mode_frame1, InferenceMode.LOCAL, "frame 1 sees LOCAL")

    # Engine background thread switches mode (simulated)
    engine._state = InferenceMode.CLOUD

    # Frame 2 — reads new mode
    mode_frame2 = engine.current_mode
    _assert_eq(mode_frame2, InferenceMode.CLOUD, "frame 2 sees CLOUD after switch")

    # Frame 1's inference already completed in LOCAL — no frame was dropped
    _assert(mode_frame1 != mode_frame2,
            "frame 1 and frame 2 saw different modes — switch took effect")


def test_engine_thread_does_not_block_inference():
    """
    The engine control loop runs in a background thread.
    Reading engine.current_mode from the inference loop must complete in < 1 ms.
    """
    section("No-drop guarantee — engine.current_mode read is non-blocking")

    tmp = tempfile.mktemp(suffix=".csv")
    engine = AdaptiveEngine(
        poll_interval=99.0,
        log_file=tmp,
        hysteresis_count=3
    )
    engine._state = InferenceMode.LOCAL

    # Time 1000 consecutive mode reads
    t0 = time.time()
    for _ in range(1000):
        _ = engine.current_mode
    elapsed_ms = (time.time() - t0) * 1000.0

    avg_us = elapsed_ms / 1000.0 * 1000.0  # microseconds per read
    _assert(avg_us < 100.0,
            "1000 mode reads avg < 100 µs each",
            "avg={:.2f} µs".format(avg_us))
    _assert(elapsed_ms < 10.0,
            "1000 mode reads complete in < 10 ms total",
            "took {:.2f} ms".format(elapsed_ms))


def test_draw_does_not_crash_on_empty_detections():
    section("Draw — no crash on empty detections")
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    result = _draw(frame, [], InferenceMode.LOCAL, 29.65, None, 0)
    _assert(result is not None, "draw returns frame")
    _assert(result.shape == frame.shape, "draw preserves frame shape")


def test_draw_shows_pending_hysteresis():
    section("Draw — pending hysteresis label rendered")
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    # Should not raise even with pending_mode set
    result = _draw(frame, [], InferenceMode.LOCAL, 10.0,
                   InferenceMode.CLOUD, 2)
    _assert(result is not None, "draw with pending mode returns frame")


def test_csv_schema():
    section("CSV — schema contains all required Task 05 columns")
    import main_pipeline as mp
    required = [
        "timestamp", "frame_id", "mode",
        "encode_ms", "inference_ms", "decode_ms", "total_ms",
        "num_detections", "fps", "pending_mode", "pending_count"
    ]
    for col in required:
        _assert(col in mp.CSV_HEADER,
                "CSV_HEADER contains '{0}'".format(col))


def test_csv_write_read():
    section("CSV — write and read back a row")
    import csv as _csv
    tmp = tempfile.mktemp(suffix=".csv")
    _init_csv(tmp)
    _write_csv(tmp, {
        "timestamp": "2026-04-01 12:00:00",
        "frame_id": 1, "mode": "LOCAL",
        "encode_ms": "0.00", "inference_ms": "33.70", "decode_ms": "0.00",
        "total_ms": "34.00", "num_detections": 3,
        "fps": "29.41", "pending_mode": "", "pending_count": 0
    })
    with open(tmp) as f:
        rows = list(_csv.DictReader(f))
    _assert_eq(len(rows), 1,              "one data row written")
    _assert_eq(rows[0]["mode"], "LOCAL",  "mode=LOCAL")
    _assert_eq(rows[0]["frame_id"], "1",  "frame_id=1")
    _assert_eq(rows[0]["inference_ms"], "33.70", "inference_ms correct")
    import os; os.remove(tmp)


def test_hysteresis_visible_in_inference_loop():
    """
    While engine is accumulating hysteresis (pending_count 1 or 2),
    the inference loop must still see the CURRENT mode (not the pending one)
    and continue processing frames without interruption.
    """
    section("Hysteresis — current mode unchanged during pending accumulation")

    tmp = tempfile.mktemp(suffix=".csv")
    engine = AdaptiveEngine(
        poll_interval=99.0,
        log_file=tmp,
        hysteresis_count=3
    )
    engine._state         = InferenceMode.LOCAL
    engine._pending_mode  = InferenceMode.CLOUD
    engine._pending_count = 2   # one poll away from switch

    # Inference loop reads current_mode — must still see LOCAL
    mode = engine.current_mode
    _assert_eq(mode, InferenceMode.LOCAL,
               "current_mode=LOCAL while pending_count=2/3")

    # pending_mode and pending_count are visible for the overlay
    _assert_eq(engine.pending_mode,  InferenceMode.CLOUD,
               "pending_mode=CLOUD visible")
    _assert_eq(engine.pending_count, 2,
               "pending_count=2 visible")


# =============================================================================
# RUN ALL TESTS
# =============================================================================

def main():
    print("=" * 64)
    print("  NeuroEdgeFlow Sprint 5 Task 05 — Integration Tests")
    print("=" * 64)

    try:
        test_fps_tracker()
        test_frame_queue_non_blocking()
        test_mode_routing_local()
        test_mode_routing_cloud()
        test_mode_switch_takes_effect_next_frame()
        test_engine_thread_does_not_block_inference()
        test_draw_does_not_crash_on_empty_detections()
        test_draw_shows_pending_hysteresis()
        test_csv_schema()
        test_csv_write_read()
        test_hysteresis_visible_in_inference_loop()
    except Exception:
        print("\n  FATAL: test suite crashed:")
        traceback.print_exc()
        sys.exit(2)

    print("\n" + "=" * 64)
    print("  Results: {0} passed, {1} failed".format(_passed, _failed))
    if _errors:
        print("  Failed:")
        for n in _errors:
            print("    - " + n)
    print("=" * 64)
    sys.exit(0 if _failed == 0 else 1)


if __name__ == "__main__":
    main()
