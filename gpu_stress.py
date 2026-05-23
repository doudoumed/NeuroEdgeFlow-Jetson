#!/usr/bin/env python3
# =============================================================================
# gpu_stress.py — NeuroEdgeFlow GPU Load Generator
#
# Saturates the Jetson GPU (Maxwell/Pascal) with continuous CUDA work to
# simulate heavy GPU contention scenarios for data collection.
#
# Three backends are tried in order:
#   1. PyCUDA  — raw CUDA kernel (best control)
#   2. OpenCV  — cv2.cuda GpuMat operations (usually available on Jetson)
#   3. NumPy   — CPU fallback with large matrices (worst case)
#
# Python 3.6 compatible — Jetson TX2, JetPack R32.7.6
# Run:  OPENBLAS_CORETYPE=ARMV8 python3 ~/gpu_stress.py
#       OPENBLAS_CORETYPE=ARMV8 python3 ~/gpu_stress.py --duration 120
# =============================================================================

from __future__ import print_function

import argparse
import signal
import sys
import time

# ─── Graceful shutdown ───────────────────────────────────────────────────────
_running = True

def _signal_handler(sig, frame):
    global _running
    _running = False
    print("\n[!] Caught signal — shutting down GPU stress...")

signal.signal(signal.SIGINT,  _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# =============================================================================
# BACKEND 1 — PyCUDA (preferred)
# =============================================================================
def _stress_pycuda(duration):
    import pycuda.autoinit          # noqa: F401
    import pycuda.gpuarray as gpuarray
    import numpy as np

    SIZE = 2048   # 2048×2048 FP32 → ~16 MB per matrix
    print("[PyCUDA] Allocating 2 × {}×{} FP32 matrices on GPU...".format(SIZE, SIZE))

    a_gpu = gpuarray.to_gpu(np.random.randn(SIZE, SIZE).astype(np.float32))
    b_gpu = gpuarray.to_gpu(np.random.randn(SIZE, SIZE).astype(np.float32))

    # Use skcuda's cublasSgemm for heavy GEMM workload
    try:
        import skcuda.linalg as linalg
        linalg.init()
        gemm = lambda: linalg.dot(a_gpu, b_gpu)
        print("[PyCUDA] Using skcuda.linalg.dot (cuBLAS SGEMM)")
    except ImportError:
        # Fallback: element-wise ops in a loop (still GPU-bound)
        gemm = lambda: a_gpu * b_gpu + a_gpu
        print("[PyCUDA] skcuda not found — using element-wise GPU ops")

    t0 = time.time()
    iters = 0
    while _running:
        if duration and (time.time() - t0) >= duration:
            break
        gemm()
        iters += 1
        if iters % 50 == 0:
            elapsed = time.time() - t0
            print("  [PyCUDA] {:,} iterations  |  {:.1f}s elapsed  |  {:.1f} iter/s".format(
                iters, elapsed, iters / elapsed))

    elapsed = time.time() - t0
    print("\n[PyCUDA] Done — {:,} iterations in {:.1f}s ({:.1f} iter/s)".format(
        iters, elapsed, iters / max(elapsed, 0.001)))


# =============================================================================
# BACKEND 2 — OpenCV CUDA (usually available on JetPack)
# =============================================================================
def _stress_opencv_cuda(duration):
    import cv2
    import numpy as np

    SIZE = 2048
    print("[OpenCV CUDA] Allocating {}×{} GpuMat matrices...".format(SIZE, SIZE))

    a_np = np.random.randn(SIZE, SIZE).astype(np.float32)
    b_np = np.random.randn(SIZE, SIZE).astype(np.float32)

    a_gpu = cv2.cuda_GpuMat()
    b_gpu = cv2.cuda_GpuMat()
    a_gpu.upload(a_np)
    b_gpu.upload(b_np)

    t0 = time.time()
    iters = 0
    while _running:
        if duration and (time.time() - t0) >= duration:
            break
        # Heavy GPU ops: multiply + add + gemm
        cv2.cuda.multiply(a_gpu, b_gpu)
        cv2.cuda.add(a_gpu, b_gpu)
        cv2.cuda.gemm(a_gpu, b_gpu, 1.0, None, 0.0)
        iters += 1
        if iters % 50 == 0:
            elapsed = time.time() - t0
            print("  [OpenCV CUDA] {:,} iterations  |  {:.1f}s elapsed  |  {:.1f} iter/s".format(
                iters, elapsed, iters / elapsed))

    elapsed = time.time() - t0
    print("\n[OpenCV CUDA] Done — {:,} iterations in {:.1f}s ({:.1f} iter/s)".format(
        iters, elapsed, iters / max(elapsed, 0.001)))


# =============================================================================
# BACKEND 3 — NumPy CPU fallback
# =============================================================================
def _stress_numpy(duration):
    import numpy as np

    SIZE = 2000
    print("[NumPy CPU] Allocating {}×{} matrices (CPU fallback)...".format(SIZE, SIZE))
    a = np.random.rand(SIZE, SIZE).astype(np.float32)
    b = np.random.rand(SIZE, SIZE).astype(np.float32)

    t0 = time.time()
    iters = 0
    while _running:
        if duration and (time.time() - t0) >= duration:
            break
        np.dot(a, b)
        iters += 1
        if iters % 10 == 0:
            elapsed = time.time() - t0
            print("  [NumPy CPU] {:,} iterations  |  {:.1f}s elapsed  |  {:.1f} iter/s".format(
                iters, elapsed, iters / elapsed))
        time.sleep(0.01)   # prevent full kernel lockup

    elapsed = time.time() - t0
    print("\n[NumPy CPU] Done — {:,} iterations in {:.1f}s ({:.1f} iter/s)".format(
        iters, elapsed, iters / max(elapsed, 0.001)))


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="NeuroEdgeFlow GPU Stress Generator — saturates the Jetson GPU"
    )
    parser.add_argument(
        "--duration", "-d", type=int, default=0,
        help="Run for N seconds then stop (0 = run until Ctrl+C)"
    )
    parser.add_argument(
        "--backend", "-b", choices=["pycuda", "opencv", "numpy", "auto"],
        default="auto",
        help="Force a specific backend (default: auto-detect)"
    )
    args = parser.parse_args()
    dur = args.duration if args.duration > 0 else None

    print("==========================================")
    print("   NEUROEDGEFLOW GPU STRESS GENERATOR     ")
    print("==========================================")
    if dur:
        print("Duration : {} seconds".format(dur))
    else:
        print("Duration : until Ctrl+C")
    print("Backend  : {}".format(args.backend))
    print("------------------------------------------")

    if args.backend in ("pycuda", "auto"):
        try:
            _stress_pycuda(dur)
            return
        except Exception as e:
            if args.backend == "pycuda":
                print("[ERROR] PyCUDA failed: {}".format(e))
                sys.exit(1)
            print("[INFO] PyCUDA not available: {} — trying OpenCV CUDA...".format(e))

    if args.backend in ("opencv", "auto"):
        try:
            import cv2
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                _stress_opencv_cuda(dur)
                return
            else:
                raise RuntimeError("No CUDA devices found by OpenCV")
        except Exception as e:
            if args.backend == "opencv":
                print("[ERROR] OpenCV CUDA failed: {}".format(e))
                sys.exit(1)
            print("[INFO] OpenCV CUDA not available: {} — falling back to NumPy CPU...".format(e))

    # Final fallback
    _stress_numpy(dur)


if __name__ == "__main__":
    main()
