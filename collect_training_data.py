#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
# collect_training_data.py  —  NeuroEdgeFlow regulator dataset collector
#
# Each cycle it runs the SAME frame through BOTH inference paths
# (local Jetson TensorRT  AND  remote Triton cloud via JPEG proxy),
# measures the real latency of each, and writes one labelled row to CSV.
#
# The label `best_mode` is ground truth: whichever path was actually faster
# (and succeeded) on that frame, under those network/hardware conditions.
#
# Two architecture changes vs the previous version:
#   1. Cloud inference goes through the Flask JPEG proxy
#      (cloud_infer_jpeg from cloud_infer_jpeg_v2.py), not direct gRPC.
#      The proxy decodes the JPEG and handles all the Triton plumbing,
#      so the Jetson only pays for JPEG encoding (~10 ms) plus the network
#      round-trip — typically ~10x faster than shipping a 4.9 MB float32
#      tensor over gRPC.
#   2. FPS is recorded each cycle using a rolling 30-sample tracker, so
#      the RF training set can learn the relationship between latency and
#      effective throughput.
#
# Run on the Jetson:   python3 collect_training_data.py
# Stop with Ctrl+C — the CSV is flushed every row so partial runs are safe.
# ─────────────────────────────────────────────────────────────────────────────
import argparse
import csv
import os
import sys
import time
import subprocess
import collections
import http.client

import cv2
import numpy as np

# ─── External cloud client (replaces direct gRPC) ───────────────────────────
# cloud_infer_jpeg() handles: JPEG encode -> HTTP POST -> proxy decodes ->
# Triton infers -> response back. Returns round-trip latency in ms, or
# -1.0 on failure.
try:
    from cloud_infer_jpeg_v2 import cloud_infer_jpeg, test_connection
except ImportError:
    print("[ERROR] cloud_infer_jpeg_v2.py not found in the current directory.")
    print("        Place it next to this script and try again.")
    sys.exit(1)

# ─── CONFIG ──────────────────────────────────────────────────────────────────
SERVER_IP    = "10.0.20.10"        # Flask proxy host (same as Triton host)
PROXY_PORT   = 5000                # Flask proxy port (NOT Triton's 8001)
INPUT_SIZE   = 640

LOCAL_ENGINE = os.path.expanduser("~/yolov5s.onnx_b1_gpu0_fp16.engine")
VIDEO_SOURCE = "/home/nvidia/bus.jpg"   # or a .jpg / camera index
CSV_OUT      = os.path.expanduser("~/regulator_dataset.csv")

CYCLE_SECONDS = 2.0          # match the engine's poll interval
MAX_SAMPLES   = 2000         # stop automatically after this many rows (0 = unlimited)

# FPS tracker window: how many recent cycles to average over. A small
# window (5-10) reacts fast to changes; a larger one is more stable.
# 10 lines up with the engine's hysteresis window so the FPS column in
# the CSV reflects roughly the same time horizon the regulator sees.
FPS_WINDOW = 10

# ─── HARDWARE METRICS ────────────────────────────────────────────────────────
def cpu_load():
    """Instantaneous CPU utilisation %, averaged over a short window."""
    try:
        with open("/proc/stat") as f:
            p1 = [float(x) for x in f.readline().split()[1:]]
        time.sleep(0.1)
        with open("/proc/stat") as f:
            p2 = [float(x) for x in f.readline().split()[1:]]
        idle = p2[3] - p1[3]
        total = sum(p2) - sum(p1)
        return 100.0 * (1.0 - idle / total) if total > 0 else 0.0
    except Exception:
        return 0.0

def ram_usage():
    """RAM utilisation %."""
    try:
        meminfo = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":")
                meminfo[k.strip()] = float(v.strip().split()[0])
        total = meminfo["MemTotal"]
        avail = meminfo.get("MemAvailable", meminfo["MemFree"])
        return 100.0 * (1.0 - avail / total) if total > 0 else 0.0
    except Exception:
        return 0.0

def gpu_load_temp():
    """Jetson GPU load % and temperature C via tegrastats-style sysfs."""
    load, temp = 0.0, 0.0
    try:
        for base in ("/sys/devices/gpu.0/load",
                     "/sys/devices/platform/gpu.0/load"):
            if os.path.exists(base):
                with open(base) as f:
                    load = float(f.read().strip()) / 10.0
                break
    except Exception:
        pass
    try:
        for z in ("/sys/devices/virtual/thermal/thermal_zone1/temp",
                  "/sys/class/thermal/thermal_zone1/temp"):
            if os.path.exists(z):
                with open(z) as f:
                    temp = float(f.read().strip()) / 1000.0
                break
    except Exception:
        pass
    return load, temp

# ─── NETWORK METRICS ─────────────────────────────────────────────────────────
def ping_rtt(host, count=3):
    """Average RTT (ms) and packet-loss (%)."""
    try:
        r = subprocess.run(["ping", "-c", str(count), "-W", "1", host],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           timeout=8)
        out = r.stdout.decode("utf-8")
        rtt, loss = 0.0, 0.0
        for line in out.splitlines():
            if "packet loss" in line:
                for tok in line.split(","):
                    if "packet loss" in tok:
                        loss = float(tok.strip().split("%")[0])
            if "min/avg/max" in line:
                rtt = float(line.split("=")[1].strip().split("/")[1])
        return rtt, loss
    except Exception:
        return 0.0, 100.0

def bandwidth_kbps(host, port=8000):
    """
    Bandwidth estimate (KB/s) using a real HTTP POST round-trip to Triton.

    The old version did `sendall()` to port 8000 and timed the call return,
    but Triton would slam the connection shut before any meaningful bytes
    flowed — so the timer almost always came back as ~0ms and the function
    reported 0 KB/s. We now POST 256 KB to Triton's /v2/health/ready and
    wait for the response, which forces a full round-trip.
    """
    payload_size = 256 * 1024
    payload = b"X" * payload_size
    try:
        t0 = time.time()
        conn = http.client.HTTPConnection(host, port, timeout=3)
        conn.request(
            "POST", "/v2/health/ready",
            body=payload,
            headers={
                "Content-Type": "application/octet-stream",
                "Content-Length": str(payload_size),
            },
        )
        resp = conn.getresponse()
        resp.read()
        conn.close()
        dt = time.time() - t0
        return (payload_size / 1024.0) / dt if dt > 0 else 0.0
    except Exception:
        return 0.0

# ─── PREPROCESS ──────────────────────────────────────────────────────────────
def preprocess(frame):
    img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    img = img[:, :, ::-1].astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return np.ascontiguousarray(np.expand_dims(img, axis=0))

# ─── LOCAL TENSORRT INFERENCE ────────────────────────────────────────────────
class LocalEngine:
    """Loads the Jetson TensorRT engine. Returns (latency_ms, ok)."""
    def __init__(self, path):
        self.ok = False
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit  # noqa
            self.cuda = cuda
            logger = trt.Logger(trt.Logger.WARNING)
            with open(path, "rb") as f, trt.Runtime(logger) as rt:
                self.engine = rt.deserialize_cuda_engine(f.read())
            self.ctx = self.engine.create_execution_context()
            self.stream = cuda.Stream()
            self.ok = True
            print("[LocalEngine] TensorRT engine loaded.")
        except Exception as exc:
            print("[LocalEngine] unavailable (%s) — local timing will be skipped"
                  % exc)

    def infer(self, tensor):
        """Returns latency in ms, or -1.0 on failure."""
        if not self.ok:
            return -1.0
        try:
            t0 = time.time()
            # Minimal execution path — enough to time GPU inference.
            d_in = self.cuda.mem_alloc(tensor.nbytes)
            out_shape = (1, 84, 8400)
            out = np.empty(out_shape, dtype=np.float32)
            d_out = self.cuda.mem_alloc(out.nbytes)
            self.cuda.memcpy_htod_async(d_in, tensor, self.stream)
            self.ctx.execute_async_v2([int(d_in), int(d_out)],
                                      self.stream.handle)
            self.cuda.memcpy_dtoh_async(out, d_out, self.stream)
            self.stream.synchronize()
            return (time.time() - t0) * 1000.0
        except Exception as exc:
            print("[LocalEngine] infer failed: %s" % exc)
            return -1.0

# ─── FPS TRACKER ─────────────────────────────────────────────────────────────
class FPSTracker:
    """
    Rolling-window FPS estimator.

    Each cycle we feed it the wall-clock time of one completed sample. It
    keeps the last N timestamps and reports frames-per-second over the
    span between the oldest and the newest. This is the same metric a
    real video pipeline would expose ("how many frames per second is the
    system actually delivering"), as opposed to the per-frame latency
    we already log in local_latency_ms / cloud_latency_ms.

    Why both columns in the CSV?
      - latency_ms answers "how long did ONE frame take?"
      - fps         answers "how many frames per second are we sustaining?"
    Both matter to the RF — high latency with high FPS (pipelined) is
    different from high latency with low FPS (serial blocking).
    """

    def __init__(self, window=FPS_WINDOW):
        self.window = window
        self._stamps = collections.deque(maxlen=window)

    def tick(self):
        """Record one completed sample, return the current FPS estimate."""
        now = time.time()
        self._stamps.append(now)
        if len(self._stamps) < 2:
            return 0.0
        span = self._stamps[-1] - self._stamps[0]
        if span <= 0:
            return 0.0
        # n-1 intervals between n stamps
        return (len(self._stamps) - 1) / span


# ─── CLOUD INFERENCE WRAPPER ─────────────────────────────────────────────────
def cloud_infer(frame):
    """
    Thin wrapper around cloud_infer_jpeg() so the main loop stays clean.
    The proxy URL is built from SERVER_IP + PROXY_PORT in CONFIG.

    Returns latency in ms, or -1.0 on failure.
    """
    return cloud_infer_jpeg(SERVER_IP, frame, port=PROXY_PORT)

# ─── MAIN COLLECTION LOOP ────────────────────────────────────────────────────
CSV_HEADER = [
    "timestamp", "sample_id", "scenario",
    # network features
    "rtt_ms", "bandwidth_kbps", "error_rate_pct",
    # hardware features
    "cpu_load", "ram_usage", "gpu_load", "gpu_temp",
    # measured ground-truth latencies
    "local_latency_ms", "cloud_latency_ms",
    # throughput (rolling FPS over the last FPS_WINDOW cycles)
    "fps",
    # derived labels
    "best_mode", "cloud_faster", "latency_gap_ms",
]

def parse_args():
    """Command-line arguments. All optional — defaults match the CONFIG block."""
    p = argparse.ArgumentParser(
        description="Collect dual-inference training data for the regulator. "
                    "Each cycle runs the same frame through both the local "
                    "TensorRT engine and the cloud (via JPEG proxy), then "
                    "labels the row with whichever was faster."
    )
    p.add_argument("--scenario", default="default",
                   help="Scenario label written into every row. Set this when "
                        "running under a specific network/CPU condition so the "
                        "RF training set can be grouped later.")
    p.add_argument("--max-samples", type=int, default=MAX_SAMPLES,
                   help="Stop after this many rows (0 = unlimited).")
    p.add_argument("--duration", type=int, default=0,
                   help="Stop after this many seconds (0 = unlimited).")
    p.add_argument("--server", default=SERVER_IP,
                   help="Flask proxy server IP.")
    p.add_argument("--port", type=int, default=PROXY_PORT,
                   help="Flask proxy server port.")
    p.add_argument("--source", default=VIDEO_SOURCE,
                   help="Video file, image, or camera index.")
    p.add_argument("--output", default=CSV_OUT,
                   help="CSV output path.")
    p.add_argument("--cycle", type=float, default=CYCLE_SECONDS,
                   help="Seconds between samples.")
    return p.parse_args()


def main():
    global SERVER_IP, PROXY_PORT   # used by cloud_infer() wrapper

    args = parse_args()
    SERVER_IP = args.server
    PROXY_PORT = args.port

    # ── banner ──
    print("=" * 70)
    print("  collect_training_data.py — JPEG proxy + FPS tracker")
    print("=" * 70)
    print(f"  Scenario:    {args.scenario}")
    print(f"  Server:      {args.server}:{args.port}")
    print(f"  Source:      {args.source}")
    print(f"  Output CSV:  {args.output}")
    print(f"  Cycle:       {args.cycle}s")
    print(f"  FPS window:  {FPS_WINDOW} samples")
    if args.max_samples > 0:
        print(f"  Max samples: {args.max_samples}")
    if args.duration > 0:
        print(f"  Duration:    {args.duration}s")
    print("=" * 70)

    # ── frame source ──
    is_image = str(args.source).lower().endswith((".jpg", ".jpeg", ".png"))
    if is_image:
        still = cv2.imread(args.source)
        if still is None:
            print(f"[ERROR] cannot read image {args.source}")
            sys.exit(1)
        cap = None
        print(f"[OK] Loaded image: {still.shape[1]}x{still.shape[0]}")
    else:
        cap = cv2.VideoCapture(args.source)
        if not cap.isOpened():
            print(f"[ERROR] cannot open video {args.source}")
            sys.exit(1)
        still = None
        print("[OK] Opened video source")

    # ── inference backends ──
    local = LocalEngine(LOCAL_ENGINE)
    # The cloud is now an HTTP proxy. Verify it once up front; if it isn't
    # reachable we still run (the collector will simply log cloud=-1.0
    # until the proxy comes back).
    print("\nChecking proxy server connection...")
    cloud_ok = test_connection(args.server, args.port)
    if not cloud_ok:
        print("[WARN] proxy not reachable — cloud_latency_ms will be -1.0")

    # ── FPS tracker (rolling-window throughput estimator) ──
    fps_tracker = FPSTracker(window=FPS_WINDOW)

    # ── csv ──
    new_file = not os.path.isfile(args.output)
    csv_f = open(args.output, "a", newline="")
    writer = csv.DictWriter(csv_f, fieldnames=CSV_HEADER)
    if new_file:
        writer.writeheader()

    print(f"[collector] writing dataset to {args.output}")
    print("[collector] Ctrl+C to stop.\n")

    sample_id = 0
    start_time = time.time()

    try:
        while True:
            cycle_start = time.time()
            sample_id += 1

            # ── grab a frame ──
            if is_image:
                frame = still.copy()
            else:
                ok, frame = cap.read()
                if not ok:                       # loop video
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = cap.read()
                    if not ok:
                        break
            tensor = preprocess(frame)

            # ── measure conditions ──
            rtt, loss = ping_rtt(args.server)
            bw         = bandwidth_kbps(args.server)
            cpu        = cpu_load()
            ram        = ram_usage()
            gload, gt  = gpu_load_temp()

            # ── run BOTH paths on the same frame ──
            # Local: TensorRT path needs the pre-processed tensor.
            # Cloud: JPEG proxy needs the raw BGR frame — the proxy does
            # its own preprocessing on the server side.
            local_ms = local.infer(tensor)
            cloud_ms = cloud_infer(frame) if cloud_ok else -1.0

            # ── update FPS tracker ──
            # tick() records the timestamp of this completed sample and
            # returns the rolling-average FPS. The first sample returns 0
            # (need at least 2 stamps to measure an interval).
            fps = fps_tracker.tick()

            # ── derive label ──
            if local_ms > 0 and cloud_ms > 0:
                cloud_faster = cloud_ms < local_ms
                best = "CLOUD" if cloud_faster else "LOCAL"
                gap  = abs(cloud_ms - local_ms)
            elif local_ms > 0:
                cloud_faster, best, gap = False, "LOCAL", 0.0
            elif cloud_ms > 0:
                cloud_faster, best, gap = True, "CLOUD", 0.0
            else:
                cloud_faster, best, gap = False, "LOCAL", 0.0

            row = {
                "timestamp":        time.strftime("%Y-%m-%d %H:%M:%S"),
                "sample_id":        sample_id,
                "scenario":         args.scenario,
                "rtt_ms":           round(rtt, 2),
                "bandwidth_kbps":   round(bw, 2),
                "error_rate_pct":   round(loss, 2),
                "cpu_load":         round(cpu, 2),
                "ram_usage":        round(ram, 2),
                "gpu_load":         round(gload, 2),
                "gpu_temp":         round(gt, 2),
                "local_latency_ms": round(local_ms, 2),
                "cloud_latency_ms": round(cloud_ms, 2),
                "fps":              round(fps, 2),
                "best_mode":        best,
                "cloud_faster":     int(cloud_faster),
                "latency_gap_ms":   round(gap, 2),
            }
            writer.writerow(row)
            csv_f.flush()

            print(f"sample {sample_id:4d}  [{args.scenario:15s}]  "
                  f"rtt={rtt:6.1f}  bw={bw:8.1f}  cpu={cpu:5.1f}%  "
                  f"local={local_ms:7.1f}  cloud={cloud_ms:7.1f}  "
                  f"fps={fps:5.2f}  -> {best}")

            # ── stop conditions ──
            if args.max_samples > 0 and sample_id >= args.max_samples:
                print(f"[collector] reached max-samples ({args.max_samples}) — stopping.")
                break
            if args.duration > 0 and (time.time() - start_time) >= args.duration:
                print(f"[collector] reached duration ({args.duration}s) — stopping.")
                break

            # ── pace the loop ──
            dt = time.time() - cycle_start
            time.sleep(max(0.0, args.cycle - dt))

    except KeyboardInterrupt:
        print("\n[collector] stopped by user.")
    finally:
        csv_f.close()
        if cap:
            cap.release()
        print(f"[collector] {sample_id} samples written to {args.output}")

if __name__ == "__main__":
    main()
