#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
# collect_training_data.py  —  NeuroEdgeFlow regulator dataset collector
#
# Each cycle it runs the SAME frame through BOTH inference paths
# (local Jetson TensorRT  AND  remote Triton cloud), measures the real
# latency of each, and writes one labelled row to a CSV.
#
# The label `best_mode` is ground truth: whichever path was actually faster
# (and succeeded) on that frame, under those network/hardware conditions.
#
# Run on the Jetson:   python3.8 collect_training_data.py
# Stop with Ctrl+C — the CSV is flushed every row so partial runs are safe.
# ─────────────────────────────────────────────────────────────────────────────
import cv2
import csv
import os
import sys
import time
import socket
import subprocess
import numpy as np

# ─── CONFIG ──────────────────────────────────────────────────────────────────
TRITON_URL   = "10.0.20.10:8001"
SERVER_IP    = "10.0.20.10"
MODEL_NAME   = "yolov5su"
INPUT_NAME   = "images"
OUTPUT_NAME  = "output0"
INPUT_SIZE   = 640

LOCAL_ENGINE = os.path.expanduser("~/yolov5s.onnx_b1_gpu0_fp16.engine")
VIDEO_SOURCE = "/home/nvidia/test_video.mp4"   # or a .jpg / camera index
CSV_OUT      = os.path.expanduser("~/regulator_dataset.csv")

CYCLE_SECONDS = 2.0          # match the engine's poll interval
MAX_SAMPLES   = 2000         # stop automatically after this many rows (0 = unlimited)

import tritonclient.grpc as grpcclient

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

def bandwidth_kbps(host):
    """Rough bandwidth estimate (KB/s) — times a 256 KB transfer."""
    payload = b"X" * (256 * 1024)
    try:
        t0 = time.time()
        with socket.create_connection((host, 8000), timeout=3) as s:
            s.sendall(payload)
        dt = time.time() - t0
        return (len(payload) / 1024.0) / dt if dt > 0 else 0.0
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

# ─── CLOUD INFERENCE ─────────────────────────────────────────────────────────
def cloud_infer(client, tensor):
    """Returns latency in ms, or -1.0 on failure."""
    try:
        inp = grpcclient.InferInput(INPUT_NAME, tensor.shape, "FP32")
        inp.set_data_from_numpy(tensor)
        out = grpcclient.InferRequestedOutput(OUTPUT_NAME)
        t0 = time.time()
        client.infer(model_name=MODEL_NAME, inputs=[inp], outputs=[out],
                     compression_algorithm="gzip")
        return (time.time() - t0) * 1000.0
    except Exception as exc:
        print("[cloud] infer failed: %s" % exc)
        return -1.0

# ─── MAIN COLLECTION LOOP ────────────────────────────────────────────────────
CSV_HEADER = [
    "timestamp", "sample_id",
    # network features
    "rtt_ms", "bandwidth_kbps", "error_rate_pct",
    # hardware features
    "cpu_load", "ram_usage", "gpu_load", "gpu_temp",
    # measured ground-truth latencies
    "local_latency_ms", "cloud_latency_ms",
    # derived labels
    "best_mode", "cloud_faster", "latency_gap_ms",
]

def main():
    # ── frame source ──
    is_image = VIDEO_SOURCE.lower().endswith((".jpg", ".jpeg", ".png"))
    if is_image:
        still = cv2.imread(VIDEO_SOURCE)
        if still is None:
            print("[ERROR] cannot read image %s" % VIDEO_SOURCE)
            sys.exit(1)
        cap = None
    else:
        cap = cv2.VideoCapture(VIDEO_SOURCE)
        if not cap.isOpened():
            print("[ERROR] cannot open video %s" % VIDEO_SOURCE)
            sys.exit(1)
        still = None

    # ── inference backends ──
    local = LocalEngine(LOCAL_ENGINE)
    try:
        client = grpcclient.InferenceServerClient(url=TRITON_URL, verbose=False)
        cloud_ok = client.is_server_ready()
    except Exception:
        cloud_ok = False
    print("[collector] cloud reachable: %s" % cloud_ok)

    # ── csv ──
    new_file = not os.path.isfile(CSV_OUT)
    csv_f = open(CSV_OUT, "a", newline="")
    writer = csv.DictWriter(csv_f, fieldnames=CSV_HEADER)
    if new_file:
        writer.writeheader()

    print("[collector] writing dataset to %s" % CSV_OUT)
    print("[collector] Ctrl+C to stop.\n")

    sample_id = 0
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
            rtt, loss = ping_rtt(SERVER_IP)
            bw         = bandwidth_kbps(SERVER_IP)
            cpu        = cpu_load()
            ram        = ram_usage()
            gload, gt  = gpu_load_temp()

            # ── run BOTH paths on the same frame ──
            local_ms = local.infer(tensor)
            cloud_ms = cloud_infer(client, tensor) if cloud_ok else -1.0

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
                "rtt_ms":           round(rtt, 2),
                "bandwidth_kbps":   round(bw, 2),
                "error_rate_pct":   round(loss, 2),
                "cpu_load":         round(cpu, 2),
                "ram_usage":        round(ram, 2),
                "gpu_load":         round(gload, 2),
                "gpu_temp":         round(gt, 2),
                "local_latency_ms": round(local_ms, 2),
                "cloud_latency_ms": round(cloud_ms, 2),
                "best_mode":        best,
                "cloud_faster":     int(cloud_faster),
                "latency_gap_ms":   round(gap, 2),
            }
            writer.writerow(row)
            csv_f.flush()

            print("sample %4d  rtt=%6.1f  bw=%8.1f  cpu=%5.1f  "
                  "local=%8.1f  cloud=%8.1f  -> %s"
                  % (sample_id, rtt, bw, cpu, local_ms, cloud_ms, best))

            if MAX_SAMPLES and sample_id >= MAX_SAMPLES:
                print("[collector] reached MAX_SAMPLES — stopping.")
                break

            # ── pace the loop ──
            dt = time.time() - cycle_start
            time.sleep(max(0.0, CYCLE_SECONDS - dt))

    except KeyboardInterrupt:
        print("\n[collector] stopped by user.")
    finally:
        csv_f.close()
        if cap:
            cap.release()
        print("[collector] %d samples written to %s" % (sample_id, CSV_OUT))

if __name__ == "__main__":
    main()
