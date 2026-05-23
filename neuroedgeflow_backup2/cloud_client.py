#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
# always_cloud_runner.py — Always-Cloud baseline for NeuroEdgeFlow
#
# Sends every frame to the cloud (Triton via JPEG proxy). No adaptive logic.
# This is the Sprint-3 baseline used for comparison against the rule-based
# and Random Forest regulators in the dissertation evaluation.
#
# Why this baseline matters:
#   - Establishes "blind offloading" performance — the upper bound on
#     network usage and the lower bound on Jetson load.
#   - Quantifies what happens when network conditions degrade and the
#     system has no fallback.
#   - Provides a reference for the adaptive systems to beat.
#
# Usage:
#   python3 always_cloud_runner.py --scenario ideal --max-samples 150
#   python3 always_cloud_runner.py --scenario high_latency --duration 300
#   python3 always_cloud_runner.py --source bus.jpg --max-samples 50
#
# Stop with Ctrl+C — CSV is flushed every row so partial runs are safe.
# ─────────────────────────────────────────────────────────────────────────────
 
import argparse
import csv
import os
import sys
import time
import subprocess
import socket
 
import cv2
import numpy as np
 
# Import the JPEG-based cloud client
try:
    from cloud_infer_jpeg_v2 import cloud_infer_jpeg, test_connection
except ImportError:
    print("[ERROR] cloud_infer_jpeg_v2.py not found in the current directory.")
    print("        Please place it next to this script.")
    sys.exit(1)
 
 
# ─── CONFIG ──────────────────────────────────────────────────────────────────
SERVER_IP    = "10.0.20.10"             # Triton proxy server IP
PROXY_PORT   = 5000                     # Flask proxy port
VIDEO_SOURCE = "/home/nvidia/test_video.mp4"
CSV_OUT      = os.path.expanduser("~/always_cloud_dataset.csv")
CYCLE_SECONDS = 2.0                     # Time between samples
MAX_SAMPLES  = 150                      # 0 = unlimited
# ─────────────────────────────────────────────────────────────────────────────
 
 
# ─── Hardware metrics (same logic as collect_training_data.py) ──────────────
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
    """Jetson GPU load % and temperature in Celsius."""
    load, temp = 0.0, 0.0
 
    # GPU load
    try:
        for base in ("/sys/devices/gpu.0/load",
                     "/sys/devices/platform/gpu.0/load",
                     "/sys/devices/17000000.gv11b/load",
                     "/sys/devices/57000000.gpu/load"):
            if os.path.exists(base):
                with open(base) as f:
                    load = float(f.read().strip()) / 10.0
                break
    except Exception:
        pass
 
    # GPU temp: match by label, not by index (jetson_exporter.py approach)
    labelled_temp = None
    all_temps = []
    try:
        for i in range(20):
            tz = f"/sys/class/thermal/thermal_zone{i}"
            type_path = tz + "/type"
            temp_path = tz + "/temp"
            if not os.path.exists(temp_path):
                continue
            try:
                with open(temp_path) as f:
                    millideg = float(f.read().strip())
                celsius = millideg / 1000.0
                all_temps.append(celsius)
                if os.path.exists(type_path):
                    with open(type_path) as f:
                        label = f.read().strip().lower()
                    if "gpu" in label:
                        labelled_temp = celsius
            except (IOError, OSError, ValueError):
                continue
    except Exception:
        pass
 
    if labelled_temp is not None:
        temp = labelled_temp
    elif all_temps:
        temp = max(all_temps)
 
    return load, temp
 
 
# ─── Network metrics ─────────────────────────────────────────────────────────
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
    """Bandwidth estimate using HTTP POST round-trip to Triton."""
    import http.client
    payload_size = 256 * 1024
    payload = b"X" * payload_size
    try:
        t0 = time.time()
        conn = http.client.HTTPConnection(host, port, timeout=3)
        conn.request("POST", "/v2/health/ready",
                     body=payload,
                     headers={"Content-Type": "application/octet-stream",
                              "Content-Length": str(payload_size)})
        resp = conn.getresponse()
        resp.read()
        conn.close()
        dt = time.time() - t0
        if dt > 0:
            return (payload_size / 1024.0) / dt
        return 0.0
    except Exception:
        return 0.0
 
 
# ─── CSV schema ──────────────────────────────────────────────────────────────
CSV_HEADER = [
    "timestamp", "frame_id", "scenario",
    "mode",                            # Always "CLOUD" for this baseline
    "cloud_latency_ms",                # Round-trip including network
    "cloud_ok",                        # 1 if cloud succeeded, 0 if failed
    # Network state
    "rtt_ms", "bandwidth_kbps", "error_rate_pct",
    # Hardware state
    "cpu_load", "ram_usage", "gpu_load", "gpu_temp",
    # Derived
    "successful_inferences",           # Running total of successful inferences
    "failed_inferences",               # Running total of failures
]
 
 
# ─── Main loop ───────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Always-Cloud baseline: sends every frame to the cloud "
                    "without any adaptive logic. Used for comparison with "
                    "rule-based and RF-based offloading."
    )
    p.add_argument("--scenario", default="ideal",
                   help="Scenario label written to every CSV row. Apply the "
                        "matching tc/netem rule before starting.")
    p.add_argument("--max-samples", type=int, default=MAX_SAMPLES,
                   help="Stop after this many frames (0 = unlimited).")
    p.add_argument("--duration", type=int, default=0,
                   help="Stop after this many seconds (0 = unlimited).")
    p.add_argument("--server", default=SERVER_IP,
                   help="Triton proxy server IP.")
    p.add_argument("--port", type=int, default=PROXY_PORT,
                   help="Triton proxy server port.")
    p.add_argument("--source", default=VIDEO_SOURCE,
                   help="Video file, image, or camera index.")
    p.add_argument("--output", default=CSV_OUT,
                   help="CSV output path.")
    p.add_argument("--cycle", type=float, default=CYCLE_SECONDS,
                   help="Seconds between samples.")
    return p.parse_args()
 
 
def main():
    args = parse_args()
 
    print("=" * 70)
    print("  Always-Cloud Baseline Runner")
    print("=" * 70)
    print(f"  Scenario:    {args.scenario}")
    print(f"  Server:      {args.server}:{args.port}")
    print(f"  Source:      {args.source}")
    print(f"  Output CSV:  {args.output}")
    print(f"  Cycle:       {args.cycle}s")
    if args.max_samples > 0:
        print(f"  Max samples: {args.max_samples}")
    if args.duration > 0:
        print(f"  Duration:    {args.duration}s")
    print("=" * 70)
 
    # Verify connection to proxy server before starting
    print("\nChecking proxy server connection...")
    if not test_connection(args.server, args.port):
        print("[ERROR] Cannot reach proxy server. Aborting.")
        sys.exit(1)
 
    # Open video source
    is_image = str(args.source).lower().endswith((".jpg", ".jpeg", ".png"))
    cap = None
    still = None
 
    if is_image:
        still = cv2.imread(args.source)
        if still is None:
            print(f"[ERROR] Cannot read image: {args.source}")
            sys.exit(1)
        print(f"[OK] Loaded image: {still.shape[1]}x{still.shape[0]}")
    else:
        cap = cv2.VideoCapture(args.source)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video: {args.source}")
            sys.exit(1)
        print(f"[OK] Opened video source")
 
    # Open CSV
    new_file = not os.path.isfile(args.output)
    csv_f = open(args.output, "a", newline="")
    writer = csv.DictWriter(csv_f, fieldnames=CSV_HEADER)
    if new_file:
        writer.writeheader()
    print(f"[OK] Writing to {args.output}")
    print("[INFO] Press Ctrl+C to stop.\n")
 
    frame_id = 0
    success_count = 0
    fail_count = 0
    start_time = time.time()
 
    try:
        while True:
            cycle_start = time.time()
            frame_id += 1
 
            # Grab frame
            if is_image:
                frame = still.copy()
            else:
                ok, frame = cap.read()
                if not ok:
                    # Loop video
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = cap.read()
                    if not ok:
                        print("[ERROR] Cannot read frame, stopping.")
                        break
 
            # Measure network conditions
            rtt, loss = ping_rtt(args.server)
            bw = bandwidth_kbps(args.server)
 
            # Measure hardware state
            cpu = cpu_load()
            ram = ram_usage()
            gpu, gpu_t = gpu_load_temp()
 
            # ALWAYS send to cloud — that's the whole point of this baseline
            cloud_ms = cloud_infer_jpeg(args.server, frame, port=args.port)
 
            if cloud_ms > 0:
                cloud_ok = 1
                success_count += 1
            else:
                cloud_ok = 0
                fail_count += 1
 
            # Build CSV row
            row = {
                "timestamp":             time.strftime("%Y-%m-%d %H:%M:%S"),
                "frame_id":              frame_id,
                "scenario":              args.scenario,
                "mode":                  "CLOUD",
                "cloud_latency_ms":      round(cloud_ms, 2),
                "cloud_ok":              cloud_ok,
                "rtt_ms":                round(rtt, 2),
                "bandwidth_kbps":        round(bw, 2),
                "error_rate_pct":        round(loss, 2),
                "cpu_load":              round(cpu, 2),
                "ram_usage":             round(ram, 2),
                "gpu_load":              round(gpu, 2),
                "gpu_temp":              round(gpu_t, 2),
                "successful_inferences": success_count,
                "failed_inferences":     fail_count,
            }
            writer.writerow(row)
            csv_f.flush()
 
            # Console output
            status = "OK " if cloud_ok else "FAIL"
            print(f"frame {frame_id:4d}  [{args.scenario:18s}]  "
                  f"rtt={rtt:6.1f}ms  bw={bw:7.1f}KBps  "
                  f"cpu={cpu:5.1f}%  cloud={cloud_ms:7.1f}ms  [{status}]  "
                  f"({success_count} ok / {fail_count} fail)")
 
            # Stop conditions
            if args.max_samples > 0 and frame_id >= args.max_samples:
                print(f"\n[INFO] Reached max-samples ({args.max_samples}). Stopping.")
                break
            if args.duration > 0 and (time.time() - start_time) >= args.duration:
                print(f"\n[INFO] Reached duration ({args.duration}s). Stopping.")
                break
 
            # Pace the loop
            dt = time.time() - cycle_start
            time.sleep(max(0.0, args.cycle - dt))
 
    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    finally:
        csv_f.close()
        if cap:
            cap.release()
 
        # Final summary
        elapsed = time.time() - start_time
        print("\n" + "=" * 70)
        print("  Summary")
        print("=" * 70)
        print(f"  Total frames:        {frame_id}")
        print(f"  Successful:          {success_count}")
        print(f"  Failed:              {fail_count}")
        if frame_id > 0:
            print(f"  Success rate:        {success_count/frame_id*100:.1f}%")
        print(f"  Duration:            {elapsed:.1f}s")
        if elapsed > 0:
            print(f"  Average rate:        {frame_id/elapsed:.2f} frames/s")
        print(f"  CSV file:            {args.output}")
        print("=" * 70)
 
 
if __name__ == "__main__":
    main()
 
