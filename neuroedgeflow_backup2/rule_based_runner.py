#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
# rule_based_runner.py — Rule-based adaptive baseline for NeuroEdgeFlow
#
# This is the third comparison baseline alongside:
#   1. always_cloud_runner.py — every frame to cloud (no adaptation)
#   2. local_only_runner.py   — every frame on Jetson via DeepStream
#   3. rule_based_runner.py   — THIS FILE: switches per frame using
#                                a hand-tuned threshold rule (rule_regulator)
#
# A fourth runner (main_pipeline.py) uses the trained Random Forest. The
# four CSVs share the same schema so they can be compared side-by-side in
# the dissertation evaluation.
#
# Why a "rule-based adaptive" baseline is necessary
# ─────────────────────────────────────────────────
# Comparing the Random Forest against always-cloud and always-local only
# shows that adaptation helps — not that *learning* helps. The rule-based
# version is the missing middle: it adapts, but using hand-written if/else
# logic instead of a trained model. The RF must beat this to justify
# the learning contribution in the thesis.
#
# Adaptation cycle (per frame)
# ────────────────────────────
#   1. Read current network conditions (RTT, bandwidth, error rate)
#   2. Read current hardware state (CPU, GPU, RAM, temp)
#   3. Call rule_regulator.predict() with these metrics
#   4. If predict says CLOUD -> send frame via cloud_infer_jpeg
#   5. If predict says LOCAL -> run TensorRT locally (edge_pipeline)
#   6. Log the frame to CSV with the same columns as the other runners
#
# Usage
# ─────
#   python3 rule_based_runner.py --scenario ideal --max-samples 150
#   python3 rule_based_runner.py --scenario high_latency --duration 300
#
# Apply tc/netem rules BEFORE running, just like with always_cloud_runner.
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import csv
import os
import sys
import time
import subprocess

import cv2
import numpy as np

# ── Local modules ───────────────────────────────────────────────────────────
try:
    from rule_regulator import RuleBasedRegulator
except ImportError:
    print("[ERROR] rule_regulator.py not found in the current directory.")
    sys.exit(1)

try:
    from cloud_infer_jpeg_v2 import cloud_infer_jpeg, test_connection
except ImportError:
    print("[ERROR] cloud_infer_jpeg_v2.py not found in the current directory.")
    sys.exit(1)

# edge_pipeline is the local TensorRT path. Import is wrapped because the
# Jetson environment may not always have TensorRT/PyCUDA available; the
# runner can still record cloud frames if local is missing.
try:
    import edge_pipeline
    EDGE_AVAILABLE = True
except Exception as exc:
    print("[WARN] edge_pipeline import failed: %s" % exc)
    print("       LOCAL inference will be skipped — frames decided as LOCAL")
    print("       will record -1.0 latency. This is fine for cloud-only runs.")
    EDGE_AVAILABLE = False


# ─── CONFIG ──────────────────────────────────────────────────────────────────
SERVER_IP    = "10.0.20.10"
PROXY_PORT   = 5000
VIDEO_SOURCE = "/home/nvidia/test_video.mp4"
CSV_OUT      = os.path.expanduser("~/rule_based_dataset.csv")
CYCLE_SECONDS = 2.0   # Time between frames (matches the other runners)


# ─── HARDWARE METRICS (same helpers as the other runners) ────────────────────
def cpu_load():
    """Instantaneous CPU utilisation % via /proc/stat (100ms sample)."""
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
    load, temp = 0.0, 0.0
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


# ─── NETWORK METRICS ─────────────────────────────────────────────────────────
def ping_rtt(host, count=3):
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
    """HTTP round-trip bandwidth estimate (same as always_cloud_runner)."""
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


# ─── LOCAL INFERENCE ─────────────────────────────────────────────────────────
def local_infer(frame):
    """
    Run a single frame through the local TensorRT engine (edge_pipeline).
    Returns latency in ms, or -1.0 if local inference is unavailable.
    """
    if not EDGE_AVAILABLE:
        return -1.0
    try:
        _detections, inference_ms = edge_pipeline.run_one_frame(frame)
        return inference_ms
    except Exception as exc:
        print(f"[local] inference failed: {exc}")
        return -1.0


# ─── CSV SCHEMA ──────────────────────────────────────────────────────────────
# Unified schema across all four runners (always_cloud, local_only,
# rule_based, main_pipeline). The four CSVs are now directly concatenable
# in Pandas — every row has the same columns regardless of which baseline
# produced it.
#
# Notes specific to this runner:
#   - mode             : CLOUD or LOCAL (the rule's decision after override)
#   - regulator_reason : explain() string from rule_regulator ("cpu hot",
#                        "rtt 200ms >= 200", etc.). Tells the dissertation
#                        exactly WHY each frame went the way it did.
#   - inference_ms     : whichever path actually ran (cloud round-trip or
#                        local TensorRT time). -1.0 if the inference failed.
#   - frame_ok         : 1 if inference succeeded, 0 otherwise
CSV_HEADER = [
    "timestamp", "frame_id", "scenario",
    "mode",
    "regulator_reason",
    "inference_ms",
    "frame_ok",
    "rtt_ms", "bandwidth_kbps", "error_rate_pct",
    "cpu_load", "ram_usage", "gpu_load", "gpu_temp",
]


# ─── MAIN ────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Rule-based adaptive baseline: per frame, picks "
                    "CLOUD or LOCAL using rule_regulator. Used for "
                    "comparison with always_cloud, local_only, and the "
                    "Random Forest adaptive system."
    )
    p.add_argument("--scenario", default="ideal",
                   help="Scenario label written to every CSV row. Apply the "
                        "matching tc/netem rule before starting.")
    p.add_argument("--max-samples", type=int, default=150,
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
                   help="Seconds between frames.")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("  Rule-Based Adaptive Baseline Runner")
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
    print(f"  Local available:   {EDGE_AVAILABLE}")
    print("=" * 70)

    # Build the rule-based regulator. weights_path is ignored — no model
    # to load — but we keep the argument for API compatibility with the
    # RF regulator.
    regulator = RuleBasedRegulator()

    # Verify cloud connectivity once at startup. We continue even if it
    # fails: the rule will simply keep picking LOCAL.
    print("\nChecking proxy server connection...")
    cloud_available = test_connection(args.server, args.port)
    if not cloud_available:
        print("[WARN] Proxy server not reachable. The rule will keep "
              "choosing LOCAL until the network recovers.")

    # Open the video source. Same logic as the other runners.
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

    # Open CSV. Append mode so multiple scenarios accumulate in one file.
    new_file = not os.path.isfile(args.output)
    csv_f = open(args.output, "a", newline="")
    writer = csv.DictWriter(csv_f, fieldnames=CSV_HEADER)
    if new_file:
        writer.writeheader()
    print(f"[OK] Writing to {args.output}")
    print("[INFO] Press Ctrl+C to stop.\n")

    # Counters used both in the live log line and in each CSV row
    frame_id = 0
    cloud_count = 0
    local_count = 0
    failed_count = 0
    start_time = time.time()

    try:
        while True:
            cycle_start = time.time()
            frame_id += 1

            # ── 1. Grab frame ─────────────────────────────────────────────
            if is_image:
                frame = still.copy()
            else:
                ok, frame = cap.read()
                if not ok:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = cap.read()
                    if not ok:
                        print("[ERROR] Cannot read frame.")
                        break

            # ── 2. Measure conditions (the rule's input) ──────────────────
            rtt, loss = ping_rtt(args.server)
            bw = bandwidth_kbps(args.server)
            cpu = cpu_load()
            ram = ram_usage()
            gpu, gpu_t = gpu_load_temp()

            # The rule has no memory of past latencies — it works from the
            # current snapshot only. We pass -1.0 for the latencies so the
            # rule falls back to the cpu_hot / network_healthy logic.
            metrics = {
                "rtt_ms":           rtt,
                "bandwidth_kbps":   bw,
                "error_rate_pct":   loss,
                "cpu_load":         cpu,
                "ram_usage":        ram,
                "local_latency_ms": -1.0,
                "cloud_latency_ms": -1.0,
            }

            # ── 3. Ask the rule ───────────────────────────────────────────
            cloud_win, prob = regulator.predict(metrics=metrics)

            # Record the rule's *raw* decision before any safety overrides.
            # If the cloud is known unreachable, we still log the rule's
            # decision but force the action to LOCAL — this lets the CSV
            # show how often the rule was overridden by reality.
            regulator_decision = "CLOUD" if cloud_win else "LOCAL"
            regulator_reason = regulator._explain(
                rtt, loss, bw, cpu,
                -1.0, -1.0,
                regulator._network_healthy(rtt, loss, bw),
                regulator._offload_worth_it(cpu, -1.0, -1.0)
            )

            # Apply the cloud-availability override
            if cloud_win and not cloud_available:
                effective_mode = "LOCAL"   # forced override
            else:
                effective_mode = regulator_decision

            # ── 4. Execute the chosen path ────────────────────────────────
            cloud_ms = -1.0
            local_ms = -1.0
            inference_ms = -1.0
            frame_ok = 0

            if effective_mode == "CLOUD":
                cloud_ms = cloud_infer_jpeg(args.server, frame, port=args.port)
                if cloud_ms > 0:
                    inference_ms = cloud_ms
                    frame_ok = 1
                    cloud_count += 1
                else:
                    failed_count += 1
            else:
                local_ms = local_infer(frame)
                if local_ms > 0:
                    inference_ms = local_ms
                    frame_ok = 1
                    local_count += 1
                else:
                    failed_count += 1

            # ── 5. Log the row using the unified schema ──────────────────
            # cloud_count, local_count, failed_count are dropped from the
            # CSV (they're easy to compute in Pandas with .cumsum()), but
            # we still track them in memory for the console summary.
            row = {
                "timestamp":        time.strftime("%Y-%m-%d %H:%M:%S"),
                "frame_id":         frame_id,
                "scenario":         args.scenario,
                "mode":             effective_mode,
                "regulator_reason": regulator_reason,
                "inference_ms":     round(inference_ms, 2),
                "frame_ok":         frame_ok,
                "rtt_ms":           round(rtt, 2),
                "bandwidth_kbps":   round(bw, 2),
                "error_rate_pct":   round(loss, 2),
                "cpu_load":         round(cpu, 2),
                "ram_usage":        round(ram, 2),
                "gpu_load":         round(gpu, 2),
                "gpu_temp":         round(gpu_t, 2),
            }
            writer.writerow(row)
            csv_f.flush()

            # ── 6. Console line ───────────────────────────────────────────
            status = "OK " if frame_ok else "FAIL"
            mode_marker = "C" if effective_mode == "CLOUD" else "L"
            print(f"frame {frame_id:4d}  [{args.scenario:15s}]  "
                  f"[{mode_marker}]  rtt={rtt:6.1f}  cpu={cpu:5.1f}%  "
                  f"inf={inference_ms:7.1f}ms  p={prob:.2f}  [{status}]  "
                  f"(c={cloud_count} l={local_count} f={failed_count})")

            # ── 7. Stop conditions ────────────────────────────────────────
            if args.max_samples > 0 and frame_id >= args.max_samples:
                print(f"\n[INFO] Reached max-samples ({args.max_samples}).")
                break
            if args.duration > 0 and (time.time() - start_time) >= args.duration:
                print(f"\n[INFO] Reached duration ({args.duration}s).")
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

        elapsed = time.time() - start_time
        print("\n" + "=" * 70)
        print("  Summary")
        print("=" * 70)
        print(f"  Total frames:        {frame_id}")
        print(f"  Cloud decisions:     {cloud_count}")
        print(f"  Local decisions:     {local_count}")
        print(f"  Failed:              {failed_count}")
        if frame_id > 0:
            cloud_pct = cloud_count / frame_id * 100
            local_pct = local_count / frame_id * 100
            print(f"  Mode split:          CLOUD {cloud_pct:.1f}% / "
                  f"LOCAL {local_pct:.1f}%")
        print(f"  Duration:            {elapsed:.1f}s")
        if elapsed > 0:
            print(f"  Average rate:        {frame_id/elapsed:.2f} frames/s")
        print(f"  CSV file:            {args.output}")
        print("=" * 70)


if __name__ == "__main__":
    main()
