#!/usr/bin/env python3
# =============================================================================
# compare_modes.py — NeuroEdgeFlow Sprint 5 Task 09
# Compare Adaptive vs Always-Local vs Always-Cloud
# on average FPS, average RTT, and total bandwidth used.
#
# Data sources:
#   Always-Local  — Sprint 2 benchmark constants (no CSV, DeepStream had no logger)
#   Always-Cloud  — Sprint 3 ~/cloud_metrics.csv  (25,603 rows, real measurements)
#   Adaptive      — Sprint 5 Task 08 adaptive_engine.csv (341 polls, real session)
#                   + mode_switches.csv (9 switches with timestamps)
#
# Usage:
#   python3 compare_modes.py \
#     --cloud  ~/cloud_metrics.csv \
#     --engine ~/session_task08/adaptive_engine.csv \
#     --switches ~/session_task08/mode_switches.csv \
#     --session-duration 682
# =============================================================================

from __future__ import print_function

import argparse
import csv
import os
import sys

# =============================================================================
# SPRINT 2 ALWAYS-LOCAL CONSTANTS (from benchmark — no CSV exists)
# =============================================================================
LOCAL_FPS_BENCHMARK       = 29.65   # Optimized + Fakesink, MAX-N, 5 min sustained
LOCAL_BANDWIDTH_KB_S      = 0.0     # No network traffic — pure edge
LOCAL_RTT_MS              = 0.0     # No network — not applicable
LOCAL_SESSION_DURATION_S  = 600.0   # Normalise to 10 min for fair comparison

# =============================================================================
# HELPERS
# =============================================================================

def load_csv(path, label):
    if not path or not os.path.exists(path):
        print("  WARNING: {0} not found — {1}".format(path, label))
        return []
    with open(path) as f:
        rows = list(csv.DictReader(f))
    print("  Loaded {0}: {1} rows".format(label, len(rows)))
    return rows


def safe_float(val, default=0.0):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def section(title):
    print("\n" + "=" * 68)
    print("  " + title)
    print("=" * 68)


def bar(value, max_value, width=30, char="█"):
    if max_value == 0:
        return " " * width
    filled = int(round(value / max_value * width))
    return char * filled + "░" * (width - filled)


# =============================================================================
# ALWAYS-CLOUD ANALYSIS (Sprint 3 cloud_metrics.csv)
# =============================================================================

def analyze_cloud(rows):
    """
    Returns dict of Always-Cloud metrics from Sprint 3 CSV.

    Columns available: timestamp, frame_id, mode, frame_size_kb,
    encode_ms, grpc_ms, inference_ms, decode_ms, total_ms,
    num_detections, fps
    """
    if not rows:
        return None

    fps_vals    = [safe_float(r.get("fps"))           for r in rows if safe_float(r.get("fps")) > 0]
    grpc_vals   = [safe_float(r.get("grpc_ms"))       for r in rows if safe_float(r.get("grpc_ms")) > 0]
    size_vals   = [safe_float(r.get("frame_size_kb")) for r in rows if safe_float(r.get("frame_size_kb")) > 0]
    total_vals  = [safe_float(r.get("total_ms"))      for r in rows if safe_float(r.get("total_ms")) > 0]

    avg_fps     = sum(fps_vals)   / len(fps_vals)   if fps_vals   else 0.0
    avg_grpc_ms = sum(grpc_vals)  / len(grpc_vals)  if grpc_vals  else 0.0
    avg_size_kb = sum(size_vals)  / len(size_vals)  if size_vals  else 0.0
    avg_total_ms= sum(total_vals) / len(total_vals) if total_vals else 0.0

    # Total bandwidth = sum of all frame sizes sent over the session
    total_bandwidth_kb = sum(size_vals)
    total_bandwidth_mb = total_bandwidth_kb / 1024.0

    # Session duration estimate from frame count and average FPS
    n_frames = len(fps_vals)
    est_duration_s = n_frames / avg_fps if avg_fps > 0 else 0.0

    # Average bandwidth rate
    avg_bandwidth_kb_s = total_bandwidth_kb / est_duration_s if est_duration_s > 0 else 0.0

    return {
        "avg_fps":           avg_fps,
        "avg_rtt_ms":        avg_grpc_ms,    # gRPC latency = effective RTT for cloud
        "avg_total_ms":      avg_total_ms,
        "avg_frame_size_kb": avg_size_kb,
        "total_bandwidth_mb":total_bandwidth_mb,
        "avg_bandwidth_kb_s":avg_bandwidth_kb_s,
        "n_frames":          n_frames,
        "est_duration_s":    est_duration_s,
    }


# =============================================================================
# ADAPTIVE ANALYSIS (Sprint 5 Task 08 session)
# =============================================================================

def analyze_adaptive(engine_rows, switch_rows, session_duration_s,
                     cloud_ideal_fps, cloud_ideal_rtt_ms):
    """
    Computes Adaptive metrics from the Task 08 session data.

    FPS strategy:
      - Cannot use main_pipeline.csv (0 frames — STUB mode, no camera)
      - Compute time spent in LOCAL vs CLOUD from engine poll data
      - Apply LOCAL=29.65 FPS and CLOUD=Sprint3 ideal FPS as weights
      - This is the correct methodology: actual decision timing, known pipeline FPS

    Bandwidth strategy:
      - Bandwidth is only consumed when in CLOUD mode
      - Use avg_bandwidth_kb_s from Sprint 3 × time spent in CLOUD

    RTT:
      - Use measured rtt_ms from every engine poll (both modes)
      - This reflects the real network conditions the engine observed
    """
    if not engine_rows:
        return None

    # ── Time in each mode from poll data ─────────────────────────────────────
    local_polls = [r for r in engine_rows if r.get("current_mode") == "LOCAL"]
    cloud_polls = [r for r in engine_rows if r.get("current_mode") == "CLOUD"]
    total_polls = len(engine_rows)

    frac_local = len(local_polls) / total_polls if total_polls > 0 else 0.5
    frac_cloud = len(cloud_polls) / total_polls if total_polls > 0 else 0.5

    time_local_s = frac_local * session_duration_s
    time_cloud_s = frac_cloud * session_duration_s

    # ── Weighted average FPS ──────────────────────────────────────────────────
    # Frames produced = time_local × local_fps + time_cloud × cloud_fps
    frames_local = time_local_s * LOCAL_FPS_BENCHMARK
    frames_cloud = time_cloud_s * cloud_ideal_fps
    total_frames  = frames_local + frames_cloud
    avg_fps       = total_frames / session_duration_s if session_duration_s > 0 else 0.0

    # ── RTT — measured from all polls ────────────────────────────────────────
    rtt_vals = [safe_float(r.get("rtt_ms"))
                for r in engine_rows
                if safe_float(r.get("rtt_ms")) > 0]
    avg_rtt_ms = sum(rtt_vals) / len(rtt_vals) if rtt_vals else 0.0

    # ── Bandwidth — only during CLOUD periods ────────────────────────────────
    # Use Sprint 3 ideal bandwidth rate × time in CLOUD
    # Sprint 3 ideal: 14.9 KB/frame × 2.4 FPS = 35.76 KB/s
    # But we use the measured avg_bandwidth_kb_s passed in as cloud_ideal_bw
    # We pass this in from the Always-Cloud analysis

    return {
        "avg_fps":           avg_fps,
        "avg_rtt_ms":        avg_rtt_ms,
        "frac_local":        frac_local,
        "frac_cloud":        frac_cloud,
        "time_local_s":      time_local_s,
        "time_cloud_s":      time_cloud_s,
        "frames_local":      frames_local,
        "frames_cloud":      frames_cloud,
        "total_polls":       total_polls,
        "local_polls":       len(local_polls),
        "cloud_polls":       len(cloud_polls),
        "n_switches":        len(switch_rows),
        "session_duration_s":session_duration_s,
    }


# =============================================================================
# MAIN COMPARISON
# =============================================================================

def compare(args):
    section("Loading data")
    cloud_rows   = load_csv(args.cloud,   "Always-Cloud (Sprint 3)")
    engine_rows  = load_csv(args.engine,  "Adaptive engine polls (Sprint 5)")
    switch_rows  = load_csv(args.switches,"Adaptive mode switches (Sprint 5)")

    section("Analyzing Always-Cloud (Sprint 3)")
    cloud = analyze_cloud(cloud_rows)
    if cloud:
        print("  Avg FPS         : {0:.2f}".format(cloud["avg_fps"]))
        print("  Avg gRPC RTT    : {0:.1f} ms".format(cloud["avg_rtt_ms"]))
        print("  Avg frame size  : {0:.1f} KB".format(cloud["avg_frame_size_kb"]))
        print("  Total bandwidth : {0:.1f} MB ({1:.0f} s session)".format(
            cloud["total_bandwidth_mb"], cloud["est_duration_s"]))
        print("  Avg BW rate     : {0:.1f} KB/s".format(cloud["avg_bandwidth_kb_s"]))
    else:
        print("  No cloud data — using Sprint 3 benchmark defaults")
        cloud = {
            "avg_fps": 2.4, "avg_rtt_ms": 350.0, "avg_frame_size_kb": 14.9,
            "total_bandwidth_mb": 0.0, "avg_bandwidth_kb_s": 35.76,
            "n_frames": 0, "est_duration_s": 600.0
        }

    section("Analyzing Adaptive (Sprint 5 Task 08 session)")
    adaptive = analyze_adaptive(
        engine_rows, switch_rows,
        session_duration_s = args.session_duration,
        cloud_ideal_fps    = cloud["avg_fps"],
        cloud_ideal_rtt_ms = cloud["avg_rtt_ms"],
    )
    if adaptive:
        print("  Session duration : {0:.0f} s".format(adaptive["session_duration_s"]))
        print("  Total polls      : {0}".format(adaptive["total_polls"]))
        print("  Polls in LOCAL   : {0} ({1:.1f}%)".format(
            adaptive["local_polls"],  adaptive["frac_local"] * 100))
        print("  Polls in CLOUD   : {0} ({1:.1f}%)".format(
            adaptive["cloud_polls"],  adaptive["frac_cloud"] * 100))
        print("  Mode switches    : {0}".format(adaptive["n_switches"]))
        print("  Weighted avg FPS : {0:.2f}".format(adaptive["avg_fps"]))
        print("  Avg RTT (all polls): {0:.1f} ms".format(adaptive["avg_rtt_ms"]))

        # Bandwidth: only during CLOUD periods
        bw_cloud_kb = cloud["avg_bandwidth_kb_s"] * adaptive["time_cloud_s"]
        bw_cloud_mb = bw_cloud_kb / 1024.0
        adaptive["total_bandwidth_mb"] = bw_cloud_mb
        adaptive["avg_bandwidth_kb_s"] = (
            bw_cloud_kb / adaptive["session_duration_s"]
            if adaptive["session_duration_s"] > 0 else 0.0
        )
        print("  Time in CLOUD    : {0:.0f} s".format(adaptive["time_cloud_s"]))
        print("  Bandwidth (CLOUD periods only): {0:.1f} MB ({1:.1f} KB/s avg)".format(
            bw_cloud_mb, adaptive["avg_bandwidth_kb_s"]))

    # ── Normalise bandwidth to a 600s (10 min) session ───────────────────────
    # Cloud session was ~{est_duration_s}s, adaptive was {session_duration_s}s
    # Normalise both to 600s for fair comparison
    norm_s = 600.0

    local_bw_norm_mb  = 0.0
    cloud_bw_norm_mb  = (cloud["avg_bandwidth_kb_s"] * norm_s) / 1024.0
    adapt_bw_norm_mb  = (adaptive["avg_bandwidth_kb_s"] * norm_s) / 1024.0 if adaptive else 0.0

    # ── Comparison table ──────────────────────────────────────────────────────
    section("Comparison Table — Adaptive vs Always-Local vs Always-Cloud")

    # FPS
    fps_local  = LOCAL_FPS_BENCHMARK
    fps_cloud  = cloud["avg_fps"]
    fps_adapt  = adaptive["avg_fps"] if adaptive else 0.0
    fps_max    = max(fps_local, fps_cloud, fps_adapt)

    # RTT
    rtt_local  = LOCAL_RTT_MS
    rtt_cloud  = cloud["avg_rtt_ms"]
    rtt_adapt  = adaptive["avg_rtt_ms"] if adaptive else 0.0
    rtt_max    = max(rtt_local, rtt_cloud, rtt_adapt, 1.0)

    # Bandwidth (normalised to 10 min)
    bw_max     = max(local_bw_norm_mb, cloud_bw_norm_mb, adapt_bw_norm_mb, 0.01)

    COL = 18
    W   = 28

    print("\n  {:<{c}}  {:>{w}}  {:>{w}}  {:>{w}}".format(
        "Metric", "Always-Local", "Always-Cloud", "Adaptive", c=COL, w=W))
    print("  " + "-" * (COL + 3 * W + 6))

    def fmt_row(label, vals, fmt, unit="", bar_vals=None, lower_is_better=False):
        cells = []
        for v in vals:
            cells.append(("{0:" + fmt + "}{1}").format(v, unit))
        print("  {:<{c}}  {:>{w}}  {:>{w}}  {:>{w}}".format(
            label, cells[0], cells[1], cells[2], c=COL, w=W))
        if bar_vals:
            bars = []
            for v in bar_vals:
                if lower_is_better and max(bar_vals) > 0:
                    b = bar(max(bar_vals) - v, max(bar_vals), width=20)
                else:
                    b = bar(v, max(bar_vals), width=20)
                bars.append(b)
            print("  {:<{c}}  {:>{w}}  {:>{w}}  {:>{w}}".format(
                "", bars[0], bars[1], bars[2], c=COL, w=W))

    fmt_row("Avg FPS",
            [fps_local, fps_cloud, fps_adapt], ".2f", " FPS",
            bar_vals=[fps_local, fps_cloud, fps_adapt])
    print()
    fmt_row("Avg RTT / Latency",
            [rtt_local, rtt_cloud, rtt_adapt], ".1f", " ms",
            bar_vals=[rtt_local, rtt_cloud, rtt_adapt],
            lower_is_better=True)
    print()
    fmt_row("Total BW (10 min)",
            [local_bw_norm_mb, cloud_bw_norm_mb, adapt_bw_norm_mb], ".1f", " MB",
            bar_vals=[local_bw_norm_mb, cloud_bw_norm_mb, adapt_bw_norm_mb],
            lower_is_better=True)
    print()

    # Derived metrics
    fps_gain_over_cloud = ((fps_adapt - fps_cloud) / fps_cloud * 100) if fps_cloud > 0 else 0
    bw_saving_vs_cloud  = ((cloud_bw_norm_mb - adapt_bw_norm_mb) / cloud_bw_norm_mb * 100) if cloud_bw_norm_mb > 0 else 0
    fps_vs_local_pct    = (fps_adapt / fps_local * 100) if fps_local > 0 else 0

    section("Key Findings")
    print("  Adaptive FPS vs Always-Cloud : {0:+.1f}%  ({1:.2f} vs {2:.2f} FPS)".format(
        fps_gain_over_cloud, fps_adapt, fps_cloud))
    print("  Adaptive FPS vs Always-Local : {0:.1f}%  ({1:.2f} vs {2:.2f} FPS)".format(
        fps_vs_local_pct, fps_adapt, fps_local))
    print("  Bandwidth saving vs Cloud    : {0:.1f}%  ({1:.1f} MB vs {2:.1f} MB per 10 min)".format(
        bw_saving_vs_cloud, adapt_bw_norm_mb, cloud_bw_norm_mb))
    print("  Adaptive RTT vs Cloud RTT   : {0:.1f} ms vs {1:.1f} ms".format(
        rtt_adapt, rtt_cloud))
    print()
    print("  Mode split (Task 08 session):")
    if adaptive:
        print("    LOCAL: {0:.1f}%  ({1:.0f} s)  → TensorRT FP16, 0 bandwidth".format(
            adaptive["frac_local"] * 100, adaptive["time_local_s"]))
        print("    CLOUD: {0:.1f}%  ({1:.0f} s)  → Triton gRPC, {2:.1f} KB/s".format(
            adaptive["frac_cloud"] * 100, adaptive["time_cloud_s"],
            cloud["avg_bandwidth_kb_s"]))

    section("Dissertation Summary Table")
    print()
    print("  | Metric                  | Always-Local | Always-Cloud | Adaptive      |")
    print("  |-------------------------|-------------|-------------|---------------|")
    print("  | Avg FPS                 | {0:>11.2f} | {1:>11.2f} | {2:>13.2f} |".format(
        fps_local, fps_cloud, fps_adapt))
    print("  | Avg RTT / gRPC latency  | {0:>9.0f} ms | {1:>9.1f} ms | {2:>11.1f} ms |".format(
        rtt_local, rtt_cloud, rtt_adapt))
    print("  | Total BW (10 min)       | {0:>9.1f} MB | {1:>9.1f} MB | {2:>11.1f} MB |".format(
        local_bw_norm_mb, cloud_bw_norm_mb, adapt_bw_norm_mb))
    print("  | FPS vs Always-Cloud     |     baseline |     baseline | {0:>+11.1f}% |".format(
        fps_gain_over_cloud))
    print("  | BW saving vs Cloud      |          N/A |     baseline | {0:>+11.1f}% |".format(
        -bw_saving_vs_cloud))
    print()

    section("Data Sources and Methodology Notes")
    print("""
  Always-Local:
    FPS: Sprint 2 benchmark — Optimized + Fakesink, MAX-N, 5 min sustained (29.65)
    RTT: Not applicable — pure edge inference, zero network traffic
    BW:  Zero — no frames sent over the network

  Always-Cloud:
    FPS: Measured from Sprint 3 cloud_metrics.csv ({0} frames)
    RTT: gRPC round-trip time (encode→send→infer→receive) from cloud_metrics.csv
    BW:  Sum of all frame_size_kb values from cloud_metrics.csv
         (JPEG-compressed frames sent to Triton at quality=70)

  Adaptive:
    FPS: Weighted average — time_LOCAL × 29.65 + time_CLOUD × {1:.2f} FPS
         Weights from Task 08 session engine poll data ({2} polls)
    RTT: Mean rtt_ms across all {2} engine polls (both modes included)
         Reflects real network conditions the engine observed
    BW:  Always-Cloud rate × time spent in CLOUD only
         Bandwidth = 0 during LOCAL periods (no network traffic)
    Session: {3:.0f} s, {4} mode switches confirmed by hysteresis
""".format(
        cloud.get("n_frames", 0),
        cloud["avg_fps"],
        len(engine_rows),
        args.session_duration,
        len(switch_rows)
    ))


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NeuroEdgeFlow Sprint 5 Task 09 — mode comparison"
    )
    parser.add_argument("--cloud",    required=True,
                        help="Path to Sprint 3 cloud_metrics.csv")
    parser.add_argument("--engine",   required=True,
                        help="Path to Sprint 5 adaptive_engine.csv")
    parser.add_argument("--switches", required=True,
                        help="Path to Sprint 5 mode_switches.csv")
    parser.add_argument("--session-duration", type=float, default=682.0,
                        help="Adaptive session duration in seconds (default 682)")
    args = parser.parse_args()
    compare(args)
