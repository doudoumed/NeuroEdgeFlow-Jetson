#!/usr/bin/env python3
# =============================================================================
# analyze_session.py — NeuroEdgeFlow Sprint 5 Task 08
# Post-session analysis for the 10-minute live degradation session.
#
# Reads three CSV files produced during the live session and confirms:
#   1. Engine started in LOCAL (fail-safe)
#   2. Switched to CLOUD once network was confirmed healthy
#   3. Switched back to LOCAL when NetEm degradation was applied
#   4. Recovered to CLOUD after NetEm was removed
#   5. Hysteresis count was respected (no premature switches)
#   6. All switch rows contain the exact triggering metric values
#
# Run on laptop after scp'ing CSVs from Jetson:
#   scp nvidia@192.168.1.16:~/mode_switches.csv     ~/session_task08/
#   scp nvidia@192.168.1.16:~/adaptive_engine.csv   ~/session_task08/
#   scp nvidia@192.168.1.16:~/main_pipeline.csv     ~/session_task08/
#
#   python3 analyze_session.py --dir ~/session_task08/
# =============================================================================

from __future__ import print_function

import argparse
import csv
import os
import sys

# =============================================================================
# CONFIGURATION — must match adaptive_engine.py constants
# =============================================================================
HYSTERESIS_COUNT        = 3
FALLBACK_ERROR_THRESHOLD = 3
FALLBACK_RECOVERY_POLLS  = 5
POLL_INTERVAL_S          = 2.0

# Expected thresholds (from network_monitor.py / Task 03 justification)
RTT_THRESHOLD_MS         = 200.0
BW_THRESHOLD_KBPS        = 100.0
ERROR_THRESHOLD_PCT       = 5.0

# =============================================================================
# HELPERS
# =============================================================================

_passed = 0
_failed = 0
_checks = []


def check(cond, name, detail=""):
    global _passed, _failed
    status = "PASS" if cond else "FAIL"
    if cond:
        _passed += 1
    else:
        _failed += 1
    msg = "  [{0}]  {1}".format(status, name)
    if detail:
        msg += "\n         " + detail
    print(msg)
    _checks.append((status, name, detail))


def section(title):
    print("\n" + "=" * 64)
    print("  " + title)
    print("=" * 64)


def load_csv(path, label):
    if not os.path.exists(path):
        print("  MISSING: {0} ({1})".format(path, label))
        return []
    with open(path) as f:
        rows = list(csv.DictReader(f))
    print("  Loaded {0}: {1} rows".format(label, len(rows)))
    return rows


# =============================================================================
# ANALYSIS
# =============================================================================

def analyze(session_dir):
    sw_path  = os.path.join(session_dir, "mode_switches.csv")
    eng_path = os.path.join(session_dir, "adaptive_engine.csv")
    pip_path = os.path.join(session_dir, "main_pipeline.csv")

    section("Loading CSV files")
    switches = load_csv(sw_path,  "mode_switches.csv")
    engine   = load_csv(eng_path, "adaptive_engine.csv")
    pipeline = load_csv(pip_path, "main_pipeline.csv")

    if not engine:
        print("\nFATAL: adaptive_engine.csv is empty or missing — cannot analyze.")
        sys.exit(1)

    # ── Session duration ─────────────────────────────────────────────────────
    section("Session Overview")
    first_ts = engine[0]["timestamp"]
    last_ts  = engine[-1]["timestamp"]
    n_polls  = len(engine)
    n_frames = len(pipeline)
    n_switches = len(switches)

    print("  First poll : {0}".format(first_ts))
    print("  Last poll  : {0}".format(last_ts))
    print("  Total polls: {0}  (~{1:.0f} s at 2 s interval)".format(
        n_polls, n_polls * POLL_INTERVAL_S))
    print("  Frames logged: {0}".format(n_frames))
    print("  Mode switches: {0}".format(n_switches))

    check(n_polls >= 250,
          "Session ran for at least ~500 s (10 min)",
          "got {0} polls = ~{1:.0f} s".format(n_polls, n_polls * POLL_INTERVAL_S))

    # ── Check 1: initial mode ─────────────────────────────────────────────────
    section("Check 1 — Initial mode is LOCAL (fail-safe)")
    first_mode = engine[0]["current_mode"]
    check(first_mode == "LOCAL",
          "Engine started in LOCAL",
          "first poll current_mode={0}".format(first_mode))

    # ── Check 2: at least one LOCAL→CLOUD switch occurred ────────────────────
    section("Check 2 — Engine switched to CLOUD when network was healthy")
    local_to_cloud = [r for r in switches
                      if r["previous_mode"] == "LOCAL"
                      and r["new_mode"] == "CLOUD"
                      and r["trigger_type"] == "HYSTERESIS"]
    check(len(local_to_cloud) >= 1,
          "At least one LOCAL→CLOUD HYSTERESIS switch",
          "found {0}".format(len(local_to_cloud)))

    if local_to_cloud:
        r = local_to_cloud[0]
        rtt  = float(r["rtt_ms"])
        bw   = float(r["bandwidth_kbps"])
        err  = float(r["error_rate_pct"])
        check(rtt  < RTT_THRESHOLD_MS,
              "RTT < 200ms at LOCAL→CLOUD switch",
              "rtt={0:.1f} ms".format(rtt))
        check(bw   > BW_THRESHOLD_KBPS,
              "BW > 100 KB/s at LOCAL→CLOUD switch",
              "bw={0:.1f} KB/s".format(bw))
        check(err  < ERROR_THRESHOLD_PCT,
              "Error rate < 5% at LOCAL→CLOUD switch",
              "err={0:.1f}%".format(err))
        check(int(r["polls_confirmed"]) == HYSTERESIS_COUNT,
              "polls_confirmed == HYSTERESIS_COUNT (3)",
              "polls_confirmed={0}".format(r["polls_confirmed"]))

    # ── Check 3: at least one CLOUD→LOCAL switch after degradation ───────────
    section("Check 3 — Engine switched back to LOCAL when network degraded")
    cloud_to_local = [r for r in switches
                      if r["previous_mode"] == "CLOUD"
                      and r["new_mode"] == "LOCAL"]
    check(len(cloud_to_local) >= 1,
          "At least one CLOUD→LOCAL switch after degradation",
          "found {0}".format(len(cloud_to_local)))

    if cloud_to_local:
        r = cloud_to_local[0]
        rtt  = float(r["rtt_ms"])
        bw   = float(r["bandwidth_kbps"])
        err  = float(r["error_rate_pct"])
        tt   = r["trigger_type"]
        check(tt in ("HYSTERESIS", "FALLBACK"),
              "CLOUD→LOCAL trigger is HYSTERESIS or FALLBACK",
              "trigger_type={0}".format(tt))
        check(rtt >= RTT_THRESHOLD_MS or bw <= BW_THRESHOLD_KBPS or err >= ERROR_THRESHOLD_PCT,
              "At least one threshold breached at CLOUD→LOCAL switch",
              "rtt={0:.1f} ms  bw={1:.1f} KB/s  err={2:.1f}%".format(rtt, bw, err))
        print("  Switch details:")
        print("    timestamp      : {0}".format(r["timestamp"]))
        print("    trigger_type   : {0}".format(tt))
        print("    rtt_ms         : {0}".format(r["rtt_ms"]))
        print("    bandwidth_kbps : {0}".format(r["bandwidth_kbps"]))
        print("    error_rate_pct : {0}".format(r["error_rate_pct"]))
        if r.get("reason"):
            print("    reason         : {0}".format(r["reason"]))

    # ── Check 4: hysteresis respected (no premature switches) ─────────────────
    section("Check 4 — Hysteresis respected (all switches confirmed by 3 polls)")
    hysteresis_switches = [r for r in switches if r["trigger_type"] == "HYSTERESIS"]
    bad_count = [r for r in hysteresis_switches
                 if int(r["polls_confirmed"]) != HYSTERESIS_COUNT]
    check(len(bad_count) == 0,
          "All HYSTERESIS switches have polls_confirmed == 3",
          "{0} bad, {1} total hysteresis switches".format(
              len(bad_count), len(hysteresis_switches)))

    # ── Check 5: FPS analysis ─────────────────────────────────────────────────
    section("Check 5 — FPS by mode")
    if pipeline:
        local_fps  = [float(r["fps"]) for r in pipeline
                      if r["mode"] == "LOCAL" and float(r["fps"]) > 0]
        cloud_fps  = [float(r["fps"]) for r in pipeline
                      if r["mode"] == "CLOUD" and float(r["fps"]) > 0]

        avg_local = sum(local_fps)  / len(local_fps)  if local_fps  else 0
        avg_cloud = sum(cloud_fps)  / len(cloud_fps)  if cloud_fps  else 0

        print("  LOCAL  frames: {0:5d}  avg FPS: {1:.2f}".format(
            len(local_fps), avg_local))
        print("  CLOUD  frames: {0:5d}  avg FPS: {1:.2f}".format(
            len(cloud_fps), avg_cloud))

        check(avg_local > avg_cloud or avg_cloud == 0,
              "LOCAL avg FPS > CLOUD avg FPS (TensorRT faster than gRPC)",
              "LOCAL={0:.2f}  CLOUD={1:.2f}".format(avg_local, avg_cloud))

    # ── Check 6: RTT during degradation ──────────────────────────────────────
    section("Check 6 — RTT spike visible in adaptive_engine.csv")
    high_rtt_polls = [r for r in engine
                      if float(r["rtt_ms"]) >= RTT_THRESHOLD_MS]
    check(len(high_rtt_polls) >= 3,
          "At least 3 polls with RTT >= 200ms (degradation was applied)",
          "found {0} polls with RTT >= 200ms".format(len(high_rtt_polls)))

    if high_rtt_polls:
        max_rtt = max(float(r["rtt_ms"]) for r in high_rtt_polls)
        print("  Peak RTT during degradation: {0:.1f} ms".format(max_rtt))

    # ── Check 7: no spurious switches during stable periods ──────────────────
    section("Check 7 — No switches during stable STAY periods")
    # Count polls where mode stayed stable vs polls near a switch
    transition_polls = [r for r in engine if r["transition"] == "1"]
    stable_polls     = [r for r in engine if r["transition"] == "0"
                        and r["pipeline_action"] == "none"]
    print("  Transition polls : {0}".format(len(transition_polls)))
    print("  Stable polls     : {0}".format(len(stable_polls)))
    check(len(stable_polls) > len(transition_polls),
          "Majority of polls are stable (engine is not thrashing)",
          "stable={0}  transition={1}".format(
              len(stable_polls), len(transition_polls)))

    # ── Check 8: mode switches CSV completeness ───────────────────────────────
    section("Check 8 — mode_switches.csv completeness")
    for i, r in enumerate(switches):
        for col in ["timestamp", "previous_mode", "new_mode", "trigger_type",
                    "rtt_ms", "bandwidth_kbps", "error_rate_pct"]:
            check(col in r and r[col] != "",
                  "Switch row {0} has non-empty '{1}'".format(i, col),
                  "value={0!r}".format(r.get(col, "MISSING")))

    # ── Summary ───────────────────────────────────────────────────────────────
    section("Session Summary — All Mode Switches")
    if switches:
        print("  {:<22} {:<8} {:<8} {:<12} {:>8} {:>10} {:>7}".format(
            "Timestamp", "From", "To", "Trigger", "RTT(ms)", "BW(KB/s)", "Err%"))
        print("  " + "-" * 80)
        for r in switches:
            print("  {:<22} {:<8} {:<8} {:<12} {:>8} {:>10} {:>7}".format(
                r["timestamp"],
                r["previous_mode"],
                r["new_mode"],
                r["trigger_type"],
                r["rtt_ms"],
                r["bandwidth_kbps"],
                r["error_rate_pct"],
            ))
    else:
        print("  No switches recorded.")

    # ── Final result ──────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("  RESULT: {0} passed, {1} failed".format(_passed, _failed))
    print("=" * 64)
    return _failed == 0


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NeuroEdgeFlow Sprint 5 Task 08 session analyzer"
    )
    parser.add_argument(
        "--dir", required=True,
        help="Directory containing mode_switches.csv, adaptive_engine.csv, main_pipeline.csv"
    )
    args = parser.parse_args()

    ok = analyze(os.path.expanduser(args.dir))
    sys.exit(0 if ok else 1)
