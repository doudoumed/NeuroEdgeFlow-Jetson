#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# collect_for_rf_training.sh — Jetson-side training data collection
#
# Companion to full_sweep_server.sh. Runs the dual-inference collector
# continuously while cycling CPU stress on the Jetson. Together they
# expose the RF training set to a wide combination of:
#
#   - Network conditions (from the server: RTT, BW, Loss)
#   - Hardware load    (from this script: CPU stress levels)
#
# The collector writes one row every ~2 seconds. Each row captures the
# current network metrics (set by the server) AND the current CPU load
# (set by this script). Over ~30-40 min that is ~1000 training samples
# with broad coverage of the input space.
#
# What it does
# ────────────
#   1. Starts collect_training_data.py in the background
#   2. Cycles CPU stress: 0% → 25% → 50% → 75% → 100% → 75% → ... → 0%
#      Each level held for STRESS_HOLD_SEC seconds.
#   3. Logs every stress transition with a timestamp so the CSV can be
#      joined to this log if needed.
#   4. On exit: kills the collector AND any stress-ng processes.
#
# Usage
# ─────
#   # Default: collector + CPU stress cycling
#   ./collect_for_rf_training.sh
#
#   # No CPU stress (just collect — server controls everything)
#   STRESS_MODE=none ./collect_for_rf_training.sh
#
#   # Custom CPU levels and hold time
#   CPU_LEVELS="0 50 100" STRESS_HOLD_SEC=120 ./collect_for_rf_training.sh
#
# Stop with Ctrl+C — everything is cleaned up automatically.
# ─────────────────────────────────────────────────────────────────────────────

set -u

# ─── CONFIG ──────────────────────────────────────────────────────────────────
COLLECTOR="${COLLECTOR:-collect_training_data.py}"
SCENARIO="${SCENARIO:-full_sweep}"
STRESS_MODE="${STRESS_MODE:-cpu}"           # cpu | none
STRESS_HOLD_SEC="${STRESS_HOLD_SEC:-90}"    # seconds per CPU level
LOG_FILE="${LOG_FILE:-/tmp/collect_for_rf_training.log}"
CPU_WORKERS="${CPU_WORKERS:-4}"

# CPU stress levels (percent). Walks up then down so the dataset sees
# samples at every load in both directions.
CPU_LEVELS="${CPU_LEVELS:-0 25 50 75 100 75 50 25 0}"


# ─── HELPERS ─────────────────────────────────────────────────────────────────
log() {
    local msg="$1"
    local ts
    ts=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$ts] $msg" | tee -a "$LOG_FILE"
}


check_dep() {
    local cmd="$1"
    local what="$2"
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "[ERROR] '$cmd' not found ($what)." >&2
        echo "        Install with: sudo apt-get install -y $cmd" >&2
        exit 1
    fi
}


# ─── COLLECTOR ───────────────────────────────────────────────────────────────
COLLECTOR_PID=""

start_collector() {
    log "Starting collector: python3 $COLLECTOR --scenario $SCENARIO"
    python3 -u "$COLLECTOR" --scenario "$SCENARIO" >> "$LOG_FILE" 2>&1 &
    COLLECTOR_PID=$!
    log "Collector pid=$COLLECTOR_PID"
}


stop_collector() {
    if [ -n "$COLLECTOR_PID" ] && kill -0 "$COLLECTOR_PID" 2>/dev/null; then
        log "Stopping collector (pid=$COLLECTOR_PID)"
        kill -INT "$COLLECTOR_PID" 2>/dev/null || true
        local waited=0
        while kill -0 "$COLLECTOR_PID" 2>/dev/null; do
            sleep 1
            waited=$(( waited + 1 ))
            if [ "$waited" -ge 5 ]; then
                log "Collector did not exit after 5s, forcing kill."
                kill -9 "$COLLECTOR_PID" 2>/dev/null || true
                break
            fi
        done
    fi
}


# ─── STRESS ──────────────────────────────────────────────────────────────────
STRESS_PID=""

start_cpu_stress() {
    local load="$1"
    if [ "$load" -eq 0 ]; then
        STRESS_PID=""
        return 0
    fi
    stress-ng --cpu "$CPU_WORKERS" --cpu-load "$load" \
              --timeout 1h >/dev/null 2>&1 &
    STRESS_PID=$!
}


stop_cpu_stress() {
    if [ -n "$STRESS_PID" ] && kill -0 "$STRESS_PID" 2>/dev/null; then
        kill "$STRESS_PID" 2>/dev/null || true
    fi
    pkill -9 stress-ng 2>/dev/null || true
    STRESS_PID=""
    sleep 1
}


# ─── SWEEP ───────────────────────────────────────────────────────────────────
sweep_cpu() {
    local total
    total=$(echo "$CPU_LEVELS" | wc -w)
    local total_sec=$(( total * STRESS_HOLD_SEC ))
    log "CPU stress sweep: ${total} levels × ${STRESS_HOLD_SEC}s = ${total_sec}s (~$((total_sec / 60)) min)"
    log "Levels: $CPU_LEVELS"

    local i=1
    for level in $CPU_LEVELS; do
        log "[cpu $i/$total] LEVEL=${level}%  (hold ${STRESS_HOLD_SEC}s)"
        stop_cpu_stress
        start_cpu_stress "$level"
        sleep "$STRESS_HOLD_SEC"
        i=$(( i + 1 ))
    done

    stop_cpu_stress
    log "CPU sweep finished."
}


# ─── CLEANUP ─────────────────────────────────────────────────────────────────
cleanup() {
    log "Cleaning up..."
    stop_cpu_stress
    stop_collector
    log "Done. Log saved to $LOG_FILE"
    exit 0
}


# ─── MAIN ────────────────────────────────────────────────────────────────────
check_dep python3 "running the collector"

if [ "$STRESS_MODE" = "cpu" ]; then
    check_dep stress-ng "CPU stress generation"
fi

if [ ! -f "$COLLECTOR" ]; then
    echo "[ERROR] Collector not found: $COLLECTOR" >&2
    echo "        Run from the directory that contains it, or pass" >&2
    echo "        COLLECTOR=/full/path/to/collect_training_data.py" >&2
    exit 1
fi

trap cleanup EXIT INT TERM

# Banner
log "═════════════════════════════════════════════════════════"
log "  collect_for_rf_training.sh — starting"
log "═════════════════════════════════════════════════════════"
log "  Collector:        $COLLECTOR"
log "  Scenario label:   $SCENARIO"
log "  Stress mode:      $STRESS_MODE"
[ "$STRESS_MODE" = "cpu" ] && log "  CPU levels:       $CPU_LEVELS"
[ "$STRESS_MODE" = "cpu" ] && log "  Hold per level:   ${STRESS_HOLD_SEC}s"
log "  Log file:         $LOG_FILE"
log "═════════════════════════════════════════════════════════"
log ""
log "  ▶ Make sure full_sweep_server.sh is running on the SERVER,"
log "    then press Enter here to start collection."
read -r _ 2>/dev/null || true

start_collector
sleep 5

if ! kill -0 "$COLLECTOR_PID" 2>/dev/null; then
    log "[ERROR] Collector died during warmup. Check $LOG_FILE for the error."
    exit 1
fi
log "Collector is running. Starting sweep..."

case "$STRESS_MODE" in
    cpu)
        sweep_cpu
        ;;
    none)
        log "STRESS_MODE=none → collector runs solo until Ctrl+C."
        wait "$COLLECTOR_PID"
        ;;
    *)
        echo "[ERROR] Unknown STRESS_MODE '$STRESS_MODE'. Use cpu|none." >&2
        exit 1
        ;;
esac

log "Sweep finished. Letting collector run for 30 more seconds before exit..."
sleep 30
