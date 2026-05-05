#!/usr/bin/env python3
# =============================================================================
# adaptive_engine.py — NeuroEdgeFlow Sprint 5
# Adaptive Offloading Engine
#
# Sprint 5 Task 04 — Hysteresis added:
#   Engine requires HYSTERESIS_COUNT (3) consecutive same-direction signals
#   before executing a mode switch. A single outlier poll cannot cause a flip.
#
# Python 3.6 compatible — Jetson TX2, JetPack R32.7.6
# Run: OPENBLAS_CORETYPE=ARMV8 python3 ~/adaptive_engine.py
# =============================================================================

from __future__ import print_function

import csv
import logging
import os
import signal
import subprocess
import sys
import threading
import time

try:
    from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logging.warning("prometheus_client not available — Pushgateway push disabled")

try:
    from network_monitor import get_current_conditions, start as start_network_monitor
    NETWORK_MONITOR_AVAILABLE = True
except ImportError:
    NETWORK_MONITOR_AVAILABLE = False
    logging.warning("network_monitor.py not found — using stub conditions")

try:
    from jetson_exporter import set_fps
    JETSON_EXPORTER_AVAILABLE = True
except ImportError:
    JETSON_EXPORTER_AVAILABLE = False
    logging.warning("jetson_exporter.py not found — set_fps() calls will be skipped")

try:
    from neural_regulator import NeuralOffloadingRegulator
    NEURAL_REGULATOR_AVAILABLE = True
except ImportError:
    NEURAL_REGULATOR_AVAILABLE = False
    logging.warning("neural_regulator.py not found — using legacy decision logic")

# =============================================================================
# CONFIGURATION
# =============================================================================

LAPTOP_IP           = "192.168.55.100"      # laptop endpoint
PUSHGATEWAY_URL     = None                # Set to IP:9091 if Pushgateway is running
PUSHGATEWAY_JOB     = "adaptive_engine"

# Cloud Health endpoint (Triton Server)
CLOUD_HEALTH_URL    = "http://" + LAPTOP_IP + ":8000/v2/health/ready"

POLL_INTERVAL_S     = 2.0

# ---------------------------------------------------------------------------
# HYSTERESIS — Task 04
# ---------------------------------------------------------------------------
# Require this many consecutive same-direction polls before switching modes.
#
# Value = 3, derived from:
#   - Poll interval = 2 s  →  3 polls = 6 s confirmation window
#   - Sprint 3: RTT spikes under +50ms NetEm lasted ~5 s before stabilising
#   - Sprint 4 Task 8: real degradation events lasted > 30 s (well above 6 s)
#   - 6 s window filters transient single-poll outliers without masking real
#     degradation events, and without adding perceptible lag on genuine switches
#   - Sprint 4 confirmed instant recovery → 6 s confirmation lag is acceptable
#
# Change ONLY this constant to adjust — all logic reads from it.
# ---------------------------------------------------------------------------
HYSTERESIS_COUNT    = 3

LOG_FILE            = os.path.expanduser("~/adaptive_engine.csv")
LOG_LEVEL           = logging.DEBUG

# ---------------------------------------------------------------------------
# FALLBACK — Task 06
# ---------------------------------------------------------------------------
# Consecutive cloud inference errors that trigger a fallback lock.
# Matches HYSTERESIS_COUNT (3) so a single bad frame cannot lock.
FALLBACK_ERROR_THRESHOLD = 3

# Consecutive healthy network polls to auto-recover from a lock.
# 5 polls = 10 s — deliberately higher than HYSTERESIS_COUNT to prevent
# thrashing immediately after a partial recovery.
FALLBACK_RECOVERY_POLLS  = 5

# Dedicated CSV — one row per LOCKED event, one row per RECOVERED event.
FALLBACK_LOG_FILE = os.path.expanduser("~/fallback_events.csv")

FALLBACK_CSV_HEADER = [
    "timestamp",
    "event",              # "LOCKED" or "RECOVERED"
    "reason",             # human-readable trigger description
    "rtt_ms",
    "bandwidth_kbps",
    "error_rate_pct",
    "inference_ms",
    "consecutive_errors", # error count at lock time (LOCKED rows)
    "recovery_polls",     # healthy polls at recovery (RECOVERED rows)
]
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# MODE SWITCH LOG — Task 07
# ---------------------------------------------------------------------------
# Dedicated CSV — one row per mode switch, nothing else.
# Every row contains the exact metric values that caused the switch.
# Unlike adaptive_engine.csv (which logs every poll), this file is
# immediately readable: filter nothing, every row is an event.
#
# trigger_type values:
#   "HYSTERESIS" — normal switch after HYSTERESIS_COUNT consecutive polls
#   "FALLBACK"   — emergency lock triggered by cloud inference failures
#   "RECOVERY"   — fallback unlock after FALLBACK_RECOVERY_POLLS healthy polls
# ---------------------------------------------------------------------------
MODE_SWITCH_LOG_FILE = os.path.expanduser("~/mode_switches.csv")

MODE_SWITCH_CSV_HEADER = [
    "timestamp",
    "previous_mode",      # mode before the switch
    "new_mode",           # mode after the switch
    "trigger_type",       # HYSTERESIS / FALLBACK / RECOVERY
    "rtt_ms",             # exact value at time of switch
    "bandwidth_kbps",     # exact value at time of switch
    "error_rate_pct",     # exact value at time of switch
    "inference_ms",       # exact value at time of switch (informational)
    "network_ok",         # boolean predicate at time of switch
    "bandwidth_ok",       # boolean predicate at time of switch
    "polls_confirmed",    # how many consecutive polls confirmed the switch
                          # (HYSTERESIS_COUNT for hysteresis, 0 for fallback,
                          #  FALLBACK_RECOVERY_POLLS for recovery)
    "reason",             # free-text reason (empty for hysteresis switches,
                          # error description for fallback/recovery)
]
# ---------------------------------------------------------------------------

CLOUD_CLIENT_SCRIPT  = os.path.expanduser("~/cloud_client.py")
EDGE_PIPELINE_SCRIPT = os.path.expanduser("~/edge_pipeline.py")

EDGE_FPS_CONSTANT   = 29.65
WEIGHTS_FILE        = os.path.expanduser("~/model_weights.json")

# =============================================================================
# MODE CONSTANTS
# =============================================================================

class InferenceMode(object):
    LOCAL  = "LOCAL"
    CLOUD  = "CLOUD"
    INT    = {"LOCAL": 0, "CLOUD": 1}


# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s  [%(levelname)-8s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger("adaptive_engine")


# =============================================================================
# PURE DECISION FUNCTION
# =============================================================================

def decide(conditions):
    """
    Pure function — no side effects. No threshold values live here.
    Thresholds are owned by network_monitor.py (single source of truth).

    Returns InferenceMode.CLOUD if offload_ok, else InferenceMode.LOCAL.
    Raises TypeError if conditions is not a dict.
    Raises KeyError if required keys are missing.
    """
    if not isinstance(conditions, dict):
        raise TypeError(
            "decide() expected dict, got {0}".format(type(conditions).__name__)
        )
    for key in ("network_ok", "bandwidth_ok"):
        if key not in conditions:
            raise KeyError(
                "conditions dict missing required key: '{0}'".format(key)
            )

    offload_ok = bool(conditions["network_ok"]) and bool(conditions["bandwidth_ok"])
    return InferenceMode.CLOUD if offload_ok else InferenceMode.LOCAL


def decide_with_detail(conditions):
    """
    Returns (mode, detail_dict) — detail carries raw signal values for logging.
    """
    mode = decide(conditions)
    detail = {
        "network_ok":    bool(conditions.get("network_ok",    False)),
        "bandwidth_ok":  bool(conditions.get("bandwidth_ok",  False)),
        "offload_ok":    mode == InferenceMode.CLOUD,
        "rtt_ms":        float(conditions.get("rtt_ms",        -1.0)),
        "bandwidth_kbps":float(conditions.get("bandwidth_kbps",  0.0)),
        "error_rate_pct":float(conditions.get("error_rate_pct",  0.0)),
        "inference_ms":  float(conditions.get("inference_ms",   -1.0)),
    }
    return mode, detail


# =============================================================================
# STUB CONDITIONS
# =============================================================================

def _stub_conditions():
    return {
        "rtt_ms": 999.0, "bandwidth_kbps": 0.0, "error_rate_pct": 100.0,
        "inference_ms": -1.0, "network_ok": False, "bandwidth_ok": False,
    }


# =============================================================================
# PUSHGATEWAY
# =============================================================================

def _push_mode_to_gateway(mode, registry=None):
    if not PROMETHEUS_AVAILABLE or PUSHGATEWAY_URL is None:
        return False
    try:
        if registry is None:
            registry = CollectorRegistry()
        g = Gauge("neuro_mode",
                  "Current inference mode: 0=LOCAL, 1=CLOUD",
                  registry=registry)
        g.set(InferenceMode.INT[mode])
        push_to_gateway(PUSHGATEWAY_URL, job=PUSHGATEWAY_JOB, registry=registry)
        return True
    except Exception as exc:
        # log.warning("Pushgateway push failed (non-fatal): %s", exc)
        return False


# =============================================================================
# CSV LOGGER
# =============================================================================

CSV_HEADER = [
    "timestamp", "poll_id",
    # Live Environment
    "rtt_ms", "bandwidth_kbps", "cpu_load", "ram_usage",
    # Temporal Features (XGBoost)
    "prev_rtt_ms", "rtt_delta", "rtt_trend", "avg_bw_last_6s",
    # Target Labels (Hindsight Generation)
    "local_latency_ms", "cloud_latency_ms", "queue_depth",
    # State tracking
    "error_rate_pct", "network_ok", "bandwidth_ok", "offload_ok", "prob",
    "raw_decision",     # what decide() returned this single poll
    "pending_mode",     # mode being accumulated toward ("" if none)
    "pending_count",    # consecutive polls accumulated so far
    "previous_mode", "current_mode",
    "transition",       # 1 if mode actually switched this tick
    "push_ok", "pipeline_action",
]


def _init_csv(filepath):
    if not os.path.exists(filepath):
        try:
            with open(filepath, "w") as f:
                csv.writer(f).writerow(CSV_HEADER)
            log.info("CSV log created: %s", filepath)
        except IOError as exc:
            log.error("Cannot create CSV log: %s", exc)


def _write_csv_row(filepath, row_dict):
    try:
        with open(filepath, "a") as f:
            csv.DictWriter(f, fieldnames=CSV_HEADER).writerow(row_dict)
    except IOError as exc:
        log.error("CSV write failed: %s", exc)


# =============================================================================
# FALLBACK CSV HELPER
# =============================================================================

def _init_fallback_csv(filepath):
    if not os.path.exists(filepath):
        try:
            with open(filepath, "w") as f:
                csv.writer(f).writerow(FALLBACK_CSV_HEADER)
            log.info("Fallback event log created: %s", filepath)
        except IOError as exc:
            log.error("Cannot create fallback CSV: %s", exc)


def _write_fallback_event(filepath, row_dict):
    try:
        with open(filepath, "a") as f:
            csv.DictWriter(f, fieldnames=FALLBACK_CSV_HEADER).writerow(row_dict)
    except IOError as exc:
        log.error("Fallback CSV write failed: %s", exc)


# =============================================================================
# MODE SWITCH CSV HELPER — Task 07
# =============================================================================

def _init_mode_switch_csv(filepath):
    if not os.path.exists(filepath):
        try:
            with open(filepath, "w") as f:
                csv.writer(f).writerow(MODE_SWITCH_CSV_HEADER)
            log.info("Mode switch log created: %s", filepath)
        except IOError as exc:
            log.error("Cannot create mode switch CSV: %s", exc)


def _write_mode_switch(filepath, row_dict):
    """
    Writes one row to the mode switch log.
    Called only when a switch actually executes — not on every poll.
    """
    try:
        with open(filepath, "a") as f:
            csv.DictWriter(f, fieldnames=MODE_SWITCH_CSV_HEADER).writerow(row_dict)
    except IOError as exc:
        log.error("Mode switch CSV write failed: %s", exc)



# =============================================================================
# PIPELINE MANAGER
# =============================================================================

class PipelineManager(object):

    STUB_MODE = not (
        os.path.exists(CLOUD_CLIENT_SCRIPT) and
        os.path.exists(EDGE_PIPELINE_SCRIPT)
    )

    def __init__(self):
        self._proc        = None
        self._active_mode = None
        self._lock        = threading.Lock()
        if self.STUB_MODE:
            log.warning(
                "PipelineManager: running in STUB mode "
                "(pipeline scripts not found — decisions logged but no real inference)"
            )

    def start(self, mode):
        with self._lock:
            self._stop_internal()
            if self.STUB_MODE:
                log.info("STUB: would start %s pipeline", mode)
                self._active_mode = mode
                return "stubbed"
            cmd = ["python3",
                   CLOUD_CLIENT_SCRIPT if mode == InferenceMode.CLOUD
                   else EDGE_PIPELINE_SCRIPT]
            return self._launch(cmd, mode)

    def stop(self):
        with self._lock:
            self._stop_internal()

    def _launch(self, cmd, mode):
        env = os.environ.copy()
        env["OPENBLAS_CORETYPE"] = "ARMV8"
        try:
            self._proc = subprocess.Popen(cmd, env=env,
                                          stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE)
            self._active_mode = mode
            log.info("Pipeline started: %s  (PID %d)", mode, self._proc.pid)
            return "started"
        except OSError as exc:
            log.error("Failed to launch %s pipeline: %s", mode, exc)
            self._proc = self._active_mode = None
            return "error"

    def _stop_internal(self):
        if self._proc is not None:
            pid = self._proc.pid
            try:
                self._proc.terminate()
                self._proc.wait()
                log.info("Pipeline stopped: %s  (PID %d)", self._active_mode, pid)
            except OSError as exc:
                log.warning("Error stopping pipeline PID %d: %s", pid, exc)
            self._proc = self._active_mode = None

    @property
    def active_mode(self):
        return self._active_mode


# =============================================================================
# ADAPTIVE ENGINE — STATE MACHINE WITH HYSTERESIS
# =============================================================================

class AdaptiveEngine(object):
    """
    Two-state machine: EDGE (LOCAL) and CLOUD.
    Initial state: EDGE (fail-safe).

    Hysteresis logic per tick:

      Case A  raw_decision == current state
                → reset pending_mode and pending_count
                → no transition

      Case B  raw_decision != current state  AND  raw_decision != pending_mode
                → new or reversed direction: set pending_mode = raw_decision,
                  pending_count = 1
                → no transition

      Case C  raw_decision != current state  AND  raw_decision == pending_mode
                → increment pending_count
                → if pending_count >= HYSTERESIS_COUNT: execute switch,
                  reset pending state
                → else: no transition yet
    """

    def __init__(self, poll_interval=POLL_INTERVAL_S, log_file=LOG_FILE,
                 hysteresis_count=HYSTERESIS_COUNT):
        self._poll_interval    = poll_interval
        self._log_file         = log_file
        self._hysteresis_count = hysteresis_count
        self._state            = InferenceMode.LOCAL
        self._poll_id          = 0
        self._running          = False
        self._thread           = None
        self._pipeline         = PipelineManager()
        self._stop_event       = threading.Event()

        # Hysteresis state
        self._pending_mode     = None
        self._pending_count    = 0

        # Fallback state — Task 06
        self._fallback_locked    = False   # True = locked to LOCAL
        self._fallback_reason    = ""      # reason string for logging
        self._recovery_count     = 0       # consecutive healthy polls since lock
        self._fallback_log_file  = FALLBACK_LOG_FILE

        # Mode switch log — Task 07
        self._switch_log_file    = MODE_SWITCH_LOG_FILE

        # Temporal Buffers (XGBoost Phase 1)
        import collections
        self._rtt_history        = collections.deque(maxlen=3)
        self._bw_history         = collections.deque(maxlen=3)
        self._queue_depth        = 0
        self._local_latency      = -1.0
        self._cloud_latency      = -1.0

        # Neural Regulator — Sprint 5 Final Task
        self._regulator = None
        
        if NEURAL_REGULATOR_AVAILABLE:
            try:
                self._regulator = NeuralOffloadingRegulator(WEIGHTS_FILE)
            except Exception as exc:
                log.error("Failed to initialize NeuralOffloadingRegulator: %s", exc)

        _init_csv(self._log_file)
        _init_fallback_csv(self._fallback_log_file)
        _init_mode_switch_csv(self._switch_log_file)

    # ── public API ─────────────────────────────────────────────────────────

    def start(self):
        log.info("=== AdaptiveEngine starting ===")
        log.info("Initial state:  %s (fail-safe)", self._state)
        log.info("Poll interval:  %.1f s", self._poll_interval)
        log.info("Hysteresis:     %d consecutive polls required to switch",
                 self._hysteresis_count)
        log.info("Switch window:  %.0f s  (%d × %.1f s)",
                 self._hysteresis_count * self._poll_interval,
                 self._hysteresis_count, self._poll_interval)
        log.info("Log file:       %s", self._log_file)
        log.info("Pushgateway:    %s", PUSHGATEWAY_URL)

        if NETWORK_MONITOR_AVAILABLE:
            try:
                start_network_monitor()
                log.info("network_monitor started")
            except Exception as exc:
                log.warning("network_monitor start failed: %s", exc)
        else:
            log.warning("network_monitor unavailable — using stub conditions")

        action  = self._pipeline.start(InferenceMode.LOCAL)
        log.info("Initial pipeline action: %s", action)

        push_ok = _push_mode_to_gateway(self._state)
        log.info("Initial Pushgateway push: %s", "ok" if push_ok else "failed (non-fatal)")

        self._update_fps(InferenceMode.LOCAL, EDGE_FPS_CONSTANT)

        self._running = True
        self._stop_event.clear()
        
        self._thread  = threading.Thread(target=self._control_loop,
                                         name="adaptive-engine-loop")
        self._thread.daemon = True
        self._thread.start()
        log.info("=== AdaptiveEngine running ===")

    def stop(self):
        log.info("=== AdaptiveEngine stopping ===")
        self._running = False
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._poll_interval + 1.0)
        self._pipeline.stop()
        log.info("=== AdaptiveEngine stopped ===")

    @property
    def current_mode(self):
        return self._state

    @property
    def pending_mode(self):
        return self._pending_mode

    @property
    def pending_count(self):
        return self._pending_count

    @property
    def fallback_locked(self):
        """True when the engine is hard-locked to LOCAL due to cloud failure."""
        return self._fallback_locked

    def trigger_fallback(self, reason, conditions=None):
        """
        Called by main_pipeline.py when consecutive cloud inference errors
        reach FALLBACK_ERROR_THRESHOLD.

        Immediately locks the engine to LOCAL mode:
          - Sets _fallback_locked = True
          - Forces state to LOCAL
          - Resets hysteresis pending state (no switch possible while locked)
          - Pushes mode=0 to Pushgateway
          - Writes a LOCKED row to the fallback events CSV
          - Logs at ERROR level so the event is clearly visible

        Args:
            reason     — human-readable string describing the failure
            conditions — optional conditions dict for logging metric context
        """
        if self._fallback_locked:
            return  # already locked — do not log duplicate events

        self._fallback_locked   = True
        self._fallback_reason   = reason
        self._recovery_count    = 0
        self._pending_mode      = None
        self._pending_count     = 0
        self._state             = InferenceMode.LOCAL

        ts = time.strftime("%Y-%m-%d %H:%M:%S")

        log.error(
            "FALLBACK LOCKED — switching to LOCAL and locking. Reason: %s",
            reason
        )

        _push_mode_to_gateway(InferenceMode.LOCAL)
        self._update_fps(InferenceMode.LOCAL, EDGE_FPS_CONSTANT)

        cond = conditions or {}
        _write_fallback_event(self._fallback_log_file, {
            "timestamp":          ts,
            "event":              "LOCKED",
            "reason":             reason,
            "rtt_ms":             "{:.2f}".format(cond.get("rtt_ms", -1.0)),
            "bandwidth_kbps":     "{:.2f}".format(cond.get("bandwidth_kbps", 0.0)),
            "error_rate_pct":     "{:.2f}".format(cond.get("error_rate_pct", 0.0)),
            "inference_ms":       "{:.2f}".format(cond.get("inference_ms", -1.0)),
            "consecutive_errors": cond.get("consecutive_errors", ""),
            "recovery_polls":     "",
        })

        # Task 07 — also write to mode switch log (CLOUD → LOCAL forced lock)
        _write_mode_switch(self._switch_log_file, {
            "timestamp":      ts,
            "previous_mode":  InferenceMode.CLOUD,
            "new_mode":       InferenceMode.LOCAL,
            "trigger_type":   "FALLBACK",
            "rtt_ms":         "{:.2f}".format(cond.get("rtt_ms", -1.0)),
            "bandwidth_kbps": "{:.2f}".format(cond.get("bandwidth_kbps", 0.0)),
            "error_rate_pct": "{:.2f}".format(cond.get("error_rate_pct", 0.0)),
            "inference_ms":   "{:.2f}".format(cond.get("inference_ms", -1.0)),
            "network_ok":     int(cond.get("network_ok", False)),
            "bandwidth_ok":   int(cond.get("bandwidth_ok", False)),
            "polls_confirmed":0,
            "reason":         reason,
        })
        
    def update_local_latency(self, latency_ms):
        """Called safely by main_pipeline.py to update inference EMA"""
        self._local_latency = latency_ms

    def update_queue_depth(self, queue_size):
        """Called safely by main_pipeline.py to report queue strain"""
        self._queue_depth = queue_size

    def _check_fallback_recovery(self, conditions):
        """
        Called every tick while fallback_locked=True.
        Counts consecutive healthy polls. After FALLBACK_RECOVERY_POLLS,
        unlocks the engine and logs a RECOVERED event.

        A poll is healthy if:
          - network_ok=True  AND  bandwidth_ok=True
          - (same predicate as offload_ok — cloud is genuinely reachable again)
        """
        network_ok   = bool(conditions.get("network_ok",   False))
        bandwidth_ok = bool(conditions.get("bandwidth_ok", False))
        cloud_healthy = network_ok and bandwidth_ok

        if not cloud_healthy:
            # Still unhealthy — reset recovery counter
            if self._recovery_count > 0:
                log.debug(
                    "Fallback recovery reset (unhealthy poll) — "
                    "was at %d/%d", self._recovery_count, FALLBACK_RECOVERY_POLLS
                )
            self._recovery_count = 0
            return

        # Healthy poll
        self._recovery_count += 1
        log.info(
            "Fallback recovery progress: %d/%d healthy polls  "
            "(rtt=%.1f ms  bw=%.1f KB/s)",
            self._recovery_count, FALLBACK_RECOVERY_POLLS,
            float(conditions.get("rtt_ms", 0.0)),
            float(conditions.get("bandwidth_kbps", 0.0))
        )

        if self._recovery_count >= FALLBACK_RECOVERY_POLLS:
            # Recovery confirmed — unlock
            self._fallback_locked  = False
            self._recovery_count   = 0
            ts = time.strftime("%Y-%m-%d %H:%M:%S")

            log.warning(
                "FALLBACK RECOVERED — engine unlocked after %d healthy polls. "
                "Returning to normal adaptive operation.",
                FALLBACK_RECOVERY_POLLS
            )

            _write_fallback_event(self._fallback_log_file, {
                "timestamp":          ts,
                "event":              "RECOVERED",
                "reason":             self._fallback_reason,
                "rtt_ms":             "{:.2f}".format(conditions.get("rtt_ms", -1.0)),
                "bandwidth_kbps":     "{:.2f}".format(conditions.get("bandwidth_kbps", 0.0)),
                "error_rate_pct":     "{:.2f}".format(conditions.get("error_rate_pct", 0.0)),
                "inference_ms":       "{:.2f}".format(conditions.get("inference_ms", -1.0)),
                "consecutive_errors": "",
                "recovery_polls":     FALLBACK_RECOVERY_POLLS,
            })

            # Task 07 — also write to mode switch log (LOCAL → CLOUD recovery)
            _write_mode_switch(self._switch_log_file, {
                "timestamp":      ts,
                "previous_mode":  InferenceMode.LOCAL,
                "new_mode":       InferenceMode.CLOUD,
                "trigger_type":   "RECOVERY",
                "rtt_ms":         "{:.2f}".format(conditions.get("rtt_ms", -1.0)),
                "bandwidth_kbps": "{:.2f}".format(conditions.get("bandwidth_kbps", 0.0)),
                "error_rate_pct": "{:.2f}".format(conditions.get("error_rate_pct", 0.0)),
                "inference_ms":   "{:.2f}".format(conditions.get("inference_ms", -1.0)),
                "network_ok":     int(conditions.get("network_ok", False)),
                "bandwidth_ok":   int(conditions.get("bandwidth_ok", False)),
                "polls_confirmed":FALLBACK_RECOVERY_POLLS,
                "reason":         self._fallback_reason,
            })
            self._fallback_reason = ""

    # ── control loop ────────────────────────────────────────────────────────

    def _get_hw_metrics(self):
        """Zero-dependency CPU/RAM fetch from /proc."""
        try:
            with open('/proc/stat','r') as f: fields = [float(c) for c in f.readline().split()[1:]]
            idle, total = fields[3], sum(fields)
            with open('/proc/meminfo','r') as f: lines = f.readlines(); t,free = float(lines[0].split()[1]), float(lines[1].split()[1])
            ram = (1.0 - free/t)*100.0
            return (idle, total, ram)
        except: return (0, 0, 0)

    def _control_loop(self):
        prev_i, prev_t, _ = self._get_hw_metrics()
        while self._running and not self._stop_event.is_set():
            try:
                # Calculate CPU delta
                curr_i, curr_t, ram = self._get_hw_metrics()
                cpu = (1.0 - (curr_i - prev_i) / (curr_t - prev_t)) * 100.0 if curr_t != prev_t else 0.0
                prev_i, prev_t = curr_i, curr_t
                
                self._tick(cpu, ram)
            except Exception as exc:
                log.error("Unhandled exception in control loop: %s", exc)
            self._stop_event.wait(timeout=self._poll_interval)

    def _tick(self, cpu_load=0.0, ram_usage=0.0):
        self._poll_id += 1
        ts = time.strftime("%Y-%m-%d %H:%M:%S")

        # ── 1. SAMPLE ────────────────────────────────────────────────────────
        if NETWORK_MONITOR_AVAILABLE:
            try:
                conditions = get_current_conditions()
            except Exception as exc:
                log.warning("get_current_conditions() failed: %s — forcing LOCAL", exc)
                conditions = _stub_conditions()
        else:
            conditions = _stub_conditions()

        # Add live hardware context
        conditions["cpu_load"] = cpu_load
        conditions["ram_usage"] = ram_usage

        # ── 2. READINESS GUARD ───────────────────────────────────────────────
        if conditions.get("rtt_ms", 0.0) == 0.0 and \
           conditions.get("bandwidth_kbps", 0.0) == 0.0:
            log.warning("network_monitor not ready yet (zero values) — forcing LOCAL")
            conditions["network_ok"]   = False
            conditions["bandwidth_ok"] = False

        # ── 3. DECIDE ────────────────────────────────────────────────────────
        if self._regulator:
            try:
                metrics_dict = {
                    "rtt_ms": float(conditions.get("rtt_ms", 34.0)),
                    "bandwidth_kbps": float(conditions.get("bandwidth_kbps", 36000.0)),
                    "cpu_load": cpu_load,
                    "ram_usage": ram_usage,
                    "local_latency_ms": self._local_latency,
                    "queue_depth": self._queue_depth,
                    "error_rate_pct": float(conditions.get("error_rate_pct", 0.0)),
                    "network_ok": conditions.get("network_ok", False),
                    "bandwidth_ok": conditions.get("bandwidth_ok", False),
                    "poll_id": self._poll_id
                }
                
                cloud_win, prob = self._regulator.predict(metrics=metrics_dict)
                raw_decision = InferenceMode.CLOUD if cloud_win else InferenceMode.LOCAL
                
                # Mock detail for CSV compatibility
                detail = {
                    "network_ok":    bool(conditions.get("network_ok", False)),
                    "bandwidth_ok":  bool(conditions.get("bandwidth_ok", False)),
                    "offload_ok":    cloud_win,
                    "prob":          prob,
                    "rtt_ms":        metrics_dict["rtt_ms"],
                    "bandwidth_kbps":metrics_dict["bandwidth_kbps"],
                    "cpu_load":      cpu_load,
                    "ram_usage":     ram_usage,
                    "error_rate_pct":metrics_dict["error_rate_pct"],
                    "inference_ms":  self._local_latency,
                }
            except Exception as exc:
                log.error("Neural regulator predict failed: %s — falling back to legacy", exc)
                raw_decision, detail = decide_with_detail(conditions)
        else:
            try:
                raw_decision, detail = decide_with_detail(conditions)
            except (KeyError, TypeError) as exc:
                log.error("decide() raised %s — forcing LOCAL", exc)
                raw_decision = InferenceMode.LOCAL
                detail = {
                    "network_ok": False, "bandwidth_ok": False, "offload_ok": False,
                    "prob": 0.0,
                    "rtt_ms": -1.0, "bandwidth_kbps": 0.0,
                    "error_rate_pct": 100.0, "inference_ms": -1.0,
                }

        previous_mode   = self._state
        transitioning   = False
        new_mode        = self._state
        push_ok         = False
        pipeline_action = "none"

        # ── 4. FALLBACK CHECK ────────────────────────────────────────────────
        # If locked, bypass hysteresis entirely. Check for recovery and stay.
        if self._fallback_locked:
            self._check_fallback_recovery(conditions)
            log.debug(
                "FALLBACK LOCKED  rtt=%.1f ms  bw=%.1f KB/s  err=%.1f%%  "
                "recovery=%d/%d",
                detail["rtt_ms"], detail["bandwidth_kbps"],
                detail["error_rate_pct"],
                self._recovery_count, FALLBACK_RECOVERY_POLLS
            )
            _write_csv_row(self._log_file, {
                "timestamp":       ts,
                "poll_id":         self._poll_id,
                "rtt_ms":          "{:.2f}".format(detail["rtt_ms"]),
                "bandwidth_kbps":  "{:.2f}".format(detail["bandwidth_kbps"]),
                "cpu_load":        "{:.2f}".format(detail.get("cpu_load", 0.0)),
                "ram_usage":       "{:.2f}".format(detail.get("ram_usage", 0.0)),
                "prev_rtt_ms":     0,
                "rtt_delta":       0,
                "rtt_trend":       0,
                "avg_bw_last_6s":  "{:.2f}".format(detail["bandwidth_kbps"]),
                "local_latency_ms":"{:.2f}".format(self._local_latency),
                "cloud_latency_ms":"{:.2f}".format(self._cloud_latency),
                "queue_depth":     self._queue_depth,
                "error_rate_pct":  "{:.2f}".format(detail["error_rate_pct"]),
                "network_ok":      int(detail["network_ok"]),
                "bandwidth_ok":    int(detail["bandwidth_ok"]),
                "offload_ok":      int(detail["offload_ok"]),
                "prob":            "{:.3f}".format(detail.get("prob", 0.0)),
                "raw_decision":    raw_decision,
                "pending_mode":    "FALLBACK",
                "pending_count":   self._recovery_count,
                "previous_mode":   previous_mode,
                "current_mode":    self._state,
                "transition":      0,
                "push_ok":         0,
                "pipeline_action": "fallback_locked",
            })
            return

        # ── 5. TEMPORAL HISTORY (XGBoost) ────────────────────────────────────
        current_rtt = float(detail["rtt_ms"])
        current_bw  = float(detail["bandwidth_kbps"])
        
        # Calculate RTT Delta & Trend
        prev_rtt_ms = self._rtt_history[-1] if len(self._rtt_history) > 0 else current_rtt
        rtt_delta = current_rtt - prev_rtt_ms
        rtt_trend = "up" if rtt_delta > 10 else ("down" if rtt_delta < -10 else "stable")
        
        # Calculate Rolling BW
        bw_sum = sum(self._bw_history) + current_bw
        avg_bw_last_6s = bw_sum / (len(self._bw_history) + 1)
        
        # Commit to History
        self._rtt_history.append(current_rtt)
        self._bw_history.append(current_bw)

        # ── 6. HYSTERESIS ────────────────────────────────────────────────────
        if raw_decision == self._state:
            # Case A — signal agrees with current state → reset
            if self._pending_count > 0:
                log.debug(
                    "Hysteresis RESET: signal returned to %s "
                    "(was pending %s for %d/%d polls)",
                    self._state, self._pending_mode,
                    self._pending_count, self._hysteresis_count
                )
            self._pending_mode  = None
            self._pending_count = 0

        elif self._pending_mode != raw_decision:
            # Case B — new direction or mid-count reversal → restart count
            if self._pending_count > 0:
                log.debug(
                    "Hysteresis DIRECTION CHANGE: was pending %s (%d/%d), "
                    "now pending %s (1/%d)",
                    self._pending_mode, self._pending_count,
                    self._hysteresis_count, raw_decision,
                    self._hysteresis_count
                )
            self._pending_mode  = raw_decision
            self._pending_count = 1
            # Check immediately — handles hysteresis_count=1 (no wait needed)
            if self._pending_count >= self._hysteresis_count:
                new_mode = self._pending_mode
                log.info(
                    "Hysteresis CONFIRMED (%d/%d): executing %s → %s  "
                    "(rtt=%.1f ms  bw=%.1f KB/s  err=%.1f%%)",
                    self._pending_count, self._hysteresis_count,
                    self._state, new_mode,
                    detail["rtt_ms"], detail["bandwidth_kbps"],
                    detail["error_rate_pct"]
                )
                transitioning       = True
                self._pending_mode  = None
                self._pending_count = 0
            else:
                log.info(
                    "Hysteresis PENDING %d/%d: %s → %s  "
                    "(rtt=%.1f ms  bw=%.1f KB/s  err=%.1f%%)",
                    self._pending_count, self._hysteresis_count,
                    self._state, self._pending_mode,
                    detail["rtt_ms"], detail["bandwidth_kbps"],
                    detail["error_rate_pct"]
                )

        else:
            # Case C — same direction → increment
            self._pending_count += 1

            if self._pending_count >= self._hysteresis_count:
                # Threshold reached — confirmed switch
                new_mode = self._pending_mode
                log.info(
                    "Hysteresis CONFIRMED (%d/%d): executing %s → %s  "
                    "(rtt=%.1f ms  bw=%.1f KB/s  err=%.1f%%)",
                    self._pending_count, self._hysteresis_count,
                    self._state, new_mode,
                    detail["rtt_ms"], detail["bandwidth_kbps"],
                    detail["error_rate_pct"]
                )
                transitioning       = True
                self._pending_mode  = None
                self._pending_count = 0
            else:
                log.info(
                    "Hysteresis PENDING %d/%d: %s → %s  "
                    "(rtt=%.1f ms  bw=%.1f KB/s  err=%.1f%%)",
                    self._pending_count, self._hysteresis_count,
                    self._state, self._pending_mode,
                    detail["rtt_ms"], detail["bandwidth_kbps"],
                    detail["error_rate_pct"]
                )

        # ── 6. ACT ───────────────────────────────────────────────────────────
        if transitioning:
            pipeline_action = self._pipeline.start(new_mode)
            self._state     = new_mode
            push_ok         = _push_mode_to_gateway(new_mode)
            self._update_fps(new_mode, conditions)

            # ── Task 07: write mode switch row ───────────────────────────────
            # polls_confirmed = the pending_count value that just triggered
            # the switch. After the switch pending_count is reset to 0,
            # so we use self._hysteresis_count as the confirmed value.
            _write_mode_switch(self._switch_log_file, {
                "timestamp":      ts,
                "previous_mode":  previous_mode,
                "new_mode":       new_mode,
                "trigger_type":   "HYSTERESIS",
                "rtt_ms":         "{:.2f}".format(detail["rtt_ms"]),
                "bandwidth_kbps": "{:.2f}".format(detail["bandwidth_kbps"]),
                "error_rate_pct": "{:.2f}".format(detail["error_rate_pct"]),
                "inference_ms":   "{:.2f}".format(detail["inference_ms"]),
                "network_ok":     int(detail["network_ok"]),
                "bandwidth_ok":   int(detail["bandwidth_ok"]),
                "polls_confirmed":self._hysteresis_count,
                "reason":         "",
            })
            log.info(
                "MODE SWITCH logged: %s → %s  "
                "(rtt=%.1f ms  bw=%.1f KB/s  err=%.1f%%)",
                previous_mode, new_mode,
                detail["rtt_ms"], detail["bandwidth_kbps"],
                detail["error_rate_pct"]
            )
        else:
            if self._state == InferenceMode.CLOUD and self._pending_count == 0:
                self._update_fps(InferenceMode.CLOUD, conditions)
            if self._pending_count == 0:
                log.debug(
                    "STAY %-5s  rtt=%.1f ms  bw=%.1f KB/s  err=%.1f%%  "
                    "network_ok=%s  bw_ok=%s",
                    self._state,
                    detail["rtt_ms"], detail["bandwidth_kbps"],
                    detail["error_rate_pct"],
                    detail["network_ok"], detail["bandwidth_ok"]
                )

        # ── 7. LOG ───────────────────────────────────────────────────────────
        _write_csv_row(self._log_file, {
            "timestamp":       ts,
            "poll_id":         self._poll_id,
            "rtt_ms":          "{:.2f}".format(detail["rtt_ms"]),
            "bandwidth_kbps":  "{:.2f}".format(detail["bandwidth_kbps"]),
            "cpu_load":        "{:.2f}".format(detail.get("cpu_load", 0.0)),
            "ram_usage":       "{:.2f}".format(detail.get("ram_usage", 0.0)),
            "prev_rtt_ms":     "{:.2f}".format(prev_rtt_ms),
            "rtt_delta":       "{:.2f}".format(rtt_delta),
            "rtt_trend":       rtt_trend,
            "avg_bw_last_6s":  "{:.2f}".format(avg_bw_last_6s),
            "local_latency_ms":"{:.2f}".format(self._local_latency),
            "cloud_latency_ms":"{:.2f}".format(self._cloud_latency),
            "queue_depth":     self._queue_depth,
            "error_rate_pct":  "{:.2f}".format(detail["error_rate_pct"]),
            "network_ok":      int(detail["network_ok"]),
            "bandwidth_ok":    int(detail["bandwidth_ok"]),
            "offload_ok":      int(detail["offload_ok"]),
            "prob":            "{:.3f}".format(detail.get("prob", 0.0)),
            "raw_decision":    raw_decision,
            "pending_mode":    self._pending_mode if self._pending_mode else "",
            "pending_count":   self._pending_count,
            "previous_mode":   previous_mode,
            "current_mode":    self._state,
            "transition":      int(transitioning),
            "push_ok":         int(push_ok),
            "pipeline_action": pipeline_action,
        })

    # ── FPS update ───────────────────────────────────────────────────────────

    def _update_fps(self, mode, conditions_or_fps=None):
        if not JETSON_EXPORTER_AVAILABLE:
            return
        try:
            if mode == InferenceMode.LOCAL:
                fps = EDGE_FPS_CONSTANT
            elif isinstance(conditions_or_fps, dict):
                ms = float(conditions_or_fps.get("inference_ms", 0.0) or 0.0)
                fps = min(1000.0 / ms, 10.0) if ms > 0 else 0.0
            elif isinstance(conditions_or_fps, (int, float)):
                fps = float(conditions_or_fps)
            else:
                fps = 0.0
            set_fps(fps)
        except Exception as exc:
            log.warning("set_fps() failed (non-fatal): %s", exc)


# =============================================================================
# SIGNAL HANDLING
# =============================================================================

_engine_instance = None


def _signal_handler(signum, frame):
    log.info("Signal %d received — initiating graceful shutdown", signum)
    if _engine_instance is not None:
        _engine_instance.stop()
    sys.exit(0)


# =============================================================================
# MAIN
# =============================================================================

def main():
    global _engine_instance
    signal.signal(signal.SIGINT,  _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    engine = AdaptiveEngine(
        poll_interval=POLL_INTERVAL_S,
        log_file=LOG_FILE,
        hysteresis_count=1
    )
    _engine_instance = engine
    engine.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()


if __name__ == "__main__":
    main()
