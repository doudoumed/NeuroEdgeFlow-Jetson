#!/usr/bin/env python3
"""
network_monitor.py — NeuroEdgeFlow Sprint 3 sensing layer + Sprint 4 Prometheus instrumentation.

Measures RTT, bandwidth, inference time, and error rate in a background thread.
Exposes a Prometheus /metrics HTTP endpoint on port 9100.
Sprint 5 reads get_current_conditions() every 2 seconds to make offload decisions.

Metric names are FIXED — Sprint 5 depends on them:
  neuro_rtt_ms, neuro_bandwidth_mbps, neuro_inference_ms, neuro_error_rate

Python 3.6 compatible — Jetson TX2 constraint.
  - No capture_output=True (added in 3.7) — use stdout=PIPE, stderr=PIPE
  - No f-string = expressions (added in 3.8)
  - subprocess stdout returns bytes — must .decode('utf-8')
"""

import threading
import time
import subprocess
import socket
import csv
import os
from datetime import datetime

# [SPRINT 4] Prometheus client imports
from prometheus_client import Gauge, start_http_server

# ─────────────────────────────────────────────
# [SPRINT 4] Prometheus Gauges — declared at module level.
#
# Gauge is the correct type: these values go up AND down over time.
# Counter would be wrong (counters only increase).
# Summary/Histogram add unnecessary complexity for scalar sensor readings.
#
# FIXED metric names — Sprint 5 will query these exact strings.
# ─────────────────────────────────────────────

PROM_RTT_MS = Gauge(
    'neuro_rtt_ms',
    'Round-trip time to cloud server in milliseconds'
)

PROM_BANDWIDTH_MBPS = Gauge(
    'neuro_bandwidth_mbps',
    'Estimated network bandwidth to cloud server in Mbps'
)

PROM_INFERENCE_MS = Gauge(
    'neuro_inference_ms',
    'Last cloud inference round-trip time in milliseconds'
)

PROM_ERROR_RATE = Gauge(
    'neuro_error_rate',
    'Network error rate as a percentage (0-100)'
)

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

CLOUD_IP         = '10.0.20.10'   # Server running Triton
PING_COUNT       = 4                # Pings per measurement cycle
MONITOR_INTERVAL = 2.0              # Seconds between measurements (matches Sprint 5 decision cycle)
CSV_LOG_PATH     = os.path.expanduser('~/network_monitor.csv')

# Decision thresholds — used by get_current_conditions()
RTT_THRESHOLD_MS    = 200.0   # Above this → network_ok = False
ERROR_THRESHOLD_PCT = 5.0     # Above this → network_ok = False
BANDWIDTH_THRESHOLD = 100.0   # KB/s minimum → below this → bandwidth_ok = False

# ─────────────────────────────────────────────
# Shared state — written by monitor thread, read by get_current_conditions()
# ─────────────────────────────────────────────

_state_lock = threading.Lock()
_state = {
    'rtt_ms':          0.0,
    'bandwidth_kbps':  0.0,
    'inference_ms':    0.0,
    'error_rate_pct':  0.0,
    'network_ok':      False,
    'bandwidth_ok':    False,
    'last_updated':    None,
}


def _ping_rtt(host, count=4):
    """
    Ping host and return (avg_rtt_ms, error_rate_pct).

    Python 3.6 note: capture_output=True was added in Python 3.7.
    Jetson TX2 runs Python 3.6 — must use stdout=PIPE, stderr=PIPE explicitly.
    subprocess stdout returns bytes in 3.6 — must .decode('utf-8').
    """
    try:
        result = subprocess.run(
            ['ping', '-c', str(count), '-W', '1', host],
            stdout=subprocess.PIPE,   # Python 3.6 compatible (no capture_output)
            stderr=subprocess.PIPE,   # Python 3.6 compatible
            timeout=10
        )
        output = result.stdout.decode('utf-8')   # bytes → str

        # Parse packet loss
        # Target line: "4 packets transmitted, 4 received, 0% packet loss, time 3004ms"
        loss_pct = 0.0
        for line in output.splitlines():
            if 'packet loss' in line:
                parts = line.split()
                for i, p in enumerate(parts):
                    if p == 'packet' and i > 0:
                        try:
                            loss_pct = float(parts[i - 1].replace('%', ''))
                        except ValueError:
                            pass

        # Parse avg RTT
        # Target line: "rtt min/avg/max/mdev = 2.483/32.857/74.378/31.318 ms"
        avg_rtt = 0.0
        for line in output.splitlines():
            if 'rtt min/avg/max' in line or 'round-trip' in line:
                try:
                    stats = line.split('=')[1].strip().split('/')
                    avg_rtt = float(stats[1])
                except (IndexError, ValueError):
                    pass

        return avg_rtt, loss_pct

    except subprocess.TimeoutExpired:
        return 0.0, 100.0
    except Exception as e:
        print('[NetworkMonitor] _ping_rtt error: {}'.format(e))
        return 0.0, 100.0


def _estimate_bandwidth_kbps(host):
    """
    Estimate bandwidth by timing a TCP transfer to Triton HTTP port 8000.
    Sends an 8 KB payload and measures how long the socket accepts it.
    Returns 0.0 gracefully if Triton is not running — does not crash the monitor.
    """
    payload_size = 8192   # 8 KB test payload
    try:
        payload = b'X' * payload_size
        start = time.time()
        with socket.create_connection((host, 8000), timeout=3) as s:
            s.sendall(payload)
        elapsed = time.time() - start
        if elapsed > 0:
            return (payload_size / 1024.0) / elapsed   # KB/s
        return 0.0
    except Exception as e:
        # Non-fatal — Triton may not be running during standalone testing
        print('[NetworkMonitor] bandwidth estimate failed (Triton running?): {}'.format(e))
        return 0.0


def _write_csv_row(row):
    """Append one row to the CSV log. Creates file with header if it does not exist."""
    file_exists = os.path.isfile(CSV_LOG_PATH)
    with open(CSV_LOG_PATH, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ─────────────────────────────────────────────
# Monitor thread
# ─────────────────────────────────────────────

def _monitor_loop():
    """
    Background thread: measures network conditions every MONITOR_INTERVAL seconds.
    Updates _state dict AND Prometheus gauges in the same loop iteration.
    No separate thread needed for Prometheus — gauges are set in-place here.
    """
    print('[NetworkMonitor] Starting monitor loop → target: {}, interval: {}s'.format(
        CLOUD_IP, MONITOR_INTERVAL))

    while True:
        loop_start = time.time()

        # ── Measurements ──────────────────────────────────────────────
        rtt_ms, error_rate_pct = _ping_rtt(CLOUD_IP, PING_COUNT)
        rtt_ms = rtt_ms 
        bandwidth_kbps         = _estimate_bandwidth_kbps(CLOUD_IP)
        bandwidth_kbps = bandwidth_kbps
        bandwidth_mbps         = bandwidth_kbps / 1024.0   # KB/s → Mbps for Prometheus gauge

        # inference_ms is updated externally by cloud_client.py via set_inference_ms()
        with _state_lock:
            inference_ms = _state['inference_ms']

        # ── Decision logic ────────────────────────────────────────────
        network_ok   = (rtt_ms < RTT_THRESHOLD_MS) and (error_rate_pct < ERROR_THRESHOLD_PCT)
        bandwidth_ok = (bandwidth_kbps > BANDWIDTH_THRESHOLD)

        # ── Update shared state ───────────────────────────────────────
        with _state_lock:
            _state['rtt_ms']         = rtt_ms
            _state['bandwidth_kbps'] = bandwidth_kbps
            _state['error_rate_pct'] = error_rate_pct
            _state['network_ok']     = network_ok
            _state['bandwidth_ok']   = bandwidth_ok
            _state['last_updated']   = datetime.now().isoformat()

        # ── [SPRINT 4] Update Prometheus gauges ───────────────────────
        # Called inside existing update loop — no new thread required.
        # Gauge.set() is thread-safe in prometheus_client.
        PROM_RTT_MS.set(rtt_ms)
        PROM_BANDWIDTH_MBPS.set(bandwidth_mbps)
        PROM_INFERENCE_MS.set(inference_ms)
        PROM_ERROR_RATE.set(error_rate_pct)
        # ─────────────────────────────────────────────────────────────

        # ── CSV log ───────────────────────────────────────────────────
        _write_csv_row({
            'timestamp':      _state['last_updated'],
            'rtt_ms':         round(rtt_ms, 2),
            'bandwidth_kbps': round(bandwidth_kbps, 2),
            'bandwidth_mbps': round(bandwidth_mbps, 4),
            'inference_ms':   round(inference_ms, 2),
            'error_rate_pct': round(error_rate_pct, 2),
            'network_ok':     int(network_ok),
            'bandwidth_ok':   int(bandwidth_ok),
        })

        print(
            '[NetworkMonitor] RTT={:.1f}ms  BW={:.1f}KB/s ({:.3f}Mbps)  '
            'Err={:.1f}%  net_ok={}  bw_ok={}'.format(
                rtt_ms, bandwidth_kbps, bandwidth_mbps,
                error_rate_pct, network_ok, bandwidth_ok
            )
        )

        # ── Sleep remainder of interval ───────────────────────────────
        elapsed    = time.time() - loop_start
        sleep_time = max(0.0, MONITOR_INTERVAL - elapsed)
        time.sleep(sleep_time)


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def set_inference_ms(ms):
    """
    Called by cloud_client.py after each gRPC inference call.
    Updates inference_ms in shared state and pushes immediately to Prometheus.
    Does not wait for the next monitor loop iteration.
    """
    with _state_lock:
        _state['inference_ms'] = ms
    PROM_INFERENCE_MS.set(ms)   # [SPRINT 4] immediate update


def get_current_conditions():
    """
    Sprint 5 entry point — returns latest network conditions snapshot.
    Called every 2 seconds by the adaptive engine.

    Returns dict with keys:
        rtt_ms          (float) — average ping RTT in ms
        bandwidth_kbps  (float) — estimated bandwidth in KB/s
        inference_ms    (float) — last cloud inference time in ms
        error_rate_pct  (float) — packet loss percentage
        network_ok      (bool)  — RTT < 200ms AND error_rate < 5%
        bandwidth_ok    (bool)  — bandwidth > 100 KB/s
        last_updated    (str)   — ISO timestamp of last measurement
    """
    with _state_lock:
        return dict(_state)


def start(prometheus_port=9100):
    """
    Start the network monitor.
    1. Launches Prometheus HTTP server on prometheus_port (default 9100).
    2. Starts the background monitoring thread.
    Call once at application startup.
    """
    # [SPRINT 4] start_http_server() launches a daemon thread internally.
    # We do not manage this thread — it lives for the process lifetime.
    start_http_server(prometheus_port)
    print('[NetworkMonitor] Prometheus metrics endpoint → http://0.0.0.0:{}/metrics'.format(
        prometheus_port))

    # Daemon thread — dies automatically when the main process exits
    t = threading.Thread(target=_monitor_loop, daemon=True)
    t.start()
    print('[NetworkMonitor] Monitor thread started.')


# ─────────────────────────────────────────────
# Standalone execution
# ─────────────────────────────────────────────

if __name__ == '__main__':
    start(prometheus_port=9100)
    print('[NetworkMonitor] Running standalone. Press Ctrl+C to stop.')
    print('[NetworkMonitor] Metrics available at http://10.0.31.140:9100/metrics')
    try:
        while True:
            time.sleep(5)
            cond = get_current_conditions()
            print('[Conditions] {}'.format(cond))
    except KeyboardInterrupt:
        print('\n[NetworkMonitor] Stopped.')
