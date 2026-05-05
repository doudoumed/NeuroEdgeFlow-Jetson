#!/usr/bin/env python3
"""
jetson_exporter.py — NeuroEdgeFlow Sprint 4 Jetson hardware metrics exporter.

Reads GPU%, CPU%, temperature, RAM, power, and FPS from sysfs/procfs.
Exposes them as Prometheus gauges on port 9102.

All reads are from Linux kernel interfaces — no extra packages required
beyond prometheus_client.

Python 3.6 compatible — Jetson TX2 constraint.
  - No f-strings with = (3.8+)
  - No capture_output=True (3.7+)
  - All subprocess stdout is bytes — must .decode('utf-8')

Metric names are FIXED — Sprint 5 and Grafana dashboard depend on them:
  neuro_gpu_percent, neuro_cpu_percent, neuro_temp_celsius,
  neuro_ram_used_mb, neuro_power_mw, neuro_fps

Port: 9102
  9100 = network_monitor.py
  9101 = reserved for future use
  9102 = jetson_exporter.py (this file)
"""

import os
import time
import glob
import threading
from prometheus_client import Gauge, start_http_server

# ─────────────────────────────────────────────
# Prometheus Gauges
# Declared at module level — one instance per metric, shared across threads.
# Gauge type: all these values go up AND down over time.
# ─────────────────────────────────────────────

PROM_GPU_PERCENT = Gauge(
    'neuro_gpu_percent',
    'Jetson TX2 GPU utilization percentage (0-100)'
)

PROM_CPU_PERCENT = Gauge(
    'neuro_cpu_percent',
    'Jetson TX2 CPU utilization percentage (0-100), averaged across all cores'
)

PROM_TEMP_CELSIUS = Gauge(
    'neuro_temp_celsius',
    'Jetson TX2 CPU/GPU junction temperature in Celsius'
)

PROM_RAM_USED_MB = Gauge(
    'neuro_ram_used_mb',
    'Jetson TX2 RAM used in megabytes'
)

PROM_POWER_MW = Gauge(
    'neuro_power_mw',
    'Jetson TX2 total board power consumption in milliwatts'
)

PROM_FPS = Gauge(
    'neuro_fps',
    'Current inference pipeline frames per second'
)

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

EXPORT_INTERVAL = 2.0   # Seconds between hardware reads (matches Sprint 5 cycle)
PROMETHEUS_PORT = 9102  # Port for this exporter

# ─────────────────────────────────────────────
# Hardware readers — each returns a float
# All paths are standard Jetson TX2 sysfs/procfs locations.
# ─────────────────────────────────────────────

def _read_gpu_percent():
    """
    Read GPU utilization from Jetson sysfs.
    Path: /sys/devices/gpu.0/load
    Returns a value 0-1000 representing 0-100% (divide by 10).
    Returns 0.0 if the file is not readable (GPU idle or path changed).
    """
    path = '/sys/devices/gpu.0/load'
    try:
        with open(path, 'r') as f:
            raw = int(f.read().strip())
        return raw / 10.0   # 0-1000 → 0.0-100.0
    except Exception as e:
        print('[JetsonExporter] GPU read failed: {}'.format(e))
        return 0.0


def _read_cpu_percent():
    """
    Read CPU utilization from /proc/stat.
    Computes the delta between two reads 200ms apart to get a real usage %.
    This is the same method used by 'top' and 'htop'.

    /proc/stat line format:
    cpu  user nice system idle iowait irq softirq steal guest guest_nice

    CPU% = 100 * (total_delta - idle_delta) / total_delta
    """
    def _read_stat():
        with open('/proc/stat', 'r') as f:
            line = f.readline()   # first line = aggregate across all cores
        parts = line.split()
        values = [int(x) for x in parts[1:]]
        total = sum(values)
        idle  = values[3]   # index 3 = idle time
        return total, idle

    try:
        total1, idle1 = _read_stat()
        time.sleep(0.2)
        total2, idle2 = _read_stat()

        total_delta = total2 - total1
        idle_delta  = idle2  - idle1

        if total_delta == 0:
            return 0.0

        cpu_percent = 100.0 * (total_delta - idle_delta) / total_delta
        return round(cpu_percent, 1)
    except Exception as e:
        print('[JetsonExporter] CPU read failed: {}'.format(e))
        return 0.0


def _read_temp_celsius():
    """
    Read temperature from Jetson thermal zones.
    The Jetson TX2 exposes multiple thermal zones under:
      /sys/devices/virtual/thermal/thermal_zone*/temp

    Each file contains temperature in millidegrees Celsius (e.g. 45000 = 45°C).
    We take the maximum across all zones — this represents the hottest component
    (relevant for thermal throttling detection in the dissertation).
    Returns 0.0 if no thermal zones are found.
    """
    paths = glob.glob('/sys/devices/virtual/thermal/thermal_zone*/temp')
    if not paths:
        print('[JetsonExporter] No thermal zone files found.')
        return 0.0

    temps = []
    for path in paths:
        try:
            with open(path, 'r') as f:
                millideg = int(f.read().strip())
            temps.append(millideg / 1000.0)   # millidegrees → degrees
        except Exception:
            continue

    if not temps:
        return 0.0

    return round(max(temps), 1)   # report hottest zone


def _read_ram_used_mb():
    """
    Read RAM usage from /proc/meminfo.
    RAM used = MemTotal - MemAvailable
    MemAvailable accounts for reclaimable cache — more accurate than MemFree alone.
    Returns value in megabytes.
    """
    try:
        mem = {}
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(':')
                    mem[key] = int(parts[1])   # value is in kB

        total_kb     = mem.get('MemTotal',     0)
        available_kb = mem.get('MemAvailable', 0)
        used_kb      = total_kb - available_kb
        return round(used_kb / 1024.0, 1)   # kB → MB
    except Exception as e:
        print('[JetsonExporter] RAM read failed: {}'.format(e))
        return 0.0


def _read_power_mw():
    """
    Read total board power from INA3221 power monitor via sysfs.

    On this Jetson TX2 (JetPack R32.7.6) the INA3221 is at:
      /sys/devices/3160000.i2c/i2c-0/*/iio:device*/in_power[0-9]_input

    The [0-9] before _input is critical — it excludes trigger files:
      in_power0_trigger_input  <- wrong, excluded by pattern
      in_power0_input          <- correct, matched by pattern

    Each file returns milliwatts for one power rail.
    We sum all rails across all INA3221 devices for total board power.

    Permissions fix (run once on Jetson):
      sudo chmod a+r /sys/devices/3160000.i2c/i2c-0/*/iio:device*/in_power[0-9]_input
    """
    pattern = '/sys/devices/3160000.i2c/i2c-0/*/iio:device*/in_power[0-9]_input'
    power_files = glob.glob(pattern)

    if not power_files:
        print('[JetsonExporter] Power files not found at: {}'.format(pattern))
        return 0.0

    total_mw = 0.0
    for path in power_files:
        try:
            with open(path, 'r') as f:
                total_mw += float(f.read().strip())
        except Exception as e:
            print('[JetsonExporter] Power read failed for {}: {}'.format(path, e))
            continue

    return round(total_mw, 1)


# ─────────────────────────────────────────────
# FPS — set externally by cloud_client.py
# ─────────────────────────────────────────────

_fps_lock = threading.Lock()
_current_fps = 0.0


def set_fps(fps):
    """
    Called by cloud_client.py after each frame is processed.
    Updates the FPS gauge immediately without waiting for the next loop.

    Usage:
        import jetson_exporter
        jetson_exporter.set_fps(29.65)
    """
    global _current_fps
    with _fps_lock:
        _current_fps = float(fps)
    PROM_FPS.set(fps)


# ─────────────────────────────────────────────
# Exporter loop
# ─────────────────────────────────────────────

def _export_loop():
    """
    Background thread: reads all hardware metrics every EXPORT_INTERVAL seconds
    and updates the corresponding Prometheus gauges.
    """
    print('[JetsonExporter] Export loop started. Interval: {}s'.format(EXPORT_INTERVAL))

    while True:
        loop_start = time.time()

        # ── Read all hardware metrics ──────────────────────────────────
        gpu_pct  = _read_gpu_percent()
        cpu_pct  = _read_cpu_percent()   # includes 200ms internal sleep
        temp_c   = _read_temp_celsius()
        ram_mb   = _read_ram_used_mb()
        power_mw = _read_power_mw()

        with _fps_lock:
            fps = _current_fps

        # ── Push to Prometheus gauges ──────────────────────────────────
        PROM_GPU_PERCENT.set(gpu_pct)
        PROM_CPU_PERCENT.set(cpu_pct)
        PROM_TEMP_CELSIUS.set(temp_c)
        PROM_RAM_USED_MB.set(ram_mb)
        PROM_POWER_MW.set(power_mw)
        PROM_FPS.set(fps)

        print(
            '[JetsonExporter] GPU={:.1f}%  CPU={:.1f}%  '
            'Temp={:.1f}C  RAM={:.0f}MB  Power={:.0f}mW  FPS={:.1f}'.format(
                gpu_pct, cpu_pct, temp_c, ram_mb, power_mw, fps
            )
        )

        # ── Sleep remainder of interval ────────────────────────────────
        elapsed    = time.time() - loop_start
        sleep_time = max(0.0, EXPORT_INTERVAL - elapsed)
        time.sleep(sleep_time)


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def start(prometheus_port=PROMETHEUS_PORT):
    """
    Start the Jetson hardware exporter.
    1. Launches Prometheus HTTP server on prometheus_port.
    2. Starts the background hardware polling thread.
    Call once at application startup.
    """
    start_http_server(prometheus_port)
    print('[JetsonExporter] Prometheus endpoint → http://0.0.0.0:{}/metrics'.format(
        prometheus_port))

    t = threading.Thread(target=_export_loop, daemon=True)
    t.start()


# ─────────────────────────────────────────────
# Standalone execution
# ─────────────────────────────────────────────

if __name__ == '__main__':
    start(prometheus_port=PROMETHEUS_PORT)
    print('[JetsonExporter] Running standalone. Press Ctrl+C to stop.')
    print('[JetsonExporter] Metrics at http://192.168.1.11:{}/metrics'.format(PROMETHEUS_PORT))
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        print('\n[JetsonExporter] Stopped.')
