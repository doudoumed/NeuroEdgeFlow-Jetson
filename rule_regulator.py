#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
# rule_regulator.py — NeuroEdgeFlow rule-based offloading regulator
#
# A plain threshold-based replacement for neural_regulator.py. No ML, no
# trained weights — just transparent if/else logic over the same metrics.
#
# WHY this exists
# ───────────────
# The Random Forest brain is the project's main contribution. For a
# dissertation you also need a baseline to compare against: something simple,
# defensible, and easy for a reader to reproduce. This is that baseline.
# Swap it in, run the same scenarios, and the comparison shows how much (or
# how little) the learned regulator beats a hand-tuned rule.
#
# API compatibility
# ─────────────────
# Same class name and same predict() signature as NeuralOffloadingRegulator,
# so adaptive_engine.py picks this up without code changes — just rename the
# import. predict() returns (cloud_win: bool, prob: float in [0,1]) exactly
# like the RF version. The "probability" is a heuristic confidence built from
# how far each metric is past its threshold; it is NOT a calibrated prob, but
# it lets the engine's logging, hysteresis, and CSV columns work unchanged.
#
# Decision rule
# ─────────────
# CLOUD is chosen when EVERY network condition is healthy AND the local CPU
# is loaded enough that offloading is worthwhile. Any single failing
# condition forces LOCAL. The rule is intentionally conservative — when in
# doubt, stay local (the fail-safe).
#
#   network healthy  := rtt_ms < RTT_MAX AND error_rate_pct < ERR_MAX
#                       AND bandwidth_kbps > BW_MIN
#   offload worth it := cpu_load > CPU_OFFLOAD_THRESHOLD
#                       OR local_latency_ms > cloud_latency_ms + MARGIN_MS
#
#   CLOUD  iff  network healthy AND offload worth it
#   LOCAL  otherwise
# ─────────────────────────────────────────────────────────────────────────────

import os


# ─── THRESHOLDS ──────────────────────────────────────────────────────────────
# All thresholds in one place so they are easy to tune and easy to cite in
# the dissertation. The values were chosen empirically on the Jetson TX2 +
# Triton + JPEG-proxy setup used in this project; rationale is documented
# inline so reviewers can audit each choice.
#
# WHY NOT CPU LOAD?
# ─────────────────
# A naive implementation would use cpu_load as the primary offload trigger
# (e.g. cpu > 60% → CLOUD). Empirical measurements on the Jetson TX2
# showed this assumption to be invalid: TensorRT inference executes on the
# integrated GPU and is largely insensitive to CPU load up to 100%. CPU
# stress alone is therefore a poor signal — the actual bottleneck is the
# integrated GPU. We retain a high CPU threshold (85%) as a safety net for
# pathological CPU starvation that would slow JPEG preprocessing, but the
# primary triggers are now GPU-side.
#
# Network thresholds
RTT_MAX_MS               = 200.0   # cloud RTT above this is too slow
ERR_MAX_PCT              = 5.0     # packet loss above this is unreliable
BW_MIN_KBPS              = 100.0   # bandwidth below this is starvation

# Hardware thresholds — TX2-specific, GPU-aware
# These are the *real* bottlenecks on this platform. The GPU is shared
# between TensorRT inference and any other CUDA work, so heavy GPU load
# directly extends the local inference latency. Thermal throttling kicks
# in around 75-80 °C on the TX2; beyond that the GPU clocks down and
# inference latency can double.
GPU_OFFLOAD_THRESHOLD    = 70.0    # GPU load above this -> offload to cloud
GPU_THERMAL_THRESHOLD    = 75.0    # GPU temp above this -> thermal risk
CPU_OFFLOAD_THRESHOLD    = 85.0    # safety net for runaway CPU; rarely fires

# Latency comparison
# Cloud must beat local by MORE than this margin to win — otherwise the
# overhead of JPEG encode + network + server decode (~150 ms baseline)
# erases the GPU advantage. Margin is intentionally generous because
# cloud round-trip variance is high.
LATENCY_MARGIN_MS        = 30.0    # cloud must beat local by this much


class RuleBasedRegulator:
    """
    Threshold-based offloading regulator. API-compatible with
    NeuralOffloadingRegulator: same constructor signature (ignores the
    weights path), same predict() return shape.
    """

    def __init__(self, weights_path=None):
        # weights_path is accepted for drop-in compatibility but unused.
        # No model to load — just thresholds.
        self.is_rf = False        # so any code that inspects this works
        self.feature_names = []   # empty: no learned features
        print("[RuleRegulator] GPU-aware rule baseline loaded "
              "(RTT<{:.0f}ms, BW>{:.0f}KBps, err<{:.0f}%, "
              "gpu_load>{:.0f}%, gpu_temp>{:.0f}C, "
              "cpu>{:.0f}%, margin={:.0f}ms)".format(
                  RTT_MAX_MS, BW_MIN_KBPS, ERR_MAX_PCT,
                  GPU_OFFLOAD_THRESHOLD, GPU_THERMAL_THRESHOLD,
                  CPU_OFFLOAD_THRESHOLD, LATENCY_MARGIN_MS))

    # ── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _network_healthy(rtt, err, bw):
        """All three network thresholds must pass."""
        return (rtt < RTT_MAX_MS) and (err < ERR_MAX_PCT) and (bw > BW_MIN_KBPS)

    @staticmethod
    def _offload_worth_it(cpu, gpu_load, gpu_temp, local_lat, cloud_lat):
        """
        Decide whether offloading is worth the cloud overhead.

        Four independent reasons — ANY one is enough:
          (a) GPU is hot — the TX2's actual inference bottleneck.
              This is the PRIMARY trigger on Jetson hardware: TensorRT
              runs on the GPU, so heavy GPU load directly slows local
              inference. Empirically the most useful signal.
          (b) GPU thermal throttling imminent. Beyond ~75 °C the TX2
              GPU clocks itself down to protect the silicon, and local
              latency can double. Pre-emptively offloading avoids the
              cliff edge.
          (c) Safety net: runaway CPU starvation. CPU rarely affects
              TRT itself, but it does affect JPEG preprocessing on the
              cloud path AND any non-TRT host code, so 85%+ is still
              a useful tripwire.
          (d) Measured cloud latency genuinely beats measured local
              latency by a margin large enough to absorb noise. This
              path only fires once both backends have run at least
              once.
        """
        gpu_busy        = gpu_load > GPU_OFFLOAD_THRESHOLD
        gpu_overheating = gpu_temp > GPU_THERMAL_THRESHOLD
        cpu_starved     = cpu > CPU_OFFLOAD_THRESHOLD
        cloud_faster    = (cloud_lat > 0
                           and local_lat > 0
                           and cloud_lat < (local_lat - LATENCY_MARGIN_MS))
        return gpu_busy or gpu_overheating or cpu_starved or cloud_faster

    @staticmethod
    def _confidence(rtt, err, bw, cpu, gpu_load, gpu_temp):
        """
        Build a [0, 1] pseudo-probability that this is a CLOUD-worthy moment.
        It is a heuristic — NOT calibrated — but it lets downstream code that
        logs/uses prob (engine CSVs, threshold filtering) keep working.

        Each metric contributes a score in [0, 1] measuring how comfortably
        inside its CLOUD-favourable range it sits. We give GPU load and
        temperature double weight because, per the rationale above, they
        are the real bottlenecks on this hardware — CPU load is a weak
        signal here.
        """
        # Lower RTT is better. 0 ms -> 1.0; RTT_MAX -> 0.0.
        rtt_score = max(0.0, min(1.0, 1.0 - rtt / RTT_MAX_MS))
        # Lower packet loss is better.
        err_score = max(0.0, min(1.0, 1.0 - err / 100.0))
        # Higher bandwidth is better; saturates at 10x the floor.
        bw_score  = max(0.0, min(1.0, bw / (BW_MIN_KBPS * 10.0)))
        # Higher CPU load means cloud is MORE worthwhile (weak signal here).
        cpu_score = max(0.0, min(1.0, cpu / 100.0))
        # Higher GPU load means cloud is MORE worthwhile (strong signal).
        gpu_score = max(0.0, min(1.0, gpu_load / 100.0))
        # Higher GPU temperature means cloud is MORE worthwhile (strong).
        # Normalised against thermal limit + headroom (90 °C).
        temp_score = max(0.0, min(1.0, gpu_temp / 90.0))
        # Weighted average — GPU signals count double because they are
        # the real bottleneck on this platform.
        return (rtt_score + err_score + bw_score + cpu_score
                + 2.0 * gpu_score + 2.0 * temp_score) / 8.0

    # ── Public API — matches NeuralOffloadingRegulator.predict() ────────────

    def predict(self, rtt=None, bw=None, cpu=None, ram=None,
                cloud_lat=None, local_lat=None, metrics=None):
        """
        Decide CLOUD vs LOCAL for the current sample.

        Returns:
            (cloud_win: bool, prob: float in [0,1])
            - cloud_win = True  -> engine should offload to cloud
            - cloud_win = False -> engine should run locally
            - prob is the heuristic confidence (0.5 = boundary).
              When the rule chooses LOCAL the value is mirrored below 0.5
              so the engine's prob>0.5 check matches the rule's decision.
        """
        # Accept either the metrics dict (preferred — same as the RF version)
        # or the legacy positional kwargs.
        if metrics is None:
            metrics = {
                "rtt_ms":           rtt       if rtt       is not None else 0.0,
                "bandwidth_kbps":   bw        if bw        is not None else 0.0,
                "cpu_load":         cpu       if cpu       is not None else 0.0,
                "ram_usage":        ram       if ram       is not None else 0.0,
                "local_latency_ms": local_lat if local_lat is not None else -1.0,
                "cloud_latency_ms": cloud_lat if cloud_lat is not None else -1.0,
                "error_rate_pct":   0.0,
                "gpu_load":         0.0,
                "gpu_temp":         0.0,
            }

        rtt_val   = float(metrics.get("rtt_ms",           0.0))
        bw_val    = float(metrics.get("bandwidth_kbps",   0.0))
        err_val   = float(metrics.get("error_rate_pct",   0.0))
        cpu_val   = float(metrics.get("cpu_load",         0.0))
        # GPU metrics — the PRIMARY signals on Jetson. Default to 0 so that
        # callers who do not yet populate them (older code paths) still get
        # network-only behaviour rather than silent failure.
        gpu_load_val = float(metrics.get("gpu_load",      0.0))
        gpu_temp_val = float(metrics.get("gpu_temp",      0.0))
        local_lat = float(metrics.get("local_latency_ms", -1.0))
        cloud_lat = float(metrics.get("cloud_latency_ms", -1.0))

        net_ok     = self._network_healthy(rtt_val, err_val, bw_val)
        worth_it   = self._offload_worth_it(
            cpu_val, gpu_load_val, gpu_temp_val, local_lat, cloud_lat
        )
        cloud_win  = net_ok and worth_it
        confidence = self._confidence(
            rtt_val, err_val, bw_val, cpu_val, gpu_load_val, gpu_temp_val
        )

        # The engine compares `prob > 0.5` to derive cloud_win. Mirror the
        # confidence below/above 0.5 so the engine's downstream logic always
        # agrees with this rule's decision, regardless of the raw score.
        if cloud_win:
            prob = max(0.5 + 1e-3, confidence)
        else:
            prob = min(0.5 - 1e-3, confidence)

        # Same log line shape as NeuralOffloadingRegulator, so existing log
        # parsers do not need to change. We now include GPU load in the
        # default log so operators can see the primary signal.
        mode_str = "CLOUD" if cloud_win else "LOCAL"
        reason   = self._explain(rtt_val, err_val, bw_val,
                                 cpu_val, gpu_load_val, gpu_temp_val,
                                 local_lat, cloud_lat, net_ok, worth_it)
        print("[RuleRegulator] RTT:{:3.0f}ms GPU:{:4.1f}%/{:.0f}C CPU:{:4.1f}% "
              "-> Mode: {} (p={:.3f})  [{}]".format(
                  rtt_val, gpu_load_val, gpu_temp_val, cpu_val,
                  mode_str, prob, reason))
        return cloud_win, float(prob)

    def extract_features(self, metrics):
        """
        Drop-in compatibility shim for code paths that expect feature
        extraction. The rule regulator has no learned features — return an
        empty list so any caller that iterates does not crash.
        """
        return []

    # ── Diagnostic ──────────────────────────────────────────────────────────

    @staticmethod
    def _explain(rtt, err, bw, cpu, gpu_load, gpu_temp,
                 local_lat, cloud_lat, net_ok, worth_it):
        """
        One short human-readable reason string for the decision log.
        Checked in priority order: network → GPU → CPU → measured latency.
        The order matters — GPU triggers come BEFORE CPU because they are
        the more useful signal on this hardware (see threshold rationale).
        """
        if not net_ok:
            if rtt >= RTT_MAX_MS:
                return "rtt {:.0f}ms >= {:.0f}".format(rtt, RTT_MAX_MS)
            if err >= ERR_MAX_PCT:
                return "loss {:.1f}% >= {:.1f}".format(err, ERR_MAX_PCT)
            if bw <= BW_MIN_KBPS:
                return "bw {:.0f}KBps <= {:.0f}".format(bw, BW_MIN_KBPS)
            return "network unhealthy"
        if not worth_it:
            return ("gpu {:.0f}% < {:.0f} and cpu {:.0f}% < {:.0f} "
                    "and cloud not faster").format(
                        gpu_load, GPU_OFFLOAD_THRESHOLD,
                        cpu, CPU_OFFLOAD_THRESHOLD)
        # Worth-it triggers, in the same order as _offload_worth_it():
        if gpu_load > GPU_OFFLOAD_THRESHOLD:
            return "gpu hot ({:.0f}% > {:.0f})".format(
                gpu_load, GPU_OFFLOAD_THRESHOLD)
        if gpu_temp > GPU_THERMAL_THRESHOLD:
            return "gpu thermal ({:.0f}C > {:.0f})".format(
                gpu_temp, GPU_THERMAL_THRESHOLD)
        if cpu > CPU_OFFLOAD_THRESHOLD:
            return "cpu starved ({:.0f}% > {:.0f})".format(
                cpu, CPU_OFFLOAD_THRESHOLD)
        return "cloud faster ({:.0f}ms < {:.0f}ms - {:.0f})".format(
            cloud_lat, local_lat, LATENCY_MARGIN_MS)


# ─── Standalone smoke test ───────────────────────────────────────────────────
if __name__ == "__main__":
    print("GPU-aware rule regulator smoke test:\n")
    r = RuleBasedRegulator()

    cases = [
        ("ideal cloud conditions, hot GPU -> CLOUD (primary trigger)",
            {"rtt_ms": 20, "bandwidth_kbps": 5000, "error_rate_pct": 0,
             "cpu_load": 10, "gpu_load": 85, "gpu_temp": 65,
             "local_latency_ms": 30, "cloud_latency_ms": 40}),
        ("ideal cloud conditions, GPU overheating -> CLOUD (thermal)",
            {"rtt_ms": 20, "bandwidth_kbps": 5000, "error_rate_pct": 0,
             "cpu_load": 10, "gpu_load": 50, "gpu_temp": 80,
             "local_latency_ms": 30, "cloud_latency_ms": 40}),
        ("ideal cloud conditions, idle hardware -> LOCAL (cheap path)",
            {"rtt_ms": 20, "bandwidth_kbps": 5000, "error_rate_pct": 0,
             "cpu_load": 10, "gpu_load": 20, "gpu_temp": 50,
             "local_latency_ms": 30, "cloud_latency_ms": 50}),
        ("high CPU but cool GPU -> LOCAL (the key TX2 behaviour)",
            {"rtt_ms": 20, "bandwidth_kbps": 5000, "error_rate_pct": 0,
             "cpu_load": 70, "gpu_load": 20, "gpu_temp": 50,
             "local_latency_ms": 30, "cloud_latency_ms": 50}),
        ("high RTT, hot GPU -> LOCAL (network safety overrides)",
            {"rtt_ms": 300, "bandwidth_kbps": 5000, "error_rate_pct": 0,
             "cpu_load": 50, "gpu_load": 90, "gpu_temp": 70,
             "local_latency_ms": 30, "cloud_latency_ms": 40}),
        ("packet loss, hot GPU -> LOCAL (network safety overrides)",
            {"rtt_ms": 30, "bandwidth_kbps": 5000, "error_rate_pct": 20,
             "cpu_load": 50, "gpu_load": 90, "gpu_temp": 70,
             "local_latency_ms": 30, "cloud_latency_ms": 40}),
        ("CPU runaway 90% safety net -> CLOUD",
            {"rtt_ms": 30, "bandwidth_kbps": 5000, "error_rate_pct": 0,
             "cpu_load": 90, "gpu_load": 30, "gpu_temp": 55,
             "local_latency_ms": 30, "cloud_latency_ms": 50}),
        ("measured cloud genuinely faster -> CLOUD",
            {"rtt_ms": 20, "bandwidth_kbps": 5000, "error_rate_pct": 0,
             "cpu_load": 20, "gpu_load": 40, "gpu_temp": 60,
             "local_latency_ms": 200, "cloud_latency_ms": 100}),
    ]
    for desc, m in cases:
        print("  case: " + desc)
        r.predict(metrics=m)
        print()
