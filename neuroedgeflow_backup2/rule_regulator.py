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
# the dissertation. The defaults match the values used elsewhere in the
# project (network_monitor.py, adaptive_engine.py) so the rule baseline does
# not silently disagree with the rest of the system.
RTT_MAX_MS               = 200.0   # cloud RTT above this is too slow
ERR_MAX_PCT              = 5.0     # packet loss above this is unreliable
BW_MIN_KBPS              = 100.0   # bandwidth below this is starvation
CPU_OFFLOAD_THRESHOLD    = 60.0    # local CPU above this -> worth offloading
LATENCY_MARGIN_MS        = 10.0    # cloud must beat local by this much to win


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
        print("[RuleRegulator] Rule-based baseline loaded "
              "(RTT<{:.0f}ms, BW>{:.0f}KBps, err<{:.0f}%, "
              "cpu_offload>{:.0f}%, margin={:.0f}ms)".format(
                  RTT_MAX_MS, BW_MIN_KBPS, ERR_MAX_PCT,
                  CPU_OFFLOAD_THRESHOLD, LATENCY_MARGIN_MS))

    # ── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _network_healthy(rtt, err, bw):
        """All three network thresholds must pass."""
        return (rtt < RTT_MAX_MS) and (err < ERR_MAX_PCT) and (bw > BW_MIN_KBPS)

    @staticmethod
    def _offload_worth_it(cpu, local_lat, cloud_lat):
        """
        Two independent reasons to offload:
          (a) local CPU is hot — the Jetson is the bottleneck,
          (b) recent cloud latency genuinely beats recent local latency.
        Either one is enough.
        """
        cpu_hot      = cpu > CPU_OFFLOAD_THRESHOLD
        cloud_faster = (cloud_lat > 0
                        and local_lat > 0
                        and cloud_lat < (local_lat - LATENCY_MARGIN_MS))
        return cpu_hot or cloud_faster

    @staticmethod
    def _confidence(rtt, err, bw, cpu):
        """
        Build a [0, 1] pseudo-probability that this is a CLOUD-worthy moment.
        It is a heuristic — NOT calibrated — but it lets downstream code that
        logs/uses prob (engine CSVs, threshold filtering) keep working.

        Each metric contributes a score in [0, 1] measuring how comfortably
        inside its CLOUD-favourable range it sits. The mean of the four
        scores is the overall confidence.
        """
        # Lower RTT is better. 0 ms -> 1.0; RTT_MAX -> 0.0.
        rtt_score = max(0.0, min(1.0, 1.0 - rtt / RTT_MAX_MS))
        # Lower packet loss is better.
        err_score = max(0.0, min(1.0, 1.0 - err / 100.0))
        # Higher bandwidth is better; saturates at 10x the floor.
        bw_score  = max(0.0, min(1.0, bw / (BW_MIN_KBPS * 10.0)))
        # Higher CPU load means cloud is MORE worthwhile.
        cpu_score = max(0.0, min(1.0, cpu / 100.0))
        return (rtt_score + err_score + bw_score + cpu_score) / 4.0

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
            }

        rtt_val   = float(metrics.get("rtt_ms",           0.0))
        bw_val    = float(metrics.get("bandwidth_kbps",   0.0))
        err_val   = float(metrics.get("error_rate_pct",   0.0))
        cpu_val   = float(metrics.get("cpu_load",         0.0))
        local_lat = float(metrics.get("local_latency_ms", -1.0))
        cloud_lat = float(metrics.get("cloud_latency_ms", -1.0))

        net_ok      = self._network_healthy(rtt_val, err_val, bw_val)
        worth_it    = self._offload_worth_it(cpu_val, local_lat, cloud_lat)
        cloud_win   = net_ok and worth_it
        confidence  = self._confidence(rtt_val, err_val, bw_val, cpu_val)

        # The engine compares `prob > 0.5` to derive cloud_win. Mirror the
        # confidence below/above 0.5 so the engine's downstream logic always
        # agrees with this rule's decision, regardless of the raw score.
        if cloud_win:
            prob = max(0.5 + 1e-3, confidence)
        else:
            prob = min(0.5 - 1e-3, confidence)

        # Same log line shape as NeuralOffloadingRegulator, so existing log
        # parsers do not need to change.
        mode_str = "CLOUD" if cloud_win else "LOCAL"
        reason   = self._explain(rtt_val, err_val, bw_val, cpu_val,
                                 local_lat, cloud_lat, net_ok, worth_it)
        print("[RuleRegulator] RTT:{:3.0f}ms CPU:{:4.1f}% -> Mode: {} "
              "(p={:.3f})  [{}]".format(
                  rtt_val, cpu_val, mode_str, prob, reason))
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
    def _explain(rtt, err, bw, cpu, local_lat, cloud_lat, net_ok, worth_it):
        """One short human-readable reason string for the decision log."""
        if not net_ok:
            if rtt >= RTT_MAX_MS:
                return "rtt {:.0f}ms >= {:.0f}".format(rtt, RTT_MAX_MS)
            if err >= ERR_MAX_PCT:
                return "loss {:.1f}% >= {:.1f}".format(err, ERR_MAX_PCT)
            if bw <= BW_MIN_KBPS:
                return "bw {:.0f}KBps <= {:.0f}".format(bw, BW_MIN_KBPS)
            return "network unhealthy"
        if not worth_it:
            return "cpu {:.0f}% < {:.0f} and cloud not faster".format(
                cpu, CPU_OFFLOAD_THRESHOLD)
        if cpu > CPU_OFFLOAD_THRESHOLD:
            return "cpu hot ({:.0f}% > {:.0f})".format(
                cpu, CPU_OFFLOAD_THRESHOLD)
        return "cloud faster ({:.0f}ms < {:.0f}ms - {:.0f})".format(
            cloud_lat, local_lat, LATENCY_MARGIN_MS)


# ─── Standalone smoke test ───────────────────────────────────────────────────
if __name__ == "__main__":
    print("Rule regulator smoke test:\n")
    r = RuleBasedRegulator()

    cases = [
        ("ideal cloud conditions, hot CPU",
            {"rtt_ms": 20, "bandwidth_kbps": 5000, "error_rate_pct": 0,
             "cpu_load": 80, "local_latency_ms": 30, "cloud_latency_ms": 40}),
        ("ideal cloud conditions, idle CPU, cloud no faster",
            {"rtt_ms": 20, "bandwidth_kbps": 5000, "error_rate_pct": 0,
             "cpu_load": 10, "local_latency_ms": 30, "cloud_latency_ms": 35}),
        ("high RTT — must stay LOCAL",
            {"rtt_ms": 300, "bandwidth_kbps": 5000, "error_rate_pct": 0,
             "cpu_load": 90, "local_latency_ms": 30, "cloud_latency_ms": 40}),
        ("packet loss — must stay LOCAL",
            {"rtt_ms": 30, "bandwidth_kbps": 5000, "error_rate_pct": 20,
             "cpu_load": 90, "local_latency_ms": 30, "cloud_latency_ms": 40}),
        ("low bandwidth — must stay LOCAL",
            {"rtt_ms": 30, "bandwidth_kbps": 50, "error_rate_pct": 0,
             "cpu_load": 90, "local_latency_ms": 30, "cloud_latency_ms": 40}),
        ("good network, idle CPU, but cloud is genuinely faster -> CLOUD",
            {"rtt_ms": 30, "bandwidth_kbps": 5000, "error_rate_pct": 0,
             "cpu_load": 10, "local_latency_ms": 100, "cloud_latency_ms": 40}),
    ]
    for desc, m in cases:
        print("  case: " + desc)
        r.predict(metrics=m)
        print()
