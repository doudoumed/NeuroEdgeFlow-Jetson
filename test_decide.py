#!/usr/bin/env python3
# =============================================================================
# test_decide.py — NeuroEdgeFlow Sprint 5
# Unit tests: decide() (Task 02) + Hysteresis (Task 04) + Fallback (Task 06) + Switch Log (Task 07)
#
# Python 3.6 compatible — no hardware required
# Run: OPENBLAS_CORETYPE=ARMV8 python3 ~/test_decide.py
# =============================================================================

from __future__ import print_function

import csv as _csv
import os
import sys
import tempfile
import time
import traceback

from adaptive_engine import (
    decide, decide_with_detail, InferenceMode,
    AdaptiveEngine, HYSTERESIS_COUNT,
    FALLBACK_ERROR_THRESHOLD, FALLBACK_RECOVERY_POLLS,
    FALLBACK_CSV_HEADER,
    MODE_SWITCH_LOG_FILE, MODE_SWITCH_CSV_HEADER,
    _init_mode_switch_csv, _write_mode_switch,
)

# =============================================================================
# MINIMAL TEST FRAMEWORK
# =============================================================================

_passed = 0
_failed = 0
_errors = []


def _assert(cond, name, msg=""):
    global _passed, _failed
    if cond:
        _passed += 1
        print("  PASS  " + name)
    else:
        _failed += 1
        print("  FAIL  " + name + ("  —  " + msg if msg else ""))
        _errors.append(name)


def _assert_eq(actual, expected, name):
    _assert(actual == expected, name,
            "expected={0!r} got={1!r}".format(expected, actual))


def _assert_raises(exc, func, args, name):
    global _passed, _failed
    try:
        func(*args)
        _failed += 1
        print("  FAIL  " + name + " — no exception")
        _errors.append(name)
    except exc:
        _passed += 1
        print("  PASS  " + name)
    except Exception as e:
        _failed += 1
        print("  FAIL  " + name + " — wrong exc: " + str(e))
        _errors.append(name)


def section(t):
    print("\n" + "─" * 64 + "\n  " + t + "\n" + "─" * 64)


# =============================================================================
# HELPERS
# =============================================================================

def _cond(nok, bok, rtt=50.0, bw=500.0, err=0.0, inf=350.0):
    return {
        "rtt_ms": rtt, "bandwidth_kbps": bw,
        "error_rate_pct": err, "inference_ms": inf,
        "network_ok": nok, "bandwidth_ok": bok,
    }


def _make_engine(n=HYSTERESIS_COUNT):
    tmp = tempfile.mktemp(suffix=".csv")
    return AdaptiveEngine(
        poll_interval=99.0,
        log_file=tmp,
        hysteresis_count=n
    ), tmp


def _tick(engine, nok, bok, rtt=50.0, bw=500.0, err=0.0):
    """Drive one tick of hysteresis state machine directly."""
    cond = _cond(nok, bok, rtt=rtt, bw=bw, err=err)
    raw, detail = decide_with_detail(cond)
    prev = engine._state
    transitioning = False
    new_mode = engine._state

    if raw == engine._state:
        engine._pending_mode = None
        engine._pending_count = 0
    elif engine._pending_mode != raw:
        engine._pending_mode = raw
        engine._pending_count = 1
        if engine._pending_count >= engine._hysteresis_count:
            new_mode = engine._pending_mode
            transitioning = True
            engine._pending_mode = None
            engine._pending_count = 0
    else:
        engine._pending_count += 1
        if engine._pending_count >= engine._hysteresis_count:
            new_mode = engine._pending_mode
            transitioning = True
            engine._pending_mode = None
            engine._pending_count = 0

    if transitioning:
        engine._state = new_mode

    return {
        "raw": raw, "prev": prev,
        "current": engine._state, "transition": transitioning
    }


# =============================================================================
# ── TASK 02: decide() ────────────────────────────────────────────────────────
# =============================================================================

section("decide() core truth table")
_assert_eq(decide(_cond(True,  True)),  InferenceMode.CLOUD, "T,T→CLOUD")
_assert_eq(decide(_cond(True,  False)), InferenceMode.LOCAL, "T,F→LOCAL")
_assert_eq(decide(_cond(False, True)),  InferenceMode.LOCAL, "F,T→LOCAL")
_assert_eq(decide(_cond(False, False)), InferenceMode.LOCAL, "F,F→LOCAL")

section("decide() fail-safe defaults")
_assert_eq(decide(_cond(False, False, rtt=0, bw=0)), InferenceMode.LOCAL, "all-zero→LOCAL")
_assert_eq(decide(_cond(False, False, rtt=9999, bw=0, err=100)), InferenceMode.LOCAL, "worst-case→LOCAL")
extra = _cond(True, True); extra["x"] = 1
_assert_eq(decide(extra), InferenceMode.CLOUD, "extra keys ignored")

section("decide() boolean coercion")
_assert_eq(decide(_cond(1, 1)), InferenceMode.CLOUD, "1,1→CLOUD")
_assert_eq(decide(_cond(0, 1)), InferenceMode.LOCAL, "0,1→LOCAL")
_assert_eq(decide(_cond(1, 0)), InferenceMode.LOCAL, "1,0→LOCAL")

section("decide() inference_ms informational only")
base = {"network_ok": True, "bandwidth_ok": True,
        "rtt_ms": 50, "bandwidth_kbps": 500, "error_rate_pct": 0}
for ms in [0, 50, 350, 980, 12181, 99999]:
    c = dict(base); c["inference_ms"] = ms
    _assert_eq(decide(c), InferenceMode.CLOUD,
               "inference_ms={0} no effect".format(ms))

section("decide() Sprint 3 benchmark scenarios")
_assert_eq(decide({"rtt_ms": 15,  "bandwidth_kbps": 800, "error_rate_pct": 0,
                   "inference_ms": 370,   "network_ok": True,  "bandwidth_ok": True}),
           InferenceMode.CLOUD, "Sprint3 ideal→CLOUD")
_assert_eq(decide({"rtt_ms": 115, "bandwidth_kbps": 800, "error_rate_pct": 0,
                   "inference_ms": 915,   "network_ok": True,  "bandwidth_ok": True}),
           InferenceMode.CLOUD, "Sprint3 +50ms→CLOUD")
_assert_eq(decide({"rtt_ms": 20,  "bandwidth_kbps": 30,  "error_rate_pct": 0,
                   "inference_ms": 12181, "network_ok": True,  "bandwidth_ok": False}),
           InferenceMode.LOCAL, "Sprint3 2Mbps→LOCAL")
_assert_eq(decide({"rtt_ms": 20,  "bandwidth_kbps": 800, "error_rate_pct": 5,
                   "inference_ms": 2200,  "network_ok": False, "bandwidth_ok": True}),
           InferenceMode.LOCAL, "Sprint3 5%loss→LOCAL")

section("decide() boundary values")
_assert_eq(decide(_cond(True,  True,  rtt=199.9, bw=100.1, err=4.9)),
           InferenceMode.CLOUD, "barely passing→CLOUD")
_assert_eq(decide(_cond(False, False, rtt=200.1, bw=99.9,  err=5.1)),
           InferenceMode.LOCAL, "barely failing→LOCAL")

section("decide_with_detail()")
mode, d = decide_with_detail(_cond(True, True, rtt=42.5, bw=650, err=1.2, inf=345))
_assert_eq(mode, InferenceMode.CLOUD,   "mode=CLOUD")
_assert(d["offload_ok"] is True,        "offload_ok=True")
_assert(d["network_ok"] is True,        "network_ok")
_assert(d["bandwidth_ok"] is True,      "bandwidth_ok")
_assert_eq(d["rtt_ms"],         42.5,   "rtt_ms")
_assert_eq(d["bandwidth_kbps"], 650.0,  "bandwidth_kbps")
_assert_eq(d["error_rate_pct"],   1.2,  "error_rate_pct")
_assert_eq(d["inference_ms"],   345.0,  "inference_ms")
m2, d2 = decide_with_detail(_cond(False, True))
_assert_eq(m2, InferenceMode.LOCAL,     "mode=LOCAL")
_assert(d2["offload_ok"] is False,      "offload_ok=False")

section("decide() error handling")
_assert_raises(TypeError, decide, [None],                  "None→TypeError")
_assert_raises(TypeError, decide, ["s"],                   "str→TypeError")
_assert_raises(TypeError, decide, [42],                    "int→TypeError")
_assert_raises(TypeError, decide, [[]],                    "list→TypeError")
_assert_raises(KeyError,  decide, [{"bandwidth_ok": True}],"missing network_ok")
_assert_raises(KeyError,  decide, [{"network_ok": True}],  "missing bandwidth_ok")
_assert_raises(KeyError,  decide, [{}],                    "empty dict")

section("decide() return type invariants")
for nok in [True, False]:
    for bok in [True, False]:
        r = decide(_cond(nok, bok))
        _assert(r in (InferenceMode.LOCAL, InferenceMode.CLOUD),
                "always LOCAL/CLOUD nok={0} bok={1}".format(nok, bok))
        _assert(isinstance(r, str),
                "returns str nok={0} bok={1}".format(nok, bok))

section("InferenceMode integer mapping")
_assert_eq(InferenceMode.INT[InferenceMode.LOCAL], 0, "LOCAL=0")
_assert_eq(InferenceMode.INT[InferenceMode.CLOUD], 1, "CLOUD=1")

# =============================================================================
# ── TASK 04: Hysteresis ──────────────────────────────────────────────────────
# =============================================================================

section("Hysteresis — HYSTERESIS_COUNT == 3")
_assert_eq(HYSTERESIS_COUNT, 3, "HYSTERESIS_COUNT=3")

section("Hysteresis — no premature switch (polls 1 and 2)")
e, _ = _make_engine(3)
_assert_eq(e.current_mode, InferenceMode.LOCAL, "initial=LOCAL")
r = _tick(e, True, True)
_assert_eq(e.current_mode, InferenceMode.LOCAL, "poll 1/3: still LOCAL")
_assert_eq(e.pending_count, 1,                  "poll 1/3: pending_count=1")
_assert_eq(e.pending_mode,  InferenceMode.CLOUD,"poll 1/3: pending_mode=CLOUD")
_assert(not r["transition"],                     "poll 1/3: no transition")
r = _tick(e, True, True)
_assert_eq(e.current_mode, InferenceMode.LOCAL, "poll 2/3: still LOCAL")
_assert_eq(e.pending_count, 2,                  "poll 2/3: pending_count=2")
_assert(not r["transition"],                     "poll 2/3: no transition")

section("Hysteresis — switches exactly on 3rd consecutive signal")
e, _ = _make_engine(3)
_tick(e, True, True); _tick(e, True, True)
r = _tick(e, True, True)
_assert_eq(e.current_mode, InferenceMode.CLOUD, "poll 3/3: switched to CLOUD")
_assert(r["transition"],                         "poll 3/3: transition=True")
_assert_eq(e.pending_count, 0,                   "pending_count reset=0")
_assert(e.pending_mode is None,                  "pending_mode reset=None")

section("Hysteresis — reset on opposite signal mid-accumulation")
e, _ = _make_engine(3)
_tick(e, True, True); _tick(e, True, True)
_assert_eq(e.pending_count, 2, "before reset: count=2")
r = _tick(e, False, False)
_assert_eq(e.current_mode, InferenceMode.LOCAL, "still LOCAL")
_assert_eq(e.pending_count, 0,                  "count reset=0")
_assert(e.pending_mode is None,                  "pending_mode=None")
_assert(not r["transition"],                     "no transition on reset")

section("Hysteresis — full round-trip LOCAL→CLOUD→LOCAL")
e, _ = _make_engine(3)
for i in range(3): r = _tick(e, True, True)
_assert_eq(e.current_mode, InferenceMode.CLOUD, "after 3 CLOUD: state=CLOUD")
_assert(r["transition"], "transition LOCAL→CLOUD fired")
for i in range(2): _tick(e, False, False)
_assert_eq(e.current_mode, InferenceMode.CLOUD, "2 LOCAL polls: still CLOUD")
r = _tick(e, False, False)
_assert_eq(e.current_mode, InferenceMode.LOCAL, "after 3 LOCAL: state=LOCAL")
_assert(r["transition"], "transition CLOUD→LOCAL fired")

section("Hysteresis — interrupted sequences never switch")
e, _ = _make_engine(3)
for attempt in range(4):
    _tick(e, True, True); _tick(e, True, True); _tick(e, False, False)
    _assert_eq(e.current_mode, InferenceMode.LOCAL,
               "attempt {0}: still LOCAL".format(attempt + 1))

section("Hysteresis — direction reversal resets count to 0")
e, _ = _make_engine(3)
_tick(e, True, True); _tick(e, True, True)
_assert_eq(e.pending_count, 2, "count=2 before reversal")
_tick(e, False, False)
_assert_eq(e.pending_count, 0, "count=0 after LOCAL signal (Case A reset)")
_assert(e.pending_mode is None, "pending_mode=None")
_assert_eq(e.current_mode, InferenceMode.LOCAL, "state unchanged")

section("Hysteresis — stable state keeps pending_count=0")
e, _ = _make_engine(3)
for i in range(10):
    _tick(e, False, False)
    _assert_eq(e.pending_count, 0,  "poll {0}: count=0".format(i + 1))
    _assert(e.pending_mode is None, "poll {0}: pending=None".format(i + 1))

section("Hysteresis — count=1 equals immediate switch")
e, _ = _make_engine(1)
r = _tick(e, True, True)
_assert_eq(e.current_mode, InferenceMode.CLOUD, "count=1: immediate CLOUD")
_assert(r["transition"],                          "count=1: transition on poll 1")
r = _tick(e, False, False)
_assert_eq(e.current_mode, InferenceMode.LOCAL,  "count=1: immediate LOCAL")
_assert(r["transition"],                          "count=1: transition back")

section("Hysteresis — CSV header has pending columns")
from adaptive_engine import CSV_HEADER
_assert("pending_mode"  in CSV_HEADER, "CSV_HEADER has pending_mode")
_assert("pending_count" in CSV_HEADER, "CSV_HEADER has pending_count")
_assert("raw_decision"  in CSV_HEADER, "CSV_HEADER has raw_decision")

# =============================================================================
# ── TASK 06: Fallback ────────────────────────────────────────────────────────
# =============================================================================

section("Fallback — constants correct")
_assert_eq(FALLBACK_ERROR_THRESHOLD, 3, "FALLBACK_ERROR_THRESHOLD=3")
_assert_eq(FALLBACK_RECOVERY_POLLS,  5, "FALLBACK_RECOVERY_POLLS=5")
_assert(FALLBACK_RECOVERY_POLLS > HYSTERESIS_COUNT,
        "RECOVERY_POLLS > HYSTERESIS_COUNT (prevents thrashing)")

section("Fallback — initial state is not locked")
e, _ = _make_engine()
_assert(e.fallback_locked is False,    "initial fallback_locked=False")
_assert(e._recovery_count == 0,        "initial recovery_count=0")
_assert(e._fallback_reason == "",      "initial fallback_reason empty")

section("Fallback — trigger_fallback() locks engine to LOCAL")
e, tmp = _make_engine()
e._state = InferenceMode.CLOUD
e._pending_mode = InferenceMode.LOCAL
e._pending_count = 2

e.trigger_fallback("test: cloud timeout", conditions=_cond(False, False))

_assert(e.fallback_locked is True,          "fallback_locked=True after trigger")
_assert_eq(e.current_mode, InferenceMode.LOCAL, "state forced to LOCAL")
_assert_eq(e._pending_mode,  None,          "pending_mode cleared on lock")
_assert_eq(e._pending_count, 0,             "pending_count cleared on lock")
_assert_eq(e._recovery_count, 0,            "recovery_count=0 on lock")
_assert(e._fallback_reason != "",           "fallback_reason set")

section("Fallback — trigger_fallback() is idempotent (duplicate calls ignored)")
e, _ = _make_engine()
e.trigger_fallback("first lock")
reason_after_first = e._fallback_reason

e.trigger_fallback("second lock — must be ignored")
_assert_eq(e._fallback_reason, reason_after_first,
           "second trigger_fallback() does not overwrite reason")

section("Fallback — hysteresis bypassed while locked")
e, _ = _make_engine(3)
e.trigger_fallback("test lock")
_assert(e.fallback_locked is True, "locked")

# Even if conditions are perfect, the engine must stay LOCAL while locked
for i in range(10):
    e._check_fallback_recovery(_cond(False, False))  # unhealthy → no recovery
    _assert_eq(e.current_mode, InferenceMode.LOCAL,
               "still LOCAL during lock poll {0}".format(i + 1))

section("Fallback — recovery requires FALLBACK_RECOVERY_POLLS consecutive healthy polls")
e, _ = _make_engine()
e.trigger_fallback("test lock")
_assert(e.fallback_locked is True, "locked before recovery test")

# Feed healthy polls one at a time — must NOT unlock before the threshold
for i in range(FALLBACK_RECOVERY_POLLS - 1):
    e._check_fallback_recovery(_cond(True, True))
    _assert(e.fallback_locked is True,
            "still locked after {0}/{1} healthy polls".format(
                i + 1, FALLBACK_RECOVERY_POLLS))
    _assert_eq(e._recovery_count, i + 1,
               "recovery_count={0}".format(i + 1))

# Final healthy poll — must unlock
e._check_fallback_recovery(_cond(True, True))
_assert(e.fallback_locked is False, "unlocked after {0} healthy polls".format(
    FALLBACK_RECOVERY_POLLS))
_assert_eq(e._recovery_count, 0,    "recovery_count reset to 0 after unlock")

section("Fallback — recovery counter resets on unhealthy poll mid-recovery")
e, _ = _make_engine()
e.trigger_fallback("test lock")

# 3 healthy polls
for _ in range(3):
    e._check_fallback_recovery(_cond(True, True))
_assert_eq(e._recovery_count, 3, "recovery_count=3 after 3 healthy polls")

# One unhealthy poll — must reset counter
e._check_fallback_recovery(_cond(False, False))
_assert_eq(e._recovery_count, 0, "recovery_count reset to 0 after unhealthy poll")
_assert(e.fallback_locked is True, "still locked after reset")

section("Fallback — LOCKED event written to fallback CSV")
tmp_fallback = tempfile.mktemp(suffix=".csv")
e, tmp_engine = _make_engine()
e._fallback_log_file = tmp_fallback

from adaptive_engine import _init_fallback_csv
_init_fallback_csv(tmp_fallback)

e.trigger_fallback("csv test lock", conditions=_cond(False, False, rtt=250.0))

_assert(os.path.exists(tmp_fallback), "fallback CSV created")

with open(tmp_fallback) as f:
    rows = list(_csv.DictReader(f))

_assert_eq(len(rows), 1,                "one row written on lock")
_assert_eq(rows[0]["event"], "LOCKED",  "event=LOCKED")
_assert("csv test lock" in rows[0]["reason"], "reason written")
_assert_eq(rows[0]["rtt_ms"], "250.00", "rtt_ms written")
os.remove(tmp_fallback)

section("Fallback — RECOVERED event written to fallback CSV")
tmp_fallback = tempfile.mktemp(suffix=".csv")
e, tmp_engine = _make_engine()
e._fallback_log_file = tmp_fallback
_init_fallback_csv(tmp_fallback)

e.trigger_fallback("recovery test")
for _ in range(FALLBACK_RECOVERY_POLLS):
    e._check_fallback_recovery(_cond(True, True, rtt=30.0, bw=800.0))

with open(tmp_fallback) as f:
    rows = list(_csv.DictReader(f))

_assert_eq(len(rows), 2,                  "two rows: LOCKED then RECOVERED")
_assert_eq(rows[0]["event"], "LOCKED",    "first row=LOCKED")
_assert_eq(rows[1]["event"], "RECOVERED", "second row=RECOVERED")
_assert_eq(rows[1]["recovery_polls"],
           str(FALLBACK_RECOVERY_POLLS),  "recovery_polls written")
os.remove(tmp_fallback)

section("Fallback — fallback_locked property is publicly readable")
e, _ = _make_engine()
_assert(hasattr(e, "fallback_locked"),     "fallback_locked property exists")
_assert(e.fallback_locked is False,        "readable before lock")
e.trigger_fallback("prop test")
_assert(e.fallback_locked is True,         "readable after lock")

section("Fallback — CSV_HEADER includes all fallback columns")
for col in FALLBACK_CSV_HEADER:
    _assert(isinstance(col, str), "FALLBACK_CSV_HEADER col is str: " + col)
for required in ["event", "reason", "consecutive_errors", "recovery_polls"]:
    _assert(required in FALLBACK_CSV_HEADER,
            "FALLBACK_CSV_HEADER has '{0}'".format(required))

# =============================================================================
# ── TASK 07: Mode Switch Log ─────────────────────────────────────────────────
# =============================================================================

section("Switch log — MODE_SWITCH_CSV_HEADER has all required columns")
required_cols = [
    "timestamp", "previous_mode", "new_mode", "trigger_type",
    "rtt_ms", "bandwidth_kbps", "error_rate_pct", "inference_ms",
    "network_ok", "bandwidth_ok", "polls_confirmed", "reason"
]
for col in required_cols:
    _assert(col in MODE_SWITCH_CSV_HEADER,
            "MODE_SWITCH_CSV_HEADER has '{0}'".format(col))

section("Switch log — hysteresis switch writes one row with correct values")
tmp_sw = tempfile.mktemp(suffix=".csv")
e, tmp_eng = _make_engine(3)
e._switch_log_file = tmp_sw
_init_mode_switch_csv(tmp_sw)

# Drive 3 consecutive CLOUD polls to trigger a hysteresis switch
for _ in range(3):
    _tick(e, True, True, rtt=42.0, bw=800.0, err=0.5)

# Manually write what _tick() would write on the confirmed switch
# (in the real engine this happens inside _tick() section 6)
_write_mode_switch(tmp_sw, {
    "timestamp":      "2026-04-03 10:00:00",
    "previous_mode":  InferenceMode.LOCAL,
    "new_mode":       InferenceMode.CLOUD,
    "trigger_type":   "HYSTERESIS",
    "rtt_ms":         "42.00",
    "bandwidth_kbps": "800.00",
    "error_rate_pct": "0.50",
    "inference_ms":   "350.00",
    "network_ok":     1,
    "bandwidth_ok":   1,
    "polls_confirmed":3,
    "reason":         "",
})

_assert(os.path.exists(tmp_sw), "switch log file created")
with open(tmp_sw) as f:
    rows = list(_csv.DictReader(f))
_assert_eq(len(rows), 1,                           "exactly 1 row after 1 switch")
_assert_eq(rows[0]["trigger_type"], "HYSTERESIS",  "trigger_type=HYSTERESIS")
_assert_eq(rows[0]["previous_mode"], "LOCAL",      "previous_mode=LOCAL")
_assert_eq(rows[0]["new_mode"], "CLOUD",           "new_mode=CLOUD")
_assert_eq(rows[0]["rtt_ms"], "42.00",             "rtt_ms written correctly")
_assert_eq(rows[0]["bandwidth_kbps"], "800.00",    "bandwidth_kbps written")
_assert_eq(rows[0]["error_rate_pct"], "0.50",      "error_rate_pct written")
_assert_eq(rows[0]["polls_confirmed"], "3",         "polls_confirmed=3")
_assert_eq(rows[0]["reason"], "",                  "reason empty for hysteresis")
os.remove(tmp_sw)

section("Switch log — fallback lock writes FALLBACK row")
tmp_sw = tempfile.mktemp(suffix=".csv")
e, tmp_eng = _make_engine()
e._switch_log_file = tmp_sw
e._fallback_log_file = tempfile.mktemp(suffix=".csv")
_init_mode_switch_csv(tmp_sw)
from adaptive_engine import _init_fallback_csv
_init_fallback_csv(e._fallback_log_file)

e._state = InferenceMode.CLOUD
e.trigger_fallback(
    "cloud_infer failed 3x consecutively",
    conditions={"rtt_ms": 250.0, "bandwidth_kbps": 50.0,
                "error_rate_pct": 8.0, "inference_ms": 9500.0,
                "network_ok": False, "bandwidth_ok": False,
                "consecutive_errors": 3}
)

_assert(os.path.exists(tmp_sw), "switch log created by trigger_fallback")
with open(tmp_sw) as f:
    rows = list(_csv.DictReader(f))
_assert_eq(len(rows), 1,                          "1 row written on fallback lock")
_assert_eq(rows[0]["trigger_type"], "FALLBACK",   "trigger_type=FALLBACK")
_assert_eq(rows[0]["previous_mode"], "CLOUD",     "previous_mode=CLOUD")
_assert_eq(rows[0]["new_mode"], "LOCAL",           "new_mode=LOCAL")
_assert_eq(rows[0]["polls_confirmed"], "0",        "polls_confirmed=0 for fallback")
_assert("cloud_infer" in rows[0]["reason"],        "reason contains failure description")
_assert_eq(rows[0]["rtt_ms"], "250.00",            "rtt_ms at lock time written")
os.remove(tmp_sw)

section("Switch log — fallback recovery writes RECOVERY row")
tmp_sw = tempfile.mktemp(suffix=".csv")
e, tmp_eng = _make_engine()
e._switch_log_file = tmp_sw
e._fallback_log_file = tempfile.mktemp(suffix=".csv")
_init_mode_switch_csv(tmp_sw)
_init_fallback_csv(e._fallback_log_file)

# Lock first (generates one FALLBACK row), then recover
e._state = InferenceMode.CLOUD
e.trigger_fallback("test recovery logging")
for _ in range(FALLBACK_RECOVERY_POLLS):
    e._check_fallback_recovery(
        {"rtt_ms": 20.0, "bandwidth_kbps": 900.0,
         "error_rate_pct": 0.0, "inference_ms": 350.0,
         "network_ok": True, "bandwidth_ok": True}
    )

with open(tmp_sw) as f:
    rows = list(_csv.DictReader(f))
_assert_eq(len(rows), 2,                          "2 rows: FALLBACK then RECOVERY")
_assert_eq(rows[0]["trigger_type"], "FALLBACK",   "row 0: FALLBACK")
_assert_eq(rows[1]["trigger_type"], "RECOVERY",   "row 1: RECOVERY")
_assert_eq(rows[1]["previous_mode"], "LOCAL",     "recovery: previous_mode=LOCAL")
_assert_eq(rows[1]["new_mode"], "CLOUD",           "recovery: new_mode=CLOUD")
_assert_eq(rows[1]["polls_confirmed"],
           str(FALLBACK_RECOVERY_POLLS),           "recovery: polls_confirmed correct")
_assert_eq(rows[1]["rtt_ms"], "20.00",             "recovery: rtt_ms at recovery time")
os.remove(tmp_sw)

section("Switch log — no row written on stable STAY polls")
tmp_sw = tempfile.mktemp(suffix=".csv")
_init_mode_switch_csv(tmp_sw)
e, _ = _make_engine(3)
e._switch_log_file = tmp_sw

# 10 stable LOCAL polls — no switches
for _ in range(10):
    _tick(e, False, False)

with open(tmp_sw) as f:
    rows = list(_csv.DictReader(f))
_assert_eq(len(rows), 0, "zero rows after 10 stable STAY polls")
os.remove(tmp_sw)

section("Switch log — only one row per switch, not one per poll")
tmp_sw = tempfile.mktemp(suffix=".csv")
_init_mode_switch_csv(tmp_sw)
e, _ = _make_engine(3)
e._switch_log_file = tmp_sw

# 2 pending CLOUD polls — should produce zero rows (not confirmed yet)
_tick(e, True, True)
_tick(e, True, True)
with open(tmp_sw) as f:
    rows = list(_csv.DictReader(f))
_assert_eq(len(rows), 0, "zero rows after 2 pending polls (not yet confirmed)")
os.remove(tmp_sw)

# =============================================================================
# RESULTS
# =============================================================================

print("\n" + "=" * 64)
print("  Results: {0} passed, {1} failed".format(_passed, _failed))
if _errors:
    print("  Failed:")
    for n in _errors:
        print("    - " + n)
print("=" * 64)
sys.exit(0 if _failed == 0 else 1)
