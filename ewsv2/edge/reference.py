"""The edge algorithm, in Python: the exact definition the Arduino firmware implements (Segment 8).

It is sized for an Arduino Uno (ATmega328P: 2 KB RAM, 32 KB flash, 8-bit CPU, only 32-bit
floats) and written so that the Uno, a PC build of the same C++ and this file compute
**bit-identical** results: integers wherever possible, 32-bit floats in a fixed order elsewhere.

Readings: the laptop sends each sensor as an integer **count** of 1/16 of the sensor's normal
unit (S4), relative to a fixed offset: ``c = round(x / q) - O`` with ``q = unit / 16``. So one
normal unit is exactly 16 counts on every sensor, and every sum below is an exact integer.
Missing readings are carried forward up to 5 minutes, as everywhere in ewsv2.

Per sensor, every minute (history kept on the device: the last 16 minutes, and sums of the last
four complete quarter-hours):

    z    = (c − mean of the previous 15 minutes) / 16        (needs ≥ 5 values; clipped ±50)
    d60  = (c − mean of the last 4 complete quarter-hours) / 16   (needs ≥ 20 values; clipped ±50)
    r15  = (max − min of the previous 15 minutes) / 16       (needs ≥ 5 values; clipped 0..50)
    m    = |z| / move_limit                                  (≥ 1: moved more than in normal water)

Station: breadth = number of sensors with m ≥ 1, maxm = largest m (top sensor = its index).

Detector: a small network (3k + 2 → 16 → 8 → 1, ReLU) on the standardised features, float32,
sums in input order. Alarm when its output (log-odds) ≥ θ. θ starts at the 1-false-alarm-per-day
threshold and adapts online (S5): every minute with data lowers it by step × budget / 1440; when an
alarm episode ends (no alarm for more than 5 minutes) the operator's verdict comes back and every
false alarm in it raises θ by ``step``. θ is a 32-bit float on the device and here.

Fault checks (edge versions of S4, on counts, exact):
    stuck   the count has been exactly constant for ≥ stuck_limit minutes
    dropout one step down by ≥ max(10 units, half its value) to ≤ low_limit; on while it stays there
    spike   the previous minute was > spike_limit units from the median of the 15 minutes before
            it, and this minute is back within 3 units
Lone jump (S4 rule): during an alarm the sensor with the largest |z| stepped ≥ 6 units in one
minute, that step is ≥ 70% of (its |z| + 1), and no other sensor has |z| ≥ 3: a sensor alert, not
a water alarm; the sensor stays a suspect for 60 minutes unless it keeps stepping the same way
by ≥ 2 units.
Triage: a flagged sensor stays a suspect for 60 minutes; the water alarm is withdrawn when every
non-suspect sensor has m < 1. Sensor alert = a flag (or a lone jump) this minute.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

K = 16                      # counts per normal unit
CLIP = np.float32(50.0)
HOLD = 60
CARRY = 5
MISSING = -32768            # the device's code for "no reading"
COUNT_MAX = 32000
JUMP_UNITS, JUMP_SHARE, Z_MOVE, CONTINUE_UNITS = 6, 0.7, 3.0, 2
GAP, MAX_EP = 5, 60
STEP, LOW, HIGH = 0.5, -10.0, 15.0

f32 = np.float32


# ---------------------------------------------------------------------------- counts
def to_counts(frame: pd.DataFrame, spec: dict) -> np.ndarray:
    """Raw readings → integer counts (float array, NaN = missing). What the laptop sends."""
    out = np.full((len(frame), len(spec["sensors"])), np.nan)
    for j, s in enumerate(spec["sensors"]):
        x = frame[s].to_numpy(dtype=float)
        c = np.round(x / spec["q"][s]) - spec["offset"][s]
        out[:, j] = np.clip(c, -COUNT_MAX, COUNT_MAX)
    return out


def carry(C: np.ndarray) -> np.ndarray:
    return pd.DataFrame(C).ffill(limit=CARRY).to_numpy()


# ---------------------------------------------------------------------------- features
def _rolling(C: np.ndarray, window: int, lag: int):
    """Sum, count, max, min, median over rows t-lag-window+1 .. t-lag (NaN ignored)."""
    df = pd.DataFrame(C).shift(lag)
    r = df.rolling(window, min_periods=1)
    return (r.sum().to_numpy(), r.count().to_numpy(), r.max().to_numpy(), r.min().to_numpy())


def _blocks60(C: np.ndarray):
    """Sum and count of the last 4 complete quarter-hours (blocks aligned with the stream start)."""
    n, k = C.shape
    nb = (n + 14) // 15
    pad = np.full((nb * 15, k), np.nan)
    pad[:n] = C
    blk = pad.reshape(nb, 15, k)
    bs = np.nansum(blk, axis=1)
    bc = np.isfinite(blk).sum(axis=1).astype(float)
    cs = np.vstack([np.zeros((1, k)), np.cumsum(bs, axis=0)])
    cc = np.vstack([np.zeros((1, k)), np.cumsum(bc, axis=0)])
    b = np.arange(n) // 15                       # current (incomplete) block
    lo = np.maximum(b - 4, 0)
    return cs[b] - cs[lo], cc[b] - cc[lo]


def features(C: np.ndarray, spec: dict):
    """(n, 3k+2) float32 features, move ratios m (n, k), |z| (n, k), all float32 like the device."""
    s15, n15, mx15, mn15 = _rolling(C, 15, 1)
    s60, n60 = _blocks60(C)
    ok = np.isfinite(C)
    with np.errstate(invalid="ignore", divide="ignore"):
        zn = np.where((n15 >= 5) & ok, n15 * C - s15, np.nan)           # exact integers
        z = (zn.astype(f32) / (n15 * K).astype(f32)).astype(f32)
        dn = np.where((n60 >= 20) & ok, n60 * C - s60, np.nan)
        d = (dn.astype(f32) / (n60 * K).astype(f32)).astype(f32)
        r = np.where(n15 >= 5, mx15 - mn15, np.nan).astype(f32) / f32(K)
    z = np.clip(np.nan_to_num(z, nan=0.0), -CLIP, CLIP).astype(f32)
    d = np.clip(np.nan_to_num(d, nan=0.0), -CLIP, CLIP).astype(f32)
    r = np.clip(np.nan_to_num(r, nan=0.0), f32(0), CLIP).astype(f32)
    ml = np.array([spec["move_limit"][s] for s in spec["sensors"]], dtype=f32)
    az = np.abs(z)
    m = (az / ml).astype(f32)
    breadth = (m >= f32(1)).sum(axis=1).astype(f32)
    F = np.column_stack([z, d, r, breadth, m.max(axis=1)]).astype(f32)
    return F, m, az


# ---------------------------------------------------------------------------- network
def mlp(F: np.ndarray, model: dict, int8: bool = True) -> np.ndarray:
    """Log-odds, float32, sums in input order exactly as the C loop does them."""
    h = ((F.astype(f32) - np.array(model["mu"], dtype=f32)) / np.array(model["sd"], dtype=f32)).astype(f32)
    layers = model["layers"]
    for li, L in enumerate(layers):
        if int8:
            W = (np.array(L["Wq"], dtype=f32) * f32(L["scale"])).astype(f32)
        else:
            W = np.array(L["W"], dtype=f32)
        acc = np.zeros((h.shape[0], W.shape[1]), dtype=f32)
        for i in range(W.shape[0]):
            acc = (acc + (h[:, i:i + 1] * W[i][None, :]).astype(f32)).astype(f32)
        acc = (acc + np.array(L["b"], dtype=f32)).astype(f32)
        h = np.maximum(acc, f32(0)) if li < len(layers) - 1 else acc
    return h[:, 0].astype(f32)


# ---------------------------------------------------------------------------- fault checks
def run_lengths(C: np.ndarray) -> np.ndarray:
    n, k = C.shape
    out = np.zeros((n, k), dtype=np.int64)
    for j in range(k):
        v = C[:, j]
        ok = np.isfinite(v)
        same = np.zeros(n, dtype=bool)
        same[1:] = ok[1:] & ok[:-1] & (v[1:] == v[:-1])
        start = np.where(~same, np.arange(n), 0)
        np.maximum.accumulate(start, out=start)
        rl = np.arange(n) - start + 1
        rl[~ok] = 0
        out[:, j] = rl
    return out


def _median2(C: np.ndarray) -> np.ndarray:
    """2 × median of rows t-16 .. t-2 (≥ 5 values), an exact integer (NaN if too few)."""
    df = pd.DataFrame(C).shift(2)
    med = df.rolling(15, min_periods=5).median().to_numpy()
    return 2 * med


def fault_flags(C: np.ndarray, spec: dict) -> np.ndarray:
    sensors = spec["sensors"]
    n, k = C.shape
    stuck = run_lengths(C) >= np.array([spec["stuck_limit"][s] for s in sensors])
    low = np.array([spec["low_count"][s] for s in sensors], dtype=float)
    base = np.array([spec["offset"][s] for s in sensors], dtype=float)   # value/q = base + count
    drop = np.zeros((n, k), dtype=bool)
    for j in range(k):
        v = C[:, j]
        below = np.where(np.isfinite(v), v, np.inf) <= low[j]
        on = False
        for t in range(n):
            if not below[t]:
                on = False
            elif not on and t > 0 and np.isfinite(v[t - 1]) and v[t - 1] > low[j]:
                need = max(2 * 10 * K, abs(base[j] + v[t - 1]))
                if 2 * (v[t - 1] - v[t]) >= need:
                    on = True
            drop[t, j] = on
    med2 = _median2(C)
    prev = np.vstack([np.full((1, k), np.nan), C[:-1]])
    lim2 = np.array([spec["spike2"][s] for s in sensors], dtype=f32).astype(float)
    with np.errstate(invalid="ignore"):
        spike = (np.abs(2 * prev - med2) > lim2) & (np.abs(2 * C - med2) < 2 * 3 * K)
    return stuck | drop | np.nan_to_num(spike, nan=0).astype(bool)


def lone_jumps(C: np.ndarray, az: np.ndarray, alarm: np.ndarray):
    n, k = C.shape
    step = np.zeros_like(C)
    step[1:] = np.nan_to_num(C[1:] - C[:-1], nan=0.0)
    lead = az.argmax(axis=1)
    if k >= 2:
        top2 = -np.sort(-az, axis=1)[:, :2]
        z1, z2 = top2[:, 0], top2[:, 1]
    else:
        z1, z2 = az[:, 0], np.zeros(n, dtype=f32)
    jl = (np.abs(step[np.arange(n), lead]).astype(f32) / f32(K)).astype(f32)
    share = (jl / (z1 + f32(1)).astype(f32)).astype(f32)
    lone = np.asarray(alarm, dtype=bool) & (jl >= f32(JUMP_UNITS)) & (z2 < f32(Z_MOVE)) & (share >= f32(JUMP_SHARE))
    jumped = np.zeros((n, k), dtype=bool)
    for t in np.flatnonzero(lone):
        j = lead[t]
        sign = np.sign(step[t, j])
        end = min(n, t + HOLD + 1)
        cont = np.flatnonzero(step[t + 1:end, j] * sign >= CONTINUE_UNITS * K)
        stop = t + 1 + cont[0] if len(cont) else end
        jumped[t:stop, j] = True
    return lone, lead, jumped


# ---------------------------------------------------------------------------- threshold
def false_count(alarm: np.ndarray, y: np.ndarray, gap: int = GAP, max_ep: int = MAX_EP) -> int:
    """False alarms in an alarm stretch, counted as the S1 evaluator counts them."""
    from ewsv2.calibrate.methods import _false_episodes
    return _false_episodes(alarm, y, gap, max_ep)


def adaptive_alarm(lp: np.ndarray, has: np.ndarray, theta0: float, y: np.ndarray | None, budget: float = 1.0,
                   adaptive: bool = True):
    """Online threshold in float32, with the operator's verdict at the end of each episode."""
    n = len(lp)
    y = np.zeros(n, dtype=bool) if y is None else np.asarray(y, dtype=bool)
    decay = f32(f32(STEP) * f32(budget) / f32(1440))
    lo, hi, step = f32(LOW), f32(HIGH), f32(STEP)
    theta = f32(theta0)
    alarm = np.zeros(n, dtype=bool)
    trace = np.full(n, np.nan, dtype=f32)
    ep_start, last_on = -1, -10**9
    asks = []
    for t in range(n):
        if not has[t]:
            continue
        trace[t] = theta
        on = bool(lp[t] >= theta)
        alarm[t] = on
        if on:
            if t - last_on > GAP:
                ep_start = t
            last_on = t
        if not adaptive:
            if not on and ep_start >= 0 and t - last_on > GAP:
                ep_start = -1
            continue
        theta = min(max(f32(theta - decay), lo), hi)
        if not on and ep_start >= 0 and t - last_on > GAP:
            nf = false_count(alarm[ep_start:last_on + 1], y[ep_start:last_on + 1])
            asks.append((t, ep_start, last_on, nf))
            if nf > 0:
                theta = min(max(f32(theta + f32(step * f32(nf))), lo), hi)
            ep_start = -1
    return alarm, trace, asks


# ---------------------------------------------------------------------------- all together
def run_counts(C_raw: np.ndarray, spec: dict, int8: bool = True, y_water: np.ndarray | None = None,
               adaptive: bool = True, rows: np.ndarray | None = None) -> pd.DataFrame:
    """Per-minute edge outputs for a stream of counts (NaN = missing): the device's answers."""
    C = carry(C_raw)
    has = np.isfinite(C).any(axis=1)
    if rows is not None:
        has = has & np.asarray(rows, dtype=bool)
    F, m, az = features(C, spec)
    lp = mlp(F, spec["model"], int8)
    alarm, trace, _ = adaptive_alarm(lp, has, spec["theta0"], y_water, spec.get("budget", 1.0), adaptive)
    flags = fault_flags(C, spec) & has[:, None]
    suspect = pd.DataFrame(flags.astype(float)).rolling(HOLD + 1, min_periods=1).max().to_numpy().astype(bool)
    lone, lead, jumped = lone_jumps(C, az, alarm)
    suspect |= jumped
    clean = np.where(suspect, f32(0), m).max(axis=1)
    explained = suspect.any(axis=1) & (clean < f32(1))
    water = alarm & ~explained & ~lone
    flags[np.flatnonzero(lone), lead[lone]] = True
    mask = (flags * (1 << np.arange(C.shape[1]))).sum(axis=1)
    return pd.DataFrame({"logit": lp, "theta": trace, "alarm": alarm, "water": water, "sensor": flags.any(axis=1),
                         "mask": mask, "top": m.argmax(axis=1), "has_data": has})


def run_edge(frame: pd.DataFrame, spec: dict, int8: bool = True, y_water: np.ndarray | None = None,
             adaptive: bool = True, rows: np.ndarray | None = None) -> pd.DataFrame:
    out = run_counts(to_counts(frame, spec), spec, int8, y_water, adaptive, rows)
    out.index = frame.index
    return out
