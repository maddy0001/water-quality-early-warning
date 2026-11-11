"""The edge algorithm one minute at a time, in Python (Segment 9).

``EdgeStream`` keeps exactly the state the Arduino keeps (``firmware/ewsv2_edge/edge_core.h``) and
updates it minute by minute, so the dashboard can run the station live, accept the operator's verdict
whenever it is given, and keep going if the Uno is unplugged. It gives the same answers as the batch
reference (``reference.run_counts``) and the Uno, bit for bit (tested).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ewsv2.edge import reference as R

f32 = np.float32
MISSING = None


@dataclass
class Minute:
    t: int
    logit: float
    theta: float
    has: bool
    alarm: bool
    water: bool
    sensor: bool
    mask: int
    top: int
    ask: bool
    ep_start: int
    last_on: int
    z: np.ndarray            # move of each sensor in normal units (signed)
    m: np.ndarray            # move ratio of each sensor (>= 1: beyond normal)


class EdgeStream:
    def __init__(self, spec: dict, int8: bool = True, adaptive: bool = True):
        self.spec, self.int8, self.adaptive = spec, int8, adaptive
        self.k = len(spec["sensors"])
        model = spec["model"]
        self.mu = np.array(model["mu"], dtype=f32)
        self.sd = np.array(model["sd"], dtype=f32)
        self.layers = []
        for L in model["layers"]:
            W = (np.array(L["Wq"], dtype=f32) * f32(L["scale"])).astype(f32) if int8 else np.array(L["W"], dtype=f32)
            self.layers.append((W, np.array(L["b"], dtype=f32)))
        s = spec["sensors"]
        self.ml = np.array([spec["move_limit"][x] for x in s], dtype=f32)
        self.spike2 = np.array([spec["spike2"][x] for x in s], dtype=f32)
        self.stuck = np.array([spec["stuck_limit"][x] for x in s])
        self.low = np.array([spec["low_count"][x] for x in s])
        self.base = np.array([spec["offset"][x] for x in s])
        self.decay = f32(f32(R.STEP) * f32(spec.get("budget", 1.0)) / f32(1440))
        self.reset()

    def reset(self) -> None:
        k = self.k
        self.t = 0
        self.theta = f32(self.spec["theta0"])
        self.ep_start, self.last_on, self.pending = -1, -1_000_000, False
        self.ring = [[None] * k for _ in range(16)]
        self.blocks: list[tuple[list[int], list[int]]] = []
        self.csum, self.ccnt = [0] * k, [0] * k
        self.last, self.since = [None] * k, [255] * k
        self.run, self.drop = [0] * k, [False] * k
        self.lastflag = [-1_000_000] * k
        self.jon = [[False, False] for _ in range(k)]
        self.jstart = [[0, 0] for _ in range(k)]

    # ------------------------------------------------------------------ helpers
    def _past(self, i: int, s: int):
        return self.ring[(self.t - i) & 15][s] if i <= self.t else None

    def _forward(self, F: np.ndarray) -> f32:
        h = ((F - self.mu) / self.sd).astype(f32)
        for li, (W, b) in enumerate(self.layers):
            acc = np.zeros(W.shape[1], dtype=f32)
            for i in range(W.shape[0]):
                acc = (acc + (h[i] * W[i]).astype(f32)).astype(f32)
            acc = (acc + b).astype(f32)
            h = np.maximum(acc, f32(0)) if li < len(self.layers) - 1 else acc
        return f32(h[0])

    def _clampt(self, v) -> f32:
        return f32(min(max(f32(v), f32(R.LOW)), f32(R.HIGH)))

    def verdict(self, n_false: int) -> None:
        """The operator's answer for the episode that just ended: the number of false alarms in it."""
        if self.adaptive and n_false > 0:
            self.theta = self._clampt(f32(self.theta + f32(f32(R.STEP) * f32(n_false))))
        self.pending = False

    # ------------------------------------------------------------------ one minute
    def push(self, raw) -> Minute:
        """``raw``: k counts (int, or None/NaN for a missing reading)."""
        if self.pending:
            self.verdict(0)
        k, t = self.k, self.t
        if t > 0 and t % 15 == 0:
            self.blocks.append((self.csum, self.ccnt))
            self.blocks = self.blocks[-4:]
            self.csum, self.ccnt = [0] * k, [0] * k
        x = []
        for s in range(k):
            v = raw[s]
            v = None if v is None or (isinstance(v, float) and not np.isfinite(v)) else int(v)
            if v is not None:
                x.append(v)
                self.last[s], self.since[s] = v, 0
            elif self.since[s] < R.CARRY:
                self.since[s] += 1
                x.append(self.last[s])
            else:
                self.since[s] = min(255, self.since[s] + 1)
                x.append(None)
        has = any(v is not None for v in x)

        # features
        z = np.zeros(k, dtype=f32)
        d = np.zeros(k, dtype=f32)
        r = np.zeros(k, dtype=f32)
        for s in range(k):
            vals = [self._past(i, s) for i in range(1, 16)]
            vals = [v for v in vals if v is not None]
            n, sm = len(vals), sum(vals)
            n60 = sum(c[s] for _, c in self.blocks)
            s60 = sum(b[s] for b, _ in self.blocks)
            if n >= 5 and x[s] is not None:
                z[s] = f32(f32(n * x[s] - sm) / f32(n * R.K))
            if n60 >= 20 and x[s] is not None:
                d[s] = f32(f32(n60 * x[s] - s60) / f32(n60 * R.K))
            if n >= 5:
                r[s] = f32(f32(max(vals) - min(vals)) / f32(16))
        z = np.clip(z, -R.CLIP, R.CLIP).astype(f32)
        d = np.clip(d, -R.CLIP, R.CLIP).astype(f32)
        r = np.clip(r, f32(0), R.CLIP).astype(f32)
        az = np.abs(z)
        m = (az / self.ml).astype(f32)
        top = int(np.argmax(m))
        F = np.concatenate([z, d, r, [f32((m >= f32(1)).sum()), m.max()]]).astype(f32)
        lp = self._forward(F)

        # threshold
        theta_out, alarm, ask, eps, lon = f32(np.nan), False, False, 0, 0
        if has:
            theta_out = self.theta
            alarm = bool(lp >= self.theta)
            if alarm:
                if t - self.last_on > R.GAP:
                    self.ep_start = t
                self.last_on = t
            if self.adaptive:
                self.theta = self._clampt(f32(self.theta - self.decay))
                if not alarm and self.ep_start >= 0 and t - self.last_on > R.GAP:
                    ask, eps, lon, self.pending, self.ep_start = True, self.ep_start, self.last_on, True, -1
            elif not alarm and self.ep_start >= 0 and t - self.last_on > R.GAP:
                self.ep_start = -1

        # lone jumps
        step = []
        for s in range(k):
            prev = self._past(1, s)
            st = 0 if x[s] is None or prev is None else x[s] - prev
            step.append(st)
            for g, sign in ((0, 1), (1, -1)):
                if self.jon[s][g] and (t > self.jstart[s][g] + R.HOLD or st * sign >= 2 * R.K):
                    self.jon[s][g] = False
        if k >= 2:
            srt = np.sort(az)[::-1]
            z1, z2 = srt[0], srt[1]
        else:
            z1, z2 = az[0], f32(0)
        lead = int(np.argmax(az))
        jl = f32(f32(abs(step[lead])) / f32(16))
        lone = alarm and jl >= f32(6) and z2 < f32(3) and f32(jl / f32(z1 + f32(1))) >= f32(0.7)
        if lone:
            g = 0 if step[lead] > 0 else 1
            self.jon[lead][g], self.jstart[lead][g] = True, t

        # fault checks and triage
        mask, any_suspect, clean = 0, False, f32(0)
        for s in range(k):
            v, prev = x[s], self._past(1, s)
            self.run[s] = self.run[s] + 1 if (v is not None and prev is not None and v == prev) else (0 if v is None else 1)
            flag = self.run[s] >= self.stuck[s]
            below = v is not None and v <= self.low[s]
            if not below:
                self.drop[s] = False
            elif not self.drop[s] and t > 0 and prev is not None and prev > self.low[s]:
                need = max(2 * 10 * R.K, abs(int(self.base[s]) + prev))
                if 2 * (prev - v) >= need:
                    self.drop[s] = True
            flag = flag or self.drop[s]
            w = [self._past(i, s) for i in range(2, 17)]
            w = sorted(u for u in w if u is not None)
            if len(w) >= 5 and prev is not None and v is not None:
                med2 = 2 * w[len(w) // 2] if len(w) % 2 else w[len(w) // 2 - 1] + w[len(w) // 2]
                if f32(abs(2 * prev - med2)) > self.spike2[s] and abs(2 * v - med2) < 2 * 3 * R.K:
                    flag = True
            if not has:
                flag = False
            if flag:
                mask |= 1 << s
                self.lastflag[s] = t
            suspect = (t - self.lastflag[s] <= R.HOLD) or self.jon[s][0] or self.jon[s][1]
            if suspect:
                any_suspect = True
            elif m[s] > clean:
                clean = m[s]
        if lone:
            mask |= 1 << lead
        water = alarm and not lone and not (any_suspect and clean < f32(1))

        for s in range(k):
            self.ring[t & 15][s] = x[s]
            if x[s] is not None:
                self.csum[s] += x[s]
                self.ccnt[s] += 1
        self.t += 1
        return Minute(t, float(lp), float(theta_out), has, alarm, water, mask != 0, mask, top, ask, eps, lon, z, m)
