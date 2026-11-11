"""Segment 8 experiment: the Uno-sized edge algorithm on GECCO-Triage, and the hardware-in-the-loop checks.

    python -m ewsv2 edge --config s8_edge                       (Uno found automatically if plugged in)

1. Reference (Python, always): the edge algorithm on the whole GECCO-Triage test stream – int8 and
   float32 weights, fixed and adaptive threshold, with and without the edge triage – scored exactly
   like the S5 pipeline.
2. Targets, each used only if available (one line says why one is skipped):
   ``desktop`` the firmware's C++ built for this computer, whole test period;
   ``sim``     the real Uno firmware on a simulated ATmega328P (Linux + simavr + avr-gcc);
   ``uno``     the Arduino Uno over USB, ``edge.minutes`` minutes (default one week).
   Every answer is compared with the reference. If the board stops answering mid-run the replay
   stops cleanly, the minutes done are compared, and the laptop's answers stand for the rest.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ewsv2.core.paths import output_dir  # noqa: E402
from ewsv2.core.registry import build  # noqa: E402
from ewsv2.core.types import CLASS_COL, EVENT_COL, Dataset  # noqa: E402
from ewsv2.edge import export as E  # noqa: E402
from ewsv2.edge import hil as H  # noqa: E402
from ewsv2.edge import reference as R  # noqa: E402
from ewsv2.inject.benchmark import test_mask  # noqa: E402
from ewsv2.triage import evaluate as TE  # noqa: E402
from ewsv2.triage.run import build_benchmark  # noqa: E402

log = logging.getLogger(__name__)
BUILD_INFO = E.FIRMWARE_DIR / "uno_build_info.json"     # sizes measured with avr-gcc (shipped)


def _names(mask: np.ndarray, sensors: list[str]) -> pd.Series:
    return pd.Series([",".join(s for j, s in enumerate(sensors) if int(v) >> j & 1) for v in mask])


class _Scorer:
    """Scores per-minute water alarms and sensor alerts the way S5 scores its pipeline."""

    def __init__(self, ds: Dataset, inj: Dataset, episodes: pd.DataFrame, cfg: dict):
        fcfg = cfg.get("fuse") or {}
        self.tol, self.gap = int(fcfg.get("tolerance_after", 0)), int(fcfg.get("merge_gap", 5))
        self.max_ep, self.grace = int(fcfg.get("max_episode_minutes", 60)), int(fcfg.get("fault_grace", 10))
        sp = fcfg.get("splitter", {"name": "official", "params": {"cv_folds": 4, "gap_minutes": 1440}})
        split = build("splitter", sp["name"], **(sp.get("params") or {})).split(inj)
        f = inj.frame
        self.index = f.index
        self.cls = f[CLASS_COL].to_numpy().astype(int)
        real = ds.frame[EVENT_COL].to_numpy().astype(int)
        self.has = ~f[inj.sensors].isna().all(axis=1).to_numpy()
        self.valid = split.test & self.has
        self.eps = TE.test_episodes(real, self.index, test_mask(inj), self.has, episodes)
        self.sensors = list(inj.sensors)

    def score(self, water, sensor, mask, rows=None) -> dict:
        valid = self.valid if rows is None else self.valid & rows
        w = np.asarray(water, dtype=bool) & valid
        s = np.asarray(sensor, dtype=bool) & valid
        names = _names(np.where(s, mask, 0), self.sensors)
        eps = self.eps
        if rows is not None:              # only the episodes inside the replayed stretch
            eps = eps[rows[eps["start_pos"].to_numpy().astype(int)] & rows[eps["end_pos"].to_numpy().astype(int)]]
        if len(eps):
            oc = TE.episode_outcomes(eps, w, s, names, self.grace, self.tol)
        else:                             # a short replay may contain no episode
            oc = eps.assign(outcome=pd.Series(dtype=object), delay_min=pd.Series(dtype=float),
                            names_right_sensor=pd.Series(dtype=object))
        return {**TE.summarise(oc), **TE.rates(self.cls, w, s, self.index, valid, self.grace, self.tol, self.gap,
                                               self.max_ep)}


KEEP = ["real_caught", "real_n", "contaminant_caught", "contaminant_n", "fault_correct", "fault_n",
        "fault_raised_water_alarm", "fault_named_right_sensor", "water_event_f1", "water_false_alarms_per_day",
        "false_sensor_alerts_per_day", "water_median_delay"]


def _row(label: str, weights: str, threshold: str, triage: str, sc: dict) -> dict:
    return {"system": label, "weights": weights, "threshold": threshold, "triage": triage,
            **{k: sc.get(k, np.nan) for k in KEEP}}


def get_spec(ds: Dataset, cfg: dict) -> dict:
    """The shipped model, or a new one when asked for or when none exists for this station."""
    ecfg = cfg.get("edge") or {}
    spec = None if ecfg.get("retrain", False) else E.load_spec(ds.name)
    if spec is None:
        log.warning("No edge model for %s yet: training one now (about a minute, needs TensorFlow).", ds.name)
        spec = E.export(ds, int(cfg.get("seed", 42)), int(ecfg.get("epochs", 15)),
                        float((cfg.get("fuse") or {}).get("budget", 1.0)),
                        sketch=ds.name == ecfg.get("sketch_dataset", "gecco2018"))
    return spec


def _target_link(target: str, ecfg: dict, out_dir: Path, header: Path, spec: dict):
    """Open a target, or raise H.DeviceError saying why it is not available here."""
    build_dir = out_dir / "build"
    if target == "desktop":
        return H.ProcessLink([str(H.build_desktop(build_dir, header))], "desktop"), {}
    if target == "sim":
        exe = H.build_sim(build_dir)
        elf, sizes = H.build_uno_elf(build_dir / "uno", header)
        return H.ProcessLink([str(exe), str(elf)], "sim"), sizes
    if target == "uno":
        link, msg = H.open_board(ecfg.get("port", "auto"), int(ecfg.get("baud", 115200)))
        if link is None:
            raise H.DeviceError(msg)
        return link, {}
    raise H.DeviceError(f"unknown target {target}")


def run_edge_experiment(ds: Dataset, cfg: dict, out_dir: Path) -> dict:
    ecfg = cfg.get("edge") or {}
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = get_spec(ds, cfg)
    header = out_dir / "build" / f"edge_model_{ds.name}.h"
    E.write_header(spec, header)

    inj, episodes = build_benchmark(ds, cfg)
    sc = _Scorer(ds, inj, episodes, cfg)
    frame = inj.frame
    water_label = sc.cls == 1
    C_all = R.to_counts(frame, spec)

    # 1. reference on the whole test stream
    rows, refs, timing = [], {}, {}
    for int8 in (True, False):
        for adaptive in (False, True):
            t0 = time.perf_counter()
            ref = R.run_counts(C_all, spec, int8, water_label, adaptive, rows=sc.valid)
            timing[(int8, adaptive)] = time.perf_counter() - t0
            refs[(int8, adaptive)] = ref
            wname, tname = ("int8" if int8 else "float32"), ("adaptive" if adaptive else "fixed")
            none = np.zeros(len(ref), dtype=bool)
            for tri in (False, True):
                water = ref["water"].to_numpy() if tri else ref["alarm"].to_numpy()
                sensor = ref["sensor"].to_numpy() if tri else none
                s = sc.score(water, sensor, ref["mask"].to_numpy())
                rows.append(_row(f"edge {wname} + {tname}" + (" + edge triage" if tri else ""), wname, tname,
                                 "edge" if tri else "none", s))
    s5 = _s5_pipeline(ds.name)
    if s5 is not None:
        rows.append(s5)
    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "edge_results.csv", index=False)
    a, b = refs[(True, True)], refs[(False, True)]
    v = sc.valid
    quant = {"alarm_agreement_int8_vs_float32": float((a["alarm"].to_numpy()[v] == b["alarm"].to_numpy()[v]).mean()),
             "max_logit_diff": float(np.abs(a["logit"].to_numpy()[v].astype(float) - b["logit"].to_numpy()[v]).max())}

    # 2. the targets
    int8, adaptive = bool(ecfg.get("int8", True)), bool(ecfg.get("adaptive", True))
    start = int(np.flatnonzero(sc.valid)[0])
    targets = list(ecfg.get("targets", ["desktop", "sim", "uno"]))
    hil, trace, sizes_all = {}, None, {}
    for target in targets:
        key = {"desktop": "desktop_minutes", "sim": "sim_minutes", "uno": "minutes"}.get(target, "minutes")
        minutes = int(ecfg.get(key, 0) or 0)
        end = len(frame) if minutes <= 0 else min(len(frame), start + minutes)
        try:
            link, sizes = _target_link(target, ecfg, out_dir, header, spec)
        except Exception as exc:
            hil[target] = {"status": f"skipped: {str(exc).splitlines()[0]}"}
            log.info("HIL %s skipped: %s", target, str(exc).splitlines()[0])
            continue
        sizes_all.update(sizes)
        C, y = C_all[start:end], water_label[start:end]
        log.info("HIL %s: replaying %d minutes ...", target, len(C))
        try:
            dev, info = H.replay(link, C, y, int8, adaptive, progress_every=1440 if target != "desktop" else 0)
        except Exception as exc:
            hil[target] = {"status": f"failed: {str(exc).splitlines()[0]}"}
            log.warning("HIL %s failed: %s", target, exc)
            continue
        finally:
            link.close()
        if info.get("dataset") != ds.name:
            hil[target] = {"status": f"failed: the board runs the {info.get('dataset')} model, not {ds.name}"}
            continue
        ref = R.run_counts(C, spec, int8, y, adaptive)
        agree = H.compare(dev, ref)
        done = len(dev)
        rows_mask = np.zeros(len(frame), dtype=bool)
        rows_mask[start:start + done] = True
        full = {c: np.zeros(len(frame), dtype=int) for c in ("water", "sensor", "mask")}
        for c in full:
            full[c][start:start + done] = dev[c].to_numpy()
        own = sc.score(full["water"].astype(bool), full["sensor"].astype(bool), full["mask"], rows_mask)
        hil[target] = {"status": "ok" if info.get("error") is None else f"partial: {info['error']}",
                       "minutes": done, "from": str(frame.index[start]), "to": str(frame.index[start + done - 1]) if done else "",
                       "device": info, "agreement": agree, "scored": {k: own.get(k) for k in KEEP}}
        dev.index = frame.index[start:start + done]
        dev.to_csv(out_dir / f"hil_{target}_minutes.csv.gz")
        trace = (target, dev, sc.cls[start:start + done])
    if not sizes_all and BUILD_INFO.exists():
        sizes_all = json.loads(BUILD_INFO.read_text(encoding="utf-8")).get(ds.name, {})
        if sizes_all:
            sizes_all["measured"] = "in the cloud with avr-gcc (shipped)"
    info = {"dataset": ds.name, "theta0": spec["theta0"], "stuck_limit": spec["stuck_limit"],
            "model": spec.get("info", {}), "uno_sizes": sizes_all, "int8_vs_float32": quant,
            "test_minutes": int(v.sum()), "reference_us_per_minute": 1e6 * timing[(True, True)] / len(frame),
            "hil": hil, "int8": int8, "adaptive": adaptive}
    (out_dir / "edge_info.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    if trace is not None and len(trace[1]):
        plot_trace(*trace, out_dir / "edge_trace.png", ds.name)
    write_report(ds, results, info, out_dir)
    return {"results": results, "info": info}


def _s5_pipeline(ds_name: str) -> dict | None:
    p = output_dir("s5_fusion", ds_name) / "pipeline_summary.csv"
    if not p.exists():
        return None
    s = pd.read_csv(p)
    r = s[s["variant"] == "rules triage"]
    if r.empty:
        return None
    r = r.iloc[0].to_dict()
    return _row(f"S5 laptop pipeline ({r['label'].split(' + ')[0]})", "laptop", "adaptive", "S4 rules", r)


def plot_trace(target: str, dev: pd.DataFrame, cls: np.ndarray, path: Path, title: str, days: int = 7) -> None:
    n = min(len(dev), days * 1440)
    idx = dev.index[:n]
    fig, ax = plt.subplots(figsize=(11, 4.0))
    ax.plot(idx, dev["logit"].to_numpy()[:n], lw=0.6, color="#2a78d6", label="detector log-odds")
    ax.plot(idx, dev["theta"].to_numpy()[:n], lw=1.2, color="#222222", label="adaptive threshold θ")
    lo, hi = -12, 17
    for code, color, lab in ((1, "#eb6834", "water event (real or injected)"), (2, "#1baf7a", "sensor fault")):
        ax.fill_between(idx, lo, hi, where=cls[:n] == code, color=color, alpha=0.18, step="mid", label=lab)
    w = dev["water"].to_numpy()[:n].astype(bool)
    s = dev["sensor"].to_numpy()[:n].astype(bool)
    ax.scatter(idx[w], np.full(w.sum(), hi - 1.5), s=6, color="#c0392b", label="water alarm (device)")
    ax.scatter(idx[s], np.full(s.sum(), lo + 1.5), s=6, color="#117a65", label="sensor alert (device)")
    ax.set_ylim(lo, hi)
    ax.set_ylabel("log-odds")
    ax.set_title(f"{title}: edge algorithm on the {target} target, first {max(1, n // 1440)} day(s) of the test period")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=7, ncol=6, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _f(v, nd=2):
    try:
        return f"{float(v):.{nd}f}" if np.isfinite(float(v)) else ""
    except (TypeError, ValueError):
        return str(v)


def write_report(ds: Dataset, results: pd.DataFrame, info: dict, out_dir: Path) -> None:
    m, u = info["model"], info.get("uno_sizes") or {}
    L = [f"# Segment 8 – Arduino Uno edge on {ds.name}", "",
         f"Edge model: {m.get('params')} weights; int8 {m.get('int8_bytes')} bytes, float32 {m.get('float32_bytes')} "
         f"bytes, TensorFlow Lite int8 file {m.get('tflite_int8_bytes', '–')} bytes. Start threshold {info['theta0']:.4f} "
         f"(log-odds; 1 false alarm/day on pre-test validation)."]
    if u:
        L.append(f"Uno firmware (avr-gcc): {u.get('flash_bytes')} bytes of 32,256 flash, {u.get('ram_static_bytes')} "
                 f"bytes of 2,048 RAM before the stack.")
    L += ["", "## Edge algorithm on the GECCO-Triage test stream (reference)", "",
          "| system | real | contaminants | faults → sensor | faults → water | water FA/day | sensor FA/day | F1 |",
          "|---|---|---|---|---|---|---|---|"]
    for _, r in results.iterrows():
        L.append(f"| {r['system']} | {int(r['real_caught'])}/{int(r['real_n'])} | {int(r['contaminant_caught'])}/"
                 f"{int(r['contaminant_n'])} | {int(r['fault_correct'])}/{int(r['fault_n'])} | "
                 f"{int(r['fault_raised_water_alarm'])} | {_f(r['water_false_alarms_per_day'])} | "
                 f"{_f(r['false_sensor_alerts_per_day'])} | {_f(r['water_event_f1'])} |")
    q = info["int8_vs_float32"]
    L += ["", f"int8 vs float32 weights (adaptive): alarm agreement {q['alarm_agreement_int8_vs_float32']:.5f} of "
          f"{info['test_minutes']} test minutes.",
          f"Reference: {info['reference_us_per_minute']:.1f} µs per minute on this computer.", "",
          "## Targets (every answer compared with the reference)", ""]
    for target, h in info["hil"].items():
        L.append(f"### {target}")
        if "agreement" not in h:
            L += ["", h.get("status", ""), ""]
            continue
        d, a, s = h["device"], h["agreement"], h["scored"]
        L += ["", f"{h['minutes']} minutes ({h['from']} → {h['to']}), {h['status']}.", "",
              "| measure | value |", "|---|---|",
              f"| alarm / water / sensor / fault-mask / top-sensor agreement | {a.get('alarm_agreement', 0):.5f} / "
              f"{a.get('water_agreement', 0):.5f} / {a.get('sensor_agreement', 0):.5f} / {a.get('mask_agreement', 0):.5f} / "
              f"{a.get('top_agreement', 0):.5f} |",
              f"| log-odds and θ bit-identical | {a.get('logit_bit_identical', 0):.5f} / {a.get('theta_bit_identical', 0):.5f} |",
              f"| compute per minute on the target (mean / max) | {d.get('mean_us', 0):.0f} µs / {d.get('max_us', 0)} µs |",
              f"| algorithm state in RAM | {d.get('state_bytes')} bytes |",
              f"| RAM never used (stack low-water mark, after the run) | {d.get('free_ram') if d.get('free_ram', -1) >= 0 else '–'} |",
              f"| round trip per minute | {d.get('round_trip_ms', 0):.1f} ms |",
              f"| its own alarms, scored | real {s['real_caught']}/{s['real_n']}, contaminants {s['contaminant_caught']}/"
              f"{s['contaminant_n']}, faults → sensor {s['fault_correct']}/{s['fault_n']}, water FA/day "
              f"{_f(s['water_false_alarms_per_day'])} |", ""]
    if (out_dir / "edge_trace.png").exists():
        L += ["![trace](edge_trace.png)", ""]
    L += ["Files: `edge_results.csv`, `edge_info.json`, `hil_<target>_minutes.csv.gz`, `build/`."]
    (out_dir / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
