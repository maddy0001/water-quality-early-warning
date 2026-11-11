"""Paper tables and figures: ``python -m ewsv2 paper`` (Segment 10).

Reads the result files in ``outputs_v2`` (falls back to the measured headline values bundled in
``ewsv2/app/snapshot.json`` when a file is missing) and writes, to ``outputs_v2/paper``:

    tables/T*.csv           one CSV per table
    tables.md               all tables in Markdown (paste into Word)
    tables.tex              all tables in LaTeX (booktabs, IEEE style)
    figures/F*.png / .pdf   figures at 300 dpi, sized for one IEEE column (3.5 in) or two (7.2 in)
    index.md                what each table and figure shows, with its source file
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from ewsv2.core.paths import OUTPUT_DIR, PACKAGE_DIR  # noqa: E402

PAL = ["#4f46e5", "#0891b2", "#059669", "#d97706", "#db2777", "#7c3aed", "#ea580c", "#0d9488", "#64748b"]
METHOD = {
    "random_score": "Random score", "squared_diff": "Squared difference", "moving_variance": "Moving variance",
    "rolling_zscore": "Rolling z-score", "static_limits": "Fixed limits", "isolation_forest": "Isolation forest",
    "random_forest": "Random forest", "v1_quorum": "V1 anomaly vote", "v1_ensemble": "V1 ensemble",
    "v1_ensemble_nocoh": "V1 ensemble (no coherence)", "v1_ensemble_basic": "V1 ensemble (basic features)",
    "forecast_persistence": "Persistence forecast", "forecast_linear": "Linear forecast",
    "v1_gru_attention": "GRU-attention forecast", "v1_bilstm_attention": "V1 BiLSTM-attention forecast",
}
FUSION = {"C alone": "Classifier", "A alone": "Anomaly", "F alone": "Forecast", "hand (v1 0.40/0.30/0.30)":
          "V1 weights", "equal (1/3 each)": "Equal", "max (any module)": "Max", "stacking (learned)": "Stacking"}
CALIB = {"cv_f1": "Fixed (CV-F1)", "budget": "Fixed (budget)", "adaptive": "Adaptive"}
DS = {"gecco2018": "GECCO 2018", "gecco2019": "GECCO 2019"}


def _csv(*parts) -> pd.DataFrame | None:
    p = OUTPUT_DIR.joinpath(*parts)
    try:
        return pd.read_csv(p) if p.exists() else None
    except Exception:  # noqa: BLE001
        return None


def _snap() -> dict:
    return json.loads((PACKAGE_DIR / "app" / "snapshot.json").read_text(encoding="utf-8"))


def _method(s: str) -> str:
    return METHOD.get(s, s)


# ---------------------------------------------------------------------------------------------- tables

def tables() -> list[tuple[str, str, pd.DataFrame, str]]:
    """(id, caption, frame, source)."""
    snap = _snap()
    out = []

    a = _csv("s1_audit", "gecco2018", "results.csv")
    if a is not None:
        a = a.copy()
        a["Protocol"] = a["protocol"].map({"A_random": "A: random split", "B_half": "B: half split",
                                          "C_oracle": "C: test-tuned", "D_honest": "D: time-ordered (honest)"})
        t = a.pivot_table(index="signal", columns="Protocol", values="event_f1").round(2)
        t.index = [_method(i) for i in t.index]
        t = t.reset_index().rename(columns={"index": "Method"})
        src = "outputs_v2/s1_audit/gecco2018/results.csv"
    else:
        lk = snap["leakage"]
        t = pd.DataFrame([{"Method": "Random forest", "A: random split": lk["random_split_event_f1"],
                           "D: time-ordered (honest)": lk["honest_event_f1"]}])
        src = "snapshot"
    out.append(("T1", "Event F1 of each detector under four evaluation protocols (GECCO 2018). Only protocol D "
                      "is free of leakage.", t, src))

    rows = []
    for ds in DS:
        for seg in ("s1_baselines", "s2_v1_port", "s3_forecast"):
            d = _csv(seg, ds, "results.csv")
            if d is None:
                continue
            d = d[(d["honest"].astype(str) == "True") & ~d["signal"].str.contains(":")]
            for _, r in d.iterrows():
                rows.append({"Dataset": DS[ds], "Method": _method(r["signal"]),
                             "Events detected": f"{int(r['detected'])}/{int(r['events'])}",
                             "Event recall": round(r["event_recall"], 2),
                             "False alarms/day": round(r["false_alarms_per_day"], 2),
                             "Median delay (min)": r["median_delay_min"], "Event F1": round(r["event_f1"], 2)})
    t = pd.DataFrame(rows).drop_duplicates(["Dataset", "Method"]) if rows else pd.DataFrame(
        [{"Dataset": DS[k], "Method": _method(m), "Event F1": v} for k, key in (("gecco2018", "baselines_2018"),
         ("gecco2019", "baselines_2019")) for m, v in snap["leakage"][key].items()])
    out.append(("T2", "Detectors on the official test periods with honestly chosen thresholds.", t,
                "outputs_v2/s1_baselines, s2_v1_port, s3_forecast" if rows else "snapshot"))

    t = pd.DataFrame([{"Version 1 component": r["part"], "Result on real utility data": r["result"]}
                      for r in snap["paper1_on_real_data"]])
    out.append(("T3", "Version 1 components re-tested on real utility data.", t, "snapshot (S2–S7 reports)"))

    rows = []
    for ds in DS:
        d = _csv("s4_triage", ds, "summary.csv")
        if d is None:
            continue
        for _, r in d.iterrows():
            rows.append({"Dataset": DS[ds], "System": r["label"],
                         "Real events": f"{int(r['real_caught'])}/{int(r['real_n'])}",
                         "Contaminations": f"{int(r['contaminant_caught'])}/{int(r['contaminant_n'])}",
                         "Faults to maintenance": f"{int(r['fault_correct'])}/{int(r['fault_n'])}",
                         "Faults raising water alarm": int(r["fault_raised_water_alarm"]),
                         "Water false alarms/day": round(r["water_false_alarms_per_day"], 2)})
    if not rows:
        for y, v in snap["triage"].items():
            rows.append({"Dataset": f"GECCO {y}", "System": "Detector only",
                         "Faults raising water alarm": v["detector_only"]["faults_to_water"]})
            rows.append({"Dataset": f"GECCO {y}", "System": "Detector + rule checks",
                         "Faults raising water alarm": v["rules"]["faults_to_water"]})
    out.append(("T4", "Water event or sensor fault: routing of real events, injected contaminations and injected "
                      "sensor faults.", pd.DataFrame(rows), "outputs_v2/s4_triage/*/summary.csv"))

    rows = []
    for ds in DS:
        d = _csv("s5_fusion", ds, "fusion_results.csv")
        if d is None:
            continue
        for _, r in d.iterrows():
            rows.append({"Dataset": DS[ds], "Fusion": FUSION.get(r["fusion"], r["fusion"]),
                         "Threshold": CALIB.get(r["calibrator"], r["calibrator"]),
                         "Real events": f"{int(r['real_caught'])}/{int(r['real_n'])}",
                         "False alarms/day": round(r["false_alarms_per_day"], 2),
                         "Worst week FA/day": round(r["worst_week_fa_per_day"], 2),
                         "Event F1": round(r["event_f1"], 2)})
    out.append(("T5", "False-alarm control: fusion rules under three threshold methods (budget = 1 per day).",
                pd.DataFrame(rows), "outputs_v2/s5_fusion/*/fusion_results.csv"))

    rows = []
    for ds in DS:
        d = _csv("s6_explain", ds, "explanation_summary.csv")
        if d is None:
            continue
        for _, r in d.iterrows():
            rows.append({"Dataset": DS[ds], "Source": r["source"], "Explainer": r["explainer"],
                         "Episodes": int(r["episodes"]),
                         "Hit@1": None if pd.isna(r["hit1"]) else round(r["hit1"], 2),
                         "Faithfulness": None if pd.isna(r["faithful1"]) else round(r["faithful1"], 2)})
    out.append(("T6", "Explanations: correct sensor named first (Hit@1) and deletion-test faithfulness.",
                pd.DataFrame(rows), "outputs_v2/s6_explain/*/explanation_summary.csv"))

    rows = []
    for ds in DS:
        d = _csv("s7_reports", ds, "report_summary.csv")
        if d is None:
            continue
        for _, r in d.iterrows():
            rows.append({"Dataset": DS[ds], "Writer": r["reporter"], "Reports": int(r["reports"]),
                         "Without factual error": round(r["faithful_share"], 2),
                         "Errors per report": round(r["errors_per_report"], 2),
                         "Median seconds": round(r["median_seconds"], 1)})
    llm = pd.DataFrame(snap["llm"])
    have = {(r["Dataset"], r["Writer"]) for r in rows}
    for _, r in llm.iterrows():
        key = (DS.get(r["station"], r["station"]), r["model"])
        if key not in have:
            rows.append({"Dataset": key[0], "Writer": r["model"], "Reports": int(r["reports"]),
                         "Without factual error": r["faithful"], "Median seconds": r["seconds"]})
    out.append(("T7", "Incident reports: share without factual error and writing time on a laptop CPU.",
                pd.DataFrame(rows), "outputs_v2/s7_reports/*/report_summary.csv + snapshot"))

    e = snap["edge"]
    d = _csv("s8_edge", "gecco2018", "edge_results.csv")
    if d is not None:
        t = pd.DataFrame([{"System": r["system"], "Real events": f"{int(r['real_caught'])}/{int(r['real_n'])}",
                           "Contaminations": f"{int(r['contaminant_caught'])}/{int(r['contaminant_n'])}",
                           "Faults to maintenance": f"{int(r['fault_correct'])}/{int(r['fault_n'])}",
                           "Water false alarms/day": round(r["water_false_alarms_per_day"], 2)}
                          for _, r in d.iterrows()])
        out.append(("T8", "Edge configurations on GECCO 2018 compared with the laptop pipeline.", t,
                    "outputs_v2/s8_edge/gecco2018/edge_results.csv"))
    t = pd.DataFrame([
        {"Quantity": "Model weights", "Value": f"{e['weights']} (int8, {e['int8_bytes']} bytes)"},
        {"Quantity": "Program memory (flash)", "Value": f"{e['flash_bytes']:,} of 32,256 bytes"},
        {"Quantity": "RAM (static)", "Value": f"{e['ram_bytes']:,} of 2,048 bytes; {e['ram_never_used']} never touched"},
        {"Quantity": "Time per minute of data", "Value": f"{e['us_per_minute'] / 1000:.1f} ms (max {e['max_us'] / 1000:.1f} ms)"},
        {"Quantity": "Minutes compared with the laptop", "Value": f"{e['uno_minutes']:,}, bit-identical: {e['bit_identical']:.0%}"},
        {"Quantity": "USB round trip", "Value": f"{e['round_trip_ms']} ms"},
        {"Quantity": "Size vs V1 BiLSTM", "Value": f"{e['bilstm_weights'] // e['weights']}× fewer weights"},
    ])
    out.append(("T9", "Arduino Uno deployment (measured on the real board).", t, "snapshot (S8 hardware run)"))
    return out


def _latex(t: pd.DataFrame, tid: str, caption: str) -> str:
    cols = "l" * len(t.columns)
    body = t.to_latex(index=False, escape=True, na_rep="--", column_format=cols)
    body = body.replace("\\toprule", "\\toprule").replace("\\midrule", "\\midrule")
    return (f"% {tid}\n\\begin{{table}}[t]\n\\centering\\scriptsize\n\\caption{{{caption}}}\n\\label{{tab:{tid}}}\n"
            f"{body}\\end{{table}}\n")


# ---------------------------------------------------------------------------------------------- figures

def _style() -> None:
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 100})


def _save(fig, out: Path, name: str) -> list[str]:
    fig.savefig(out / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    return [f"{name}.png", f"{name}.pdf"]


def fig_architecture(out: Path) -> list[str]:
    fig, ax = plt.subplots(figsize=(7.2, 3.7))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 54)
    ax.axis("off")

    def box(x, y, w, h, title, text, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", linewidth=1.1,
                                    edgecolor=color, facecolor=color + "1a"))
        nt = title.count("\n") + 1
        ax.text(x + w / 2, y + h - 1.6, title, ha="center", va="top", fontsize=6.6, fontweight="bold", color=color,
                linespacing=1.15)
        ax.text(x + w / 2, y + h - 2.4 - 3.0 * nt, text, ha="center", va="top", fontsize=5.5, color="#1f2937",
                linespacing=1.3)

    def arrow(x0, y0, x1, y1, color="#475569", ls="-"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8, linewidth=1.0,
                                     color=color, linestyle=ls))

    box(0.5, 30, 17, 18, "Utility station", "9 sensors, one row\nper minute: pH, chlorine,\nredox, conductivity,\n"
        "turbidity, temperature,\nflow (GECCO 2018/19)", "#0891b2")
    box(0.5, 6, 17, 18, "Demo station", "Arduino Uno with\nconductivity probe,\ntouch pad and DHT11\n"
        "(USB, every 2 s)", "#059669")
    stages = [("1  Sensor\nvalidation", "stuck · drop-out ·\nspike → sensor-\nfault alert", "#d97706"),
              ("2  Change\nfeatures", "15- and 60-min\nchange vs normal\n(integer counts)", "#4f46e5"),
              ("3  Anomaly\nscore", "int8 MLP\n29-16-8-1\n625 weights", "#4f46e5"),
              ("4  Adaptive\nthreshold", "about 1 false\nalarm per day;\n+0.5 per rejection", "#7c3aed")]
    x0, w, gap = 23, 16, 2.4
    for i, (t, d, c) in enumerate(stages):
        x = x0 + i * (w + gap)
        box(x, 31, w, 16, t, d, c)
        if i:
            arrow(x - gap + 0.2, 39, x - 0.2, 39)
    arrow(18.2, 39, 22.6, 39)
    arrow(18.2, 15, 22.6, 33)
    box(23, 7, 30, 16, "Decision", "water-quality alarm · sensor-fault\nalert · normal operation\n"
        "explained by the sensor that moved most", "#dc2626")
    last_x = x0 + 3 * (w + gap)
    arrow(last_x + w / 2, 30.4, 47, 23.6)
    box(57, 7, 19, 16, "Operator\nreview", "dashboard or touch pad:\nconfirm or reject", "#7c3aed")
    box(79.5, 7, 19.5, 16, "Incident\nreport", "template or local LLM,\nevery fact checked", "#db2777")
    arrow(53.6, 15, 56.4, 15)
    arrow(76.6, 15, 79, 15)
    arrow(70, 23.6, last_x + w * 0.75, 30.4, color="#7c3aed", ls="--")
    ax.text(73.5, 25.6, "review feedback", fontsize=5.4, color="#7c3aed", rotation=38)
    ax.add_patch(FancyBboxPatch((21.8, 29.6), 74.6, 18.9, boxstyle="round,pad=0.2,rounding_size=1.5",
                                linewidth=0.9, edgecolor="#64748b", facecolor="none", linestyle=":"))
    ax.text(96.4, 50.3, "stages 1–4 run bit-identically on the laptop and on the Arduino Uno", ha="right",
            va="bottom", fontsize=5.6, color="#64748b", style="italic")
    ax.text(50, 1.5, "Evaluation: time-ordered test periods · injected contaminations and sensor faults with known "
                     "times · every part optional with an automatic replacement", ha="center", fontsize=5.6,
            color="#475569")
    return _save(fig, out, "F1_architecture")


def fig_protocols(out: Path) -> list[str]:
    a = _csv("s1_audit", "gecco2018", "results.csv")
    if a is None:
        return []
    a = a[a["signal"] != "random_score"].copy()
    piv = a.pivot_table(index="signal", columns="protocol", values="event_f1")
    piv = piv.loc[piv["D_honest"].sort_values().index]
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    y = np.arange(len(piv))
    for i, (p, lab, c) in enumerate((("A_random", "A random split", "#94a3b8"), ("C_oracle", "C test-tuned", "#f59e0b"),
                                     ("D_honest", "D time-ordered", "#4f46e5"))):
        ax.barh(y + (i - 1) * 0.26, piv[p], height=0.26, color=c, label=lab)
    ax.set_yticks(y, [_method(s) for s in piv.index])
    ax.set_xlabel("event F1")
    ax.set_title("Evaluation protocol vs score (GECCO 2018)")
    ax.legend(loc="lower right", frameon=False)
    return _save(fig, out, "F2_protocols")


def fig_detection(out: Path) -> list[str]:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
    done = False
    for ax, ds in zip(axes, DS):
        frames = [_csv(s, ds, "results.csv") for s in ("s1_baselines", "s2_v1_port", "s3_forecast")]
        frames = [f for f in frames if f is not None]
        if not frames:
            continue
        d = pd.concat(frames).drop_duplicates("signal")
        d = d[(d["honest"].astype(str) == "True") & ~d["signal"].str.contains(":")]
        label = set(d.sort_values("event_f1", ascending=False)["signal"].head(4)) | {"v1_quorum", "random_score"}
        for i, (_, r) in enumerate(d.iterrows()):
            ax.scatter(max(r["false_alarms_per_day"], 0.01), r["event_recall"], s=20 + 160 * (r["event_f1"] or 0),
                       color=PAL[i % len(PAL)], alpha=0.8, edgecolor="white", linewidth=0.6)
            if r["signal"] in label:
                ax.annotate(_method(r["signal"]), (max(r["false_alarms_per_day"], 0.01), r["event_recall"]),
                            fontsize=5.5, xytext=(4, -3), textcoords="offset points")
        ax.set_xscale("log")
        ax.axvspan(0.01, 1, color="#10b981", alpha=0.07)
        ax.set_xlabel("false alarms per day (log)")
        ax.set_title(DS[ds])
        done = True
    axes[0].set_ylabel("share of events detected")
    if not done:
        plt.close(fig)
        return []
    return _save(fig, out, "F3_detection")


def fig_routing(out: Path) -> list[str]:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4), sharey=True)
    done = False
    for ax, ds in zip(axes, DS):
        d = _csv("s4_triage", ds, "summary.csv")
        if d is None:
            continue
        cols = [("real_caught", "Real events caught"), ("contaminant_caught", "Contaminations caught"),
                ("fault_correct", "Faults to maintenance"), ("fault_raised_water_alarm", "Faults → water alarm")]
        x = np.arange(len(cols))
        for i, (_, r) in enumerate(d.iterrows()):
            ax.bar(x + (i - 1) * 0.27, [r[c] for c, _ in cols], width=0.27, color=["#94a3b8", "#4f46e5", "#7c3aed"][i % 3],
                   label=r["label"])
        ax.set_xticks(x, [lab for _, lab in cols], rotation=15)
        ax.set_title(DS[ds])
        done = True
    if not done:
        plt.close(fig)
        return []
    axes[0].set_ylabel("count")
    axes[0].legend(frameon=False, fontsize=6)
    return _save(fig, out, "F4_routing")


def fig_weekly(out: Path) -> list[str]:
    w = _csv("s5_fusion", "gecco2018", "weekly_false_alarms.csv")
    if w is None:
        return []
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    t = pd.to_datetime(w["time"])
    for cal, c in (("cv_f1", "#94a3b8"), ("budget", "#f59e0b"), ("adaptive", "#4f46e5")):
        col = f"stacking (learned) | {cal}"
        if col in w:
            ax.plot(t, w[col], marker="o", ms=2.5, lw=1.4 if cal != "adaptive" else 2, color=c, label=CALIB[cal])
    ax.axhline(1, color="#dc2626", ls="--", lw=1)
    ax.set_ylabel("false alarms per day")
    ax.set_title("Weekly false alarms, stacking fusion (GECCO 2018)")
    ax.legend(frameon=False)
    fig.autofmt_xdate()
    return _save(fig, out, "F5_weekly_false_alarms")


def fig_explain(out: Path) -> list[str]:
    d = _csv("s6_explain", "gecco2018", "explanation_summary.csv")
    if d is None:
        return []
    piv = d.pivot_table(index="explainer", columns="source", values="faithful1")
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    x = np.arange(len(piv))
    for i, s in enumerate(piv.columns):
        ax.bar(x + (i - 1) * 0.27, piv[s], width=0.27, color=PAL[i], label=s)
    names = {"deviation": "Deviation", "forecast_residual": "Forecast error", "occlusion": "Occlusion",
             "random": "Random", "tree_shap": "TreeSHAP (V1)"}
    ax.set_xticks(x, [names.get(i, i) for i in piv.index], rotation=15)
    ax.set_ylabel("faithfulness")
    ax.set_ylim(0, 1.05)
    ax.set_title("Deletion-test faithfulness (GECCO 2018)", pad=16)
    ax.legend(frameon=False, fontsize=6, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    return _save(fig, out, "F6_explanations")


def fig_edge(out: Path) -> list[str]:
    d = _csv("s8_edge", "gecco2018", "edge_results.csv")
    if d is None:
        return []
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    y = np.arange(len(d))
    ax.barh(y, d["water_false_alarms_per_day"], color=["#dc2626" if v > 1.5 else "#059669" if v < 0.6 else "#f59e0b"
                                                      for v in d["water_false_alarms_per_day"]])
    ax.set_yticks(y, d["system"].str.replace("S5 laptop pipeline (stacking (learned))", "laptop pipeline",
                                             regex=False), fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("water false alarms per day")
    ax.set_title("Edge configurations (GECCO 2018)")
    return _save(fig, out, "F7_edge")


# ---------------------------------------------------------------------------------------------- build

def build(out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    _style()
    md, tex, index = ["# Paper tables", ""], ["% Paper tables – generated by python -m ewsv2 paper", ""], [
        "# Paper tables and figures", "", "| Item | Content | Source |", "|---|---|---|"]
    for tid, caption, t, src in tables():
        if t is None or t.empty:
            continue
        t.to_csv(out / "tables" / f"{tid}.csv", index=False)
        md += [f"## {tid}. {caption}", "", _md(t),
               "", f"*Source: {src}*", ""]
        tex.append(_latex(t, tid, caption))
        index.append(f"| {tid} | {caption} | {src} |")
        print(f"  {tid}: {len(t)} rows – {caption[:70]}")
    figs = []
    for fn, cap in ((fig_architecture, "System architecture"), (fig_protocols, "Evaluation protocol vs score"),
                    (fig_detection, "Detection vs false alarms per method"),
                    (fig_routing, "Routing of events and sensor faults"),
                    (fig_weekly, "Weekly false alarms by threshold method"),
                    (fig_explain, "Explanation faithfulness"), (fig_edge, "Edge configurations")):
        try:
            names = fn(out / "figures")
        except Exception as exc:  # noqa: BLE001 – one damaged file must not stop the others
            print(f"  ! {cap}: skipped ({type(exc).__name__}: {exc})")
            continue
        if names:
            figs.append(names[0])
            index.append(f"| {names[0].split('_')[0]} | {cap} | figures/{names[0]} (+ .pdf) |")
            print(f"  {names[0].split('_')[0]}: {cap}")
    (out / "tables.md").write_text("\n".join(md), encoding="utf-8")
    (out / "tables.tex").write_text("\n".join(tex), encoding="utf-8")
    (out / "index.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    return out


def _md(t: pd.DataFrame) -> str:
    """Markdown table without extra packages."""
    def cell(v) -> str:
        return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v).replace("|", "/")
    head = "| " + " | ".join(map(str, t.columns)) + " |"
    sep = "|" + "---|" * len(t.columns)
    rows = ["| " + " | ".join(cell(v) for v in r) + " |" for r in t.itertuples(index=False)]
    return "\n".join([head, sep, *rows])
