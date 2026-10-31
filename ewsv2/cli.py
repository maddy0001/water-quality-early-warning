"""Command line: ``python -m ewsv2 <command>``.

Commands
--------
info                       version, registered plug-ins, data files found
fetch {gecco2018|gecco2019|all} [--force]
                           download the GECCO data into data/raw/gecco/
inspect --config NAME      data inspection report (Segment 0)
evaluate --config NAME     run signals under evaluation protocols
                           (s1_audit, s1_baselines, s2_v1_port, s3_forecast, ...)
triage --config NAME       GECCO-Triage benchmark + water-vs-sensor systems (s4_triage)
inject --config NAME       only write the GECCO-Triage benchmark files
fuse --config NAME         fusion + calibration under a false-alarm budget (s5_fusion)
explain --config NAME      explanations checked against known answers (s6_explain; run fuse first)
transfer --config NAME     train on GECCO 2018, run on GECCO 2019 (s6_explain)
report --config NAME       incident reports by a local LLM, fact-checked (s7_reports)
edge --config NAME         Arduino Uno edge: reference, PC build, simulated Uno, USB replay (s8_edge)
board [--config s8_edge]   find the Arduino, say which sensors answer and what the system will use
run --config NAME          run the pipeline steps built so far

``--set key=value`` overrides any config value, e.g.
``python -m ewsv2 inspect --config s0_inspect --set data.source=gecco2019``
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import numpy as np

from ewsv2 import SEGMENT, __version__
from ewsv2.core.config import ConfigError, load_config, parse_overrides
from ewsv2.core.paths import DATA_DIR, OUTPUT_DIR, output_dir
from ewsv2.core.registry import RegistryError, available, build

log = logging.getLogger("ewsv2")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )


def _load_dataset(cfg: dict):
    data = cfg["data"]
    source = build("source", data["source"], **(data.get("params") or {}))
    t0 = time.perf_counter()
    try:
        ds = source.load()
    except FileNotFoundError as exc:
        fallback = data.get("fallback")
        if not fallback:
            raise
        print(f"! {data['source']} data not found – using the '{fallback}' source instead "
              f"(results are about that source, not about {data['source']}).\n  ({str(exc).splitlines()[0]})")
        source = build("source", fallback, **(data.get("fallback_params") or {}))
        ds = source.load()
    log.info("Loaded %s in %.1f s: %s", data["source"], time.perf_counter() - t0, ds)
    return ds


def cmd_info(args) -> int:
    print(f"ewsv2 {__version__}  (built up to segment {SEGMENT})")
    print(f"data folder   : {DATA_DIR}")
    files = sorted(p.name for p in DATA_DIR.glob("*")) if DATA_DIR.is_dir() else []
    print(f"data files    : {', '.join(files) if files else '(none - run: python -m ewsv2 fetch gecco2018)'}")
    print(f"output folder : {OUTPUT_DIR}")
    print("registered plug-ins:")
    for kind, names in available().items():
        print(f"  {kind:<11} {', '.join(names) if names else '-'}")
    return 0


def cmd_fetch(args) -> int:
    from ewsv2.data.fetch import FETCHERS, FetchError

    names = list(FETCHERS) if args.dataset == "all" else [args.dataset]
    failed = False
    for name in names:
        try:
            paths = FETCHERS[name](force=args.force)
            for p in paths:
                print(f"  {name}: {p}")
        except FetchError as exc:
            failed = True
            print(f"\n[{name}] download failed:\n{exc}\n", file=sys.stderr)
    return 1 if failed else 0


def cmd_inspect(args) -> int:
    from ewsv2.data.inspect import inspect_dataset

    cfg = load_config(args.config, parse_overrides(args.set))
    ds = _load_dataset(cfg)
    opts = cfg.get("inspect") or {}
    out = output_dir(cfg["run_name"], ds.name)
    summary = inspect_dataset(
        ds, out,
        baseline_minutes=int(opts.get("baseline_minutes", 60)),
        shift_threshold=float(opts.get("shift_threshold_sd", 1.0)),
        make_plots=bool(opts.get("plots", True)),
    )
    print()
    print(f"Dataset      : {summary['dataset']}")
    print(f"Rows         : {summary['sampling']['rows']:,}  ({summary['sampling']['start']} -> {summary['sampling']['end']})")
    for name, p in summary["periods"].items():
        print(f"  {name:<6}: {p['rows']:,} rows, {p['events']} events, {p['event_minutes']:,} event minutes "
              f"({p['event_minute_pct']}%)")
    print(f"Events total : {summary['events_total']}  (median length {summary['event_length_min']['median']} min)")
    print(f"Top sensors during events: {', '.join(summary['event_response_top_sensors'])}")
    print(f"Report       : {out / 'report.md'}")
    return 0


def cmd_evaluate(args) -> int:
    from ewsv2.evaluate.audit import HEADLINE_NAMES, run_protocols

    cfg = load_config(args.config, parse_overrides(args.set))
    if "evaluate" not in cfg:
        raise ConfigError(f"{cfg['_file']} has no 'evaluate:' section")
    ds = _load_dataset(cfg)
    out = output_dir(cfg["run_name"], ds.name)
    t0 = time.perf_counter()
    results = run_protocols(ds, cfg, out)
    print(f"\n{cfg.get('title', 'Evaluation')} on {ds.name} ({time.perf_counter() - t0:.0f} s)")
    for proto, sub in results.groupby("protocol", sort=False):
        metric = sub["headline_metric"].iloc[0]
        print(f"\n{sub['protocol_label'].iloc[0]}  [{HEADLINE_NAMES.get(metric, metric)}]")
        width = max(17, int(sub["signal"].str.len().max()))
        print(f"  {'signal':<{width}} {'headline':>8} {'caught':>8} {'recall':>7} {'FA/day':>7} {'delay':>6}")
        for _, r in sub.iterrows():
            delay = "" if not np.isfinite(r["median_delay_min"]) else f"{r['median_delay_min']:.0f}"
            print(f"  {r['signal']:<{width}} {r['headline_value']:>8.2f} {int(r['detected']):>3}/{int(r['events']):<4} "
                  f"{r['event_recall']:>7.2f} {r['false_alarms_per_day']:>7.2f} {delay:>6}")
    print(f"\nReport : {out / 'report.md'}")
    return 0


def cmd_triage(args) -> int:
    from ewsv2.triage.run import run_triage

    cfg = load_config(args.config, parse_overrides(args.set))
    ds = _load_dataset(cfg)
    out = output_dir(cfg["run_name"], ds.name)
    t0 = time.perf_counter()
    res = run_triage(ds, cfg, out)
    summ = res["summary"]
    print(f"\n{cfg.get('title', 'Triage')} on {ds.name} ({time.perf_counter() - t0:.0f} s), official test period")
    width = max(31, int(summ["label"].str.len().max()))
    print(f"  {'system':<{width}} {'real':>7} {'contam.':>8} {'waterFA/d':>9} | {'fault->sensor':>13} "
          f"{'fault->water':>12} {'missed':>6} {'sensorFA/d':>10}")
    for _, r in summ.iterrows():
        print(f"  {r['label']:<{width}} {r['real_caught']:>3}/{r['real_n']:<3} {r['contaminant_caught']:>4}/{r['contaminant_n']:<3} "
              f"{r['water_false_alarms_per_day']:>9.2f} | {r['fault_correct']:>9}/{r['fault_n']:<3} "
              f"{r['fault_raised_water_alarm']:>8}/{r['fault_n']:<3} {r['fault_missed']:>6} "
              f"{r['false_sensor_alerts_per_day']:>10.2f}")
    scan = res["scan"]
    if len(scan):
        top = scan.sort_values("span_minutes", ascending=False).head(3)
        print("\nLongest fault flags on the real data (nothing injected):")
        for _, r in top.iterrows():
            print(f"  {r['check']:<7} {r['sensor']:<6} from {r['start']:%Y-%m-%d %H:%M}  {r['span_minutes']:>5} min  ({r['period']})")
    print(f"\nReport : {out / 'report.md'}")
    return 0


def cmd_fuse(args) -> int:
    from ewsv2.fusion.run import run_fusion

    cfg = load_config(args.config, parse_overrides(args.set))
    ds = _load_dataset(cfg)
    out = output_dir(cfg["run_name"], ds.name)
    t0 = time.perf_counter()
    res = run_fusion(ds, cfg, out)
    r = res["results"]
    b = res["info"]["budget"]
    print(f"\n{cfg.get('title', 'Fusion')} on {ds.name} ({time.perf_counter() - t0:.0f} s), official test period, "
          f"budget {b:g} false alarm/day")
    width = max(26, int(r["fusion"].str.len().max()))
    print(f"  {'fusion':<{width}} {'threshold':<9} {'real':>7} {'contam.':>8} {'FA/day':>7} {'worst wk':>8} {'F1':>5}")
    for _, x in r.iterrows():
        print(f"  {x['fusion']:<{width}} {x['calibrator']:<9} {x['real_caught']:>3}/{x['real_n']:<3} "
              f"{x['contaminant_caught']:>4}/{x['contaminant_n']:<3} {x['false_alarms_per_day']:>7.2f} "
              f"{x['worst_week_fa_per_day']:>8.2f} {x['event_f1']:>5.2f}")
    print(f"\nFull pipeline: {res['info']['main_fusion']} (chosen on pre-test data) + adaptive threshold:")
    for p in res["pipelines"]:
        print(f"  {p['variant']:<35} real {p['real_caught']:>2}/{p['real_n']:<2} contam. {p['contaminant_caught']:>2}/"
              f"{p['contaminant_n']:<2} waterFA/d {p['water_false_alarms_per_day']:.2f} | faults->sensor "
              f"{p['fault_correct']:>2}/{p['fault_n']} ->water {p['fault_raised_water_alarm']:>2} | "
              f"sensorFA/d {p['false_sensor_alerts_per_day']:.2f}")
    print(f"\nReport : {out / 'report.md'}")
    return 0


def cmd_explain(args) -> int:
    from ewsv2.explain.run import LABELS, run_explain

    cfg = load_config(args.config, parse_overrides(args.set))
    ds = _load_dataset(cfg)
    out = output_dir(cfg["run_name"], ds.name)
    t0 = time.perf_counter()
    res = run_explain(ds, cfg, out)
    s = res["summary"]
    print(f"\nExplanations on {ds.name} ({time.perf_counter() - t0:.0f} s), first alarm minute of each alarmed test episode")
    for src in s["source"].unique():
        sub = s[s["source"] == src]
        print(f"\n  {src}s ({int(sub['episodes'].iloc[0])})")
        print(f"  {'explainer':<26} {'hit@1':>6} {'lead@1':>7} {'cover':>6} {'faithful@1':>11}")
        for _, r in sub.iterrows():
            vals = [r["hit1"], r["lead1"], r["cover"], r["faithful1"]]
            txt = [f"{v:.2f}" if np.isfinite(v) else "" for v in vals]
            print(f"  {LABELS[r['explainer']]:<26} {txt[0]:>6} {txt[1]:>7} {txt[2]:>6} {txt[3]:>11}")
    print(f"\nReport : {out / 'report.md'}")
    return 0


def cmd_transfer(args) -> int:
    from ewsv2.explain.transfer import run_transfer

    cfg = load_config(args.config, parse_overrides(args.set))
    t = cfg.get("transfer") or {}
    out = output_dir(cfg["run_name"], f"transfer_{t.get('source', 'gecco2018')}_to_{t.get('target', 'gecco2019')}")
    t0 = time.perf_counter()
    res = run_transfer(cfg, out)
    print(f"\nTransfer {t.get('source', 'gecco2018')} -> {t.get('target', 'gecco2019')} ({time.perf_counter() - t0:.0f} s), "
          f"target official test period")
    width = int(res["system"].str.len().max())
    print(f"  {'system':<{width}} {'threshold':<27} {'caught':>7} {'FA/day':>7} {'F1':>5}")
    for _, r in res.iterrows():
        print(f"  {r['system']:<{width}} {r['threshold']:<27} {r['caught']:>3}/{r['events']:<3} "
              f"{r['false_alarms_per_day']:>7.2f} {r['event_f1']:>5.2f}")
    print(f"\nReport : {out / 'report.md'}")
    return 0


def cmd_report(args) -> int:
    from ewsv2.llm.run import run_reports

    cfg = load_config(args.config, parse_overrides(args.set))
    ds = _load_dataset(cfg)
    out = output_dir(cfg["run_name"], ds.name)
    t0 = time.perf_counter()
    res = run_reports(ds, cfg, out)
    s = res["summary"]
    print(f"\nIncident reports on {ds.name} ({time.perf_counter() - t0:.0f} s): "
          f"{len(res['records'])} alarm records, {int(s['reports'].iloc[0])} reported")
    for name, why in res["status"].items():
        if why != "ok":
            print(f"  ! {name} skipped: {why}")
    print(f"  {'reporter':<22} {'faithful':>8} {'err/rep':>8} {'decision':>8} {'time':>5} {'number':>6} "
          f"{'sensor':>6} {'cite':>5} {'top sensor':>10} {'sec':>5}")
    for _, r in s.iterrows():
        print(f"  {r['reporter']:<22} {r['faithful_share']:>8.2f} {r['errors_per_report']:>8.2f} "
              f"{r['with_decision_error']:>8.2f} {r['with_time_error']:>5.2f} {r['with_number_error']:>6.2f} "
              f"{r['with_sensor_error']:>6.2f} {r['with_citation_error']:>5.2f} {r['mentions_top_sensor']:>10.2f} "
              f"{r['median_seconds']:>5.1f}")
    print(f"\nReport : {out / 'report.md'}")
    return 0


def cmd_edge(args) -> int:
    from ewsv2.edge.run import run_edge_experiment

    cfg = load_config(args.config, parse_overrides(args.set))
    ds = _load_dataset(cfg)
    out = output_dir(cfg["run_name"], ds.name)
    t0 = time.perf_counter()
    res = run_edge_experiment(ds, cfg, out)
    r, info = res["results"], res["info"]
    m, u = info["model"], info.get("uno_sizes") or {}
    print(f"\nEdge algorithm on {ds.name} ({time.perf_counter() - t0:.0f} s): {m.get('params')} weights, "
          f"int8 {m.get('int8_bytes')} B, float32 {m.get('float32_bytes')} B, TFLite int8 {m.get('tflite_int8_bytes', '-')} B")
    if u:
        print(f"Uno firmware: {u.get('flash_bytes')} B flash of 32256, {u.get('ram_static_bytes')} B RAM of 2048 before the stack")
    print(f"  {'system':<44} {'real':>7} {'contam.':>8} {'faults->sensor':>14} {'->water':>7} {'waterFA/d':>9} "
          f"{'sensorFA/d':>10}")
    for _, x in r.iterrows():
        print(f"  {x['system']:<44} {int(x['real_caught']):>3}/{int(x['real_n']):<3} {int(x['contaminant_caught']):>4}/"
              f"{int(x['contaminant_n']):<3} {int(x['fault_correct']):>10}/{int(x['fault_n']):<3} "
              f"{int(x['fault_raised_water_alarm']):>7} {x['water_false_alarms_per_day']:>9.2f} "
              f"{x['false_sensor_alerts_per_day']:>10.2f}")
    q = info["int8_vs_float32"]
    print(f"  int8 vs float32: alarm agreement {q['alarm_agreement_int8_vs_float32']:.5f}")
    print("\nTargets (each answer compared with the laptop's reference, minute by minute):")
    for target, h in info["hil"].items():
        if "agreement" not in h:
            print(f"  {target:<8} {h.get('status')}")
            continue
        a, d = h["agreement"], h["device"]
        ram = f", RAM never used {d['free_ram']} B" if d.get("free_ram", -1) >= 0 else ""
        print(f"  {target:<8} {h['minutes']} min: alarm {a.get('alarm_agreement', 0):.5f} water {a.get('water_agreement', 0):.5f} "
              f"sensor {a.get('sensor_agreement', 0):.5f} top {a.get('top_agreement', 0):.5f} | bit-identical "
              f"{a.get('logit_bit_identical', 0):.5f} | {d.get('mean_us', 0):.0f} us/min (max {d.get('max_us', 0)}), "
              f"state {d.get('state_bytes')} B{ram}, {d.get('round_trip_ms', 0):.1f} ms round trip")
        if h["status"] != "ok":
            print(f"           {h['status']} – the laptop's answers stand for the rest")
    print(f"\nReport : {out / 'report.md'}")
    return 0


def cmd_board(args) -> int:
    from ewsv2.edge.board import detect

    cfg = load_config(args.config, parse_overrides(args.set))
    e, b = cfg.get("edge") or {}, cfg.get("board") or {}
    print("Looking for the Arduino ...")
    res = detect(e.get("port", "auto"), int(e.get("baud", 115200)), int(b.get("readings", 3)))
    print(f"  {res['message']}")
    if res["board"]:
        print("  parts: " + ", ".join(f"{k} {'yes' if v else 'no'}" for k, v in res["parts"].items()))
        for r in res["readings"]:
            dht = f"{r['temperature_c']} C, {r['humidity_pct']} %" if r["dht_status"] == 0 else "no answer"
            print(f"  probe {r['probe']:>4} (off {r['probe_off']}) | DHT11 {dht} | touches {r['touches']}")
    print("\nWhat the system will use:")
    for k, v in res["choice"].items():
        print(f"  {k:<6} {v}")
    return 0


def cmd_inject(args) -> int:
    from ewsv2.triage.run import build_benchmark

    cfg = load_config(args.config, parse_overrides(args.set))
    ds = _load_dataset(cfg)
    inj, episodes = build_benchmark(ds, cfg)
    out = output_dir("benchmarks")
    stream = out / f"gecco_triage_{ds.name}.csv"
    eps = out / f"gecco_triage_{ds.name}_episodes.csv"
    inj.frame.to_csv(stream, date_format="%Y-%m-%d %H:%M:%S")
    episodes.to_csv(eps, index=False)
    print(f"\nGECCO-Triage ({ds.name}): {len(episodes)} injected episodes "
          f"{episodes.groupby(['period', 'kind']).size().to_dict()}")
    print(f"Stream   : {stream}")
    print(f"Episodes : {eps}")
    return 0


def cmd_run(args) -> int:
    cfg = load_config(args.config, parse_overrides(args.set))
    ds = _load_dataset(cfg)
    steps = cfg.get("pipeline", ["load"])
    done = ["load", "split", "evaluate", "inject", "triage", "fuse", "calibrate", "explain"]  # via the commands above
    pending = [s for s in steps if s not in done]
    print(f"\nRun '{cfg['run_name']}' (segment {SEGMENT}). Steps done: {', '.join(done)}.")
    if pending:
        print(f"Steps not built yet (arrive in later segments): {', '.join(pending)}")
    print("Split and evaluation run through:  python -m ewsv2 evaluate --config s1_baselines")
    out = output_dir(cfg["run_name"])
    (out / "run_info.json").write_text(json.dumps({
        "config": cfg["_file"], "segment": SEGMENT, "dataset": ds.name,
        "rows": len(ds.frame), "sensors": ds.sensors, "steps_done": done, "steps_pending": pending,
    }, indent=2), encoding="utf-8")
    print(f"Wrote {out / 'run_info.json'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m ewsv2", description="Water EWS v2 command line")
    p.add_argument("-v", "--verbose", action="store_true", help="more log output")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="show version, plug-ins and data files").set_defaults(func=cmd_info)

    f = sub.add_parser("fetch", help="download GECCO data")
    f.add_argument("dataset", choices=["gecco2018", "gecco2019", "all"])
    f.add_argument("--force", action="store_true", help="download again even if present")
    f.set_defaults(func=cmd_fetch)

    for name, func, helptext in (
        ("inspect", cmd_inspect, "write the data inspection report"),
        ("evaluate", cmd_evaluate, "evaluate signals under one or more protocols"),
        ("triage", cmd_triage, "GECCO-Triage benchmark and the water-vs-sensor systems"),
        ("inject", cmd_inject, "write the GECCO-Triage benchmark files"),
        ("fuse", cmd_fuse, "fusion and calibration under a false-alarm budget"),
        ("explain", cmd_explain, "explanations checked against known answers"),
        ("transfer", cmd_transfer, "train on one station, run on another"),
        ("report", cmd_report, "incident reports by a local LLM, fact-checked"),
        ("edge", cmd_edge, "Arduino Uno edge: reference, PC build, simulated Uno, USB replay"),
        ("board", cmd_board, "find the Arduino and say which sensors answer"),
        ("run", cmd_run, "run the pipeline steps built so far"),
    ):
        c = sub.add_parser(name, help=helptext)
        if name == "board":
            c.add_argument("--config", default="s8_edge", help="config name (default s8_edge)")
        else:
            c.add_argument("--config", required=True, help="config name in ewsv2/configs or a path")
        c.add_argument("--set", action="append", metavar="KEY=VALUE", help="override a config value")
        c.set_defaults(func=func)
    return p


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # hide TensorFlow start-up chatter
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    try:
        return args.func(args)
    except (ConfigError, RegistryError, FileNotFoundError) as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 2
