"""Command line: ``python -m ewsv2 <command>``.

Commands
--------
info                       version, registered plug-ins, data files found
fetch {gecco2018|gecco2019|all} [--force]
                           download the GECCO data into data/raw/gecco/
inspect --config NAME      data inspection report (Segment 0)
run --config NAME          run the pipeline steps built so far

``--set key=value`` overrides any config value, e.g.
``python -m ewsv2 inspect --config s0_inspect --set data.source=gecco2019``
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time

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


def cmd_run(args) -> int:
    cfg = load_config(args.config, parse_overrides(args.set))
    ds = _load_dataset(cfg)
    steps = cfg.get("pipeline", ["load"])
    done = ["load"]
    pending = [s for s in steps if s not in done]
    print(f"\nRun '{cfg['run_name']}' (segment {SEGMENT}). Steps done: {', '.join(done)}.")
    if pending:
        print(f"Steps not built yet (arrive in later segments): {', '.join(pending)}")
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
        ("run", cmd_run, "run the pipeline steps built so far"),
    ):
        c = sub.add_parser(name, help=helptext)
        c.add_argument("--config", required=True, help="config name in ewsv2/configs or a path")
        c.add_argument("--set", action="append", metavar="KEY=VALUE", help="override a config value")
        c.set_defaults(func=func)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    try:
        return args.func(args)
    except (ConfigError, RegistryError, FileNotFoundError) as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 2
