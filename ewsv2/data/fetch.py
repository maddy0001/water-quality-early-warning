"""Download the GECCO datasets into ``data/raw/gecco/``.

* ``gecco2018``: the two R data files bundled in EventDetectR on GitHub
  (identical to the official 2018 train/test sets), converted to CSV.
* ``gecco2019``: the train / validation / test CSV files from Zenodo record 4304080.

Both are free. If a download is blocked (proxy, firewall), the error message
says which file to download by hand and where to put it.
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import urllib.request
from pathlib import Path

from ewsv2.core.paths import DATA_DIR
from ewsv2.data.base import read_table, standardise
from ewsv2.data.gecco import GECCO2018_SENSORS

log = logging.getLogger(__name__)

EVENTDETECTR_RAW = "https://raw.githubusercontent.com/frehbach/EventDetectR/master/data/{name}"
GECCO2018_FILES = {"train": "geccoIC2018Train.rda", "test": "geccoIC2018Test.rda"}

ZENODO_2019_API = "https://zenodo.org/api/records/4304080"
ZENODO_2019_FILE = "https://zenodo.org/records/4304080/files/{name}?download=1"
# One file holds train + validation + test; the loader splits it at the official boundaries.
GECCO2019_FILE = "5_gecco2019_all_water_quality.csv"

_UA = {"User-Agent": "ewsv2-fetch/1.0 (research use)"}


class FetchError(RuntimeError):
    pass


def _download(url: str, dest: Path, timeout: int = 120) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers=_UA)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp, tempfile.NamedTemporaryFile(
            delete=False, dir=dest.parent, suffix=".part"
        ) as tmp:
            shutil.copyfileobj(resp, tmp)
            tmp_path = Path(tmp.name)
    except Exception as exc:  # network errors vary by platform
        raise FetchError(f"Could not download {url}: {exc}") from exc
    tmp_path.replace(dest)
    return dest


def fetch_gecco2018(data_dir: Path = DATA_DIR, force: bool = False) -> list[Path]:
    """Download, convert to CSV, and return the two CSV paths."""
    data_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for split, rda_name in GECCO2018_FILES.items():
        csv_path = data_dir / f"gecco2018_{split}.csv"
        if csv_path.is_file() and not force:
            log.info("Already present: %s", csv_path)
            out.append(csv_path)
            continue
        with tempfile.TemporaryDirectory() as tmp:
            rda = Path(tmp) / rda_name
            try:
                _download(EVENTDETECTR_RAW.format(name=rda_name), rda)
            except FetchError as exc:
                raise FetchError(
                    f"{exc}\nManual fix: download {EVENTDETECTR_RAW.format(name=rda_name)} "
                    f"into {data_dir}, or the GECCO 2018 CSV from https://zenodo.org/records/3884398 "
                    f"(the loader also reads those files directly)."
                ) from exc
            frame, _, _ = standardise(read_table(rda), GECCO2018_SENSORS)
        frame.drop(columns=["LABEL_CLASS"]).to_csv(csv_path, date_format="%Y-%m-%d %H:%M:%S")
        log.info("Wrote %s (%d rows)", csv_path, len(frame))
        out.append(csv_path)
    return out


def _zenodo_links() -> dict[str, str]:
    req = urllib.request.Request(ZENODO_2019_API, headers=_UA)
    with urllib.request.urlopen(req, timeout=60) as resp:
        record = json.load(resp)
    links = {}
    for f in record.get("files", []):
        key = f.get("key") or f.get("filename")
        link = (f.get("links") or {}).get("self") or (f.get("links") or {}).get("download")
        if key and link:
            links[key] = link
    return links


def fetch_gecco2019(data_dir: Path = DATA_DIR, force: bool = False) -> list[Path]:
    """Download the GECCO 2019 'all' file (train + validation + test) from Zenodo."""
    data_dir.mkdir(parents=True, exist_ok=True)
    existing = [p for p in data_dir.glob("*gecco2019_all*") if p.is_file()]
    if existing and not force:
        log.info("Already present: %s", existing[0])
        return existing[:1]
    try:
        links = _zenodo_links()
    except Exception as exc:
        log.warning("Zenodo API not reachable (%s); trying the direct file link", exc)
        links = {}
    # Use the API's name for the "all" file if it differs slightly from ours.
    name = next((k for k in links if "gecco2019_all" in k), GECCO2019_FILE)
    url = links.get(name, ZENODO_2019_FILE.format(name=name))
    dest = data_dir / name
    try:
        _download(url, dest)
    except FetchError as exc:
        raise FetchError(
            f"{exc}\nManual fix: open https://zenodo.org/records/4304080, download {GECCO2019_FILE} "
            f"and save it in {data_dir}"
        ) from exc
    log.info("Wrote %s", dest)
    return [dest]


FETCHERS = {"gecco2018": fetch_gecco2018, "gecco2019": fetch_gecco2019}
