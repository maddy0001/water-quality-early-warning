"""Offline time capsule: ``python -m ewsv2 capsule --dest E:\\ewsv2_capsule`` (Segment 10).

Writes everything needed to rebuild the project on a fresh Windows laptop *without internet* to one folder
(a pendrive):

    project/            the repository (version 1 + version 2), data, results, models – without the venvs
    locks/              exact package versions of venv2 (and of version 1's venv, if found)
    wheelhouse/v2, v1/  every package as a Windows (64-bit, Python 3.11) wheel
    ollama/models/      the local AI model (optional, --with-ollama)
    arduino/Arduino15/  the Arduino IDE board files (optional, --with-arduino)
    installers/         Python 3.11, Ollama and Arduino IDE installers (optional, --with-installers)
    RESTORE.md          the restore steps, in plain words
    restore.ps1         does the restore in one command
    manifest.json       size and SHA-256 of every file;  python -m ewsv2 capsule --check <folder> verifies it

Run it while you still have internet (it downloads the wheels). Nothing in the project is changed.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import time
import urllib.request
from importlib import metadata
from pathlib import Path

from ewsv2 import SEGMENT, __version__
from ewsv2.core.paths import REPO_ROOT

LOCK_DIR = Path(__file__).with_name("locks")
SKIP_DIRS = {"venv", "venv2", "__pycache__", ".pytest_cache", ".ruff_cache", "node_modules", ".ipynb_checkpoints"}
SKIP_NAMES = {"structure.txt", "fix_pipeline.py"}
STALE = __import__("re").compile(r"^ewsv2_S\d+$")              # old extracted segment copies
WHEEL_TARGET = ["--platform", "win_amd64", "--python-version", "3.11", "--implementation", "cp"]
NOT_LOCKED = {"pip", "setuptools", "ewsv2"}
INSTALLERS = [
    ("python-{v}-amd64.exe", "https://www.python.org/ftp/python/{v}/python-{v}-amd64.exe"),
    ("OllamaSetup.exe", "https://ollama.com/download/OllamaSetup.exe"),
]
MANUAL = [
    ("Python 3.11 (64-bit)", "https://www.python.org/downloads/windows/"),
    ("Arduino IDE 2", "https://www.arduino.cc/en/software"),
    ("CH340 USB driver (for the Uno clone)", "https://www.wch-ic.com/downloads/CH341SER_EXE.html"),
    ("Ollama", "https://ollama.com/download/windows"),
]


# ---------------------------------------------------------------------------------------------- helpers

def _size(p: Path) -> int:
    if p.is_file():
        return p.stat().st_size
    total = 0
    for root, dirs, files in os.walk(p):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in files:
            try:
                total += (Path(root) / f).stat().st_size
            except OSError:
                pass
    return total


def _gb(n: float) -> str:
    return f"{n / 1e9:.2f} GB" if n >= 1e8 else f"{n / 1e6:.1f} MB"


def _ignore(dest: Path):
    dest = dest.resolve()

    def ign(directory, names):
        d = Path(directory).resolve()
        out = {n for n in names if n in SKIP_DIRS or n in SKIP_NAMES or STALE.match(n)}
        out |= {n for n in names if (d / n).resolve() == dest}            # never copy the capsule into itself
        return out
    return ign


FAT32_MAX = 4 * 1024 ** 3 - 1


def _filesystem(path: Path) -> str:
    """File system name of the drive holding ``path`` (Windows only; '' elsewhere)."""
    if not sys.platform.startswith("win"):
        return ""
    try:
        import ctypes
        root = str(Path(path).resolve().anchor)
        buf = ctypes.create_unicode_buffer(64)
        ok = ctypes.windll.kernel32.GetVolumeInformationW(ctypes.c_wchar_p(root), None, 0, None, None, None, buf, 64)
        return buf.value if ok else ""
    except Exception:  # noqa: BLE001
        return ""


def _writable(p: Path) -> None:
    """Git stores its objects read-only; Windows then refuses to overwrite or delete them. Clear the flag."""
    try:
        os.chmod(p, stat.S_IWRITE | stat.S_IREAD)
    except OSError:
        pass


def _rm_error(func, path, _exc) -> None:
    _writable(Path(path))
    try:
        func(path)
    except OSError:
        pass


def sync_tree(src: Path, dst: Path, ignore=None, mirror: bool = False, max_file: int | None = None) -> dict:
    """Copy src to dst, skipping files that are already there with the same size and time.

    mirror=True also deletes files in dst that no longer exist in src (so removed files leave the capsule).
    max_file: files larger than this are skipped (FAT32 pendrives cannot hold files of 4 GB or more).
    """
    stats = {"copied": 0, "same": 0, "removed": 0, "too_big": []}
    keep = set()
    for root, dirs, files in os.walk(src):
        r = Path(root)
        if ignore is not None:
            skip = ignore(root, dirs + files)
            dirs[:] = [d for d in dirs if d not in skip]
            files = [f for f in files if f not in skip]
        rel = r.relative_to(src)
        (dst / rel).mkdir(parents=True, exist_ok=True)
        keep.add((dst / rel).resolve())
        for f in files:
            s_, d_ = r / f, dst / rel / f
            keep.add(d_.resolve())
            try:
                st = s_.stat()
            except OSError:
                continue
            if max_file is not None and st.st_size > max_file:
                stats["too_big"].append(str(s_))
                continue
            if d_.exists():
                dt = d_.stat()
                if dt.st_size == st.st_size and abs(dt.st_mtime - st.st_mtime) < 2.1:
                    stats["same"] += 1
                    continue
            if d_.exists():
                _writable(d_)
            try:
                shutil.copy2(s_, d_)
            except PermissionError:
                _writable(d_)
                d_.unlink(missing_ok=True)
                shutil.copy2(s_, d_)
            stats["copied"] += 1
    if mirror and dst.exists():
        for root, dirs, files in os.walk(dst, topdown=False):
            for f in files:
                p = Path(root) / f
                if p.resolve() not in keep:
                    _writable(p)
                    p.unlink()
                    stats["removed"] += 1
            for d in dirs:
                p = Path(root) / d
                if p.resolve() not in keep:
                    shutil.rmtree(p, onerror=_rm_error)
    return stats


def freeze() -> list[str]:
    """Exact versions of the running environment (like pip freeze, without pip/setuptools/wheel)."""
    pins = {}
    for d in metadata.distributions():
        try:
            name = d.metadata["Name"]
        except Exception:  # noqa: BLE001
            continue
        if not name or name.lower().replace("_", "-") in NOT_LOCKED:
            continue
        pins[name.lower().replace("_", "-")] = d.version
    return [f"{n}=={v}" for n, v in sorted(pins.items())]


def freeze_site_packages(site: Path) -> list[str]:
    """Exact versions of another environment, read from its *.dist-info folder names (it is not run)."""
    pins = {}
    for d in site.glob("*.dist-info"):
        stem = d.name[: -len(".dist-info")]
        if "-" not in stem:
            continue
        name, ver = stem.rsplit("-", 1)
        n = name.lower().replace("_", "-")
        if n not in NOT_LOCKED:
            pins[n] = ver
    return [f"{n}=={v}" for n, v in sorted(pins.items())]


def _pip(args: list[str]) -> tuple[int, str]:
    r = subprocess.run([sys.executable, "-m", "pip", *args, "--disable-pip-version-check"], capture_output=True,
                       text=True)
    return r.returncode, (r.stdout + r.stderr)[-2000:]


def wheelhouse(lock: Path, out: Path) -> list[str]:
    """Download one Windows wheel per locked package; build a wheel when only a source archive exists."""
    out.mkdir(parents=True, exist_ok=True)
    reqs = [ln.split("#")[0].strip() for ln in lock.read_text(encoding="utf-8").splitlines()]
    reqs = [r for r in reqs if "==" in r]
    have = {f.name.lower() for f in out.glob("*.whl")}

    def present(r: str) -> bool:
        n, v = r.split("==", 1)
        stem = n.lower().replace("-", "_").replace(".", "_") + "-" + v.lower() + "-"
        return any(h.startswith(stem) for h in have)
    todo = [r for r in reqs if not present(r)]
    print(f"  {len(reqs) - len(todo)} of {len(reqs)} packages already in {out}; downloading {len(todo)} …")
    if not todo:
        return []
    lock = out / "_todo.txt"
    lock.write_text("\n".join(todo) + "\n", encoding="utf-8")
    reqs = todo
    code, msg = _pip(["download", "--no-deps", "--only-binary=:all:", *WHEEL_TARGET, "-d", str(out), "-r", str(lock)])
    failed = []
    if code != 0:                                        # one at a time, to find the ones without a wheel
        for r in reqs:
            c, _ = _pip(["download", "--no-deps", "--only-binary=:all:", *WHEEL_TARGET, "-d", str(out), r])
            if c != 0:
                c2, m2 = _pip(["wheel", "--no-deps", "-w", str(out), r])     # pure-Python source packages
                if c2 != 0:
                    failed.append(r)
                    print(f"  ! no wheel for {r}: {m2.splitlines()[-1] if m2 else ''}")
                else:
                    print(f"  built a wheel for {r} from its source archive")
    _pip(["download", "--only-binary=:all:", *WHEEL_TARGET, "-d", str(out), "pip", "setuptools", "wheel"])
    lock.unlink(missing_ok=True)
    return failed


def _download(url: str, target: Path) -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ewsv2-capsule"})
        with urllib.request.urlopen(req, timeout=60) as r, open(target, "wb") as f:  # noqa: S310
            shutil.copyfileobj(r, f, length=1 << 20)
        return target.stat().st_size > 100_000
    except Exception as exc:  # noqa: BLE001
        print(f"  ! could not download {url}: {exc}")
        target.unlink(missing_ok=True)
        return False


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(dest: Path) -> dict:
    files = {}
    for root, _, names in os.walk(dest):
        for n in names:
            p = Path(root) / n
            rel = p.relative_to(dest).as_posix()
            if rel == "manifest.json":
                continue
            files[rel] = {"bytes": p.stat().st_size, "sha256": _sha256(p)}
    man = {"created": time.strftime("%Y-%m-%d %H:%M"), "ewsv2": __version__, "segment": SEGMENT,
           "files": len(files), "bytes": sum(f["bytes"] for f in files.values()), "entries": files}
    (dest / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    return man


# ---------------------------------------------------------------------------------------------- build

def plan(wheels: bool, v1: bool, ollama: bool, arduino: bool, installers: bool) -> list[tuple[str, str, int | None]]:
    items = [("Project (version 1 + 2, data, results, models; no venvs)", str(REPO_ROOT), _size(REPO_ROOT))]
    if wheels:
        items.append(("Wheelhouse for version 2 (Windows, Python 3.11)", "download", 900_000_000))
        if v1:
            items.append(("Wheelhouse for version 1", "download", 1_100_000_000))
    if ollama:
        src = _ollama_models()
        items.append(("Ollama models folder", str(src), _size(src) if src and src.exists() else 0))
    if arduino:
        src = _arduino15()
        items.append(("Arduino IDE board files (Arduino15)", str(src), _size(src) if src and src.exists() else 0))
    if installers:
        items.append(("Installers (Python 3.11, Ollama)", "download", 1_300_000_000))
    return items


def _ollama_models() -> Path | None:
    p = os.environ.get("OLLAMA_MODELS")
    return Path(p) if p else Path.home() / ".ollama" / "models"


def _arduino15() -> Path | None:
    if sys.platform.startswith("win"):
        base = os.environ.get("LOCALAPPDATA")
        return Path(base) / "Arduino15" if base else None
    return Path.home() / (".arduino15" if sys.platform.startswith("linux") else "Library/Arduino15")


def build(dest: Path | None, wheels: bool = True, v1: bool = True, ollama: bool = False, arduino: bool = False,
          installers: bool = False, dry_run: bool = False) -> int:
    if dest is None:
        print("Give the target folder, for example:  python -m ewsv2 capsule --dest E:\\ewsv2_capsule")
        return 2
    dest = dest.expanduser().resolve()
    items = plan(wheels, v1, ollama, arduino, installers)
    need = sum(s or 0 for _, _, s in items)
    print(f"Time capsule → {dest}")
    for label, src, size in items:
        print(f"  - {label}: about {_gb(size or 0)}" + ("" if src == "download" else f"  ({src})"))
    probe = dest if dest.exists() else next((p for p in dest.parents if p.exists()), dest.anchor)
    free = shutil.disk_usage(probe).free
    print(f"  needs about {_gb(need)}; {_gb(free)} free on the target drive")
    if dry_run:
        print("Dry run: nothing written.")
        return 0
    if free < need * 1.05:
        print("Not enough free space on the target drive. Free some space or leave out the optional parts.")
        return 2
    t0 = time.time()
    dest.mkdir(parents=True, exist_ok=True)

    fs = _filesystem(dest)
    max_file = FAT32_MAX if fs.upper() == "FAT32" else None
    if max_file:
        print("  note: the target drive is FAT32 – files of 4 GB or more cannot be stored on it and are skipped")

    print("\n[1] Copying the project (files removed from the project are removed from the capsule too) …")
    st_ = sync_tree(REPO_ROOT, dest / "project", ignore=_ignore(dest), mirror=True, max_file=max_file)
    print(f"  {st_['copied']} copied, {st_['same']} unchanged, {st_['removed']} removed")
    too_big = list(st_["too_big"])

    print("[2] Writing the lock files …")
    locks = dest / "locks"
    locks.mkdir(exist_ok=True)
    v2 = freeze()
    (locks / "requirements-lock-v2.txt").write_text(
        f"# venv2 on {platform.node()} ({platform.system()} {platform.release()}, Python {platform.python_version()}), "
        f"{time.strftime('%Y-%m-%d')}\n" + "\n".join(v2) + "\n", encoding="utf-8")
    site_v1 = REPO_ROOT / "venv" / "Lib" / "site-packages"
    if site_v1.exists():
        v1pins = freeze_site_packages(site_v1)
        (locks / "requirements-lock-v1.txt").write_text(
            f"# version 1's venv on {platform.node()}, {time.strftime('%Y-%m-%d')}\n" + "\n".join(v1pins) + "\n",
            encoding="utf-8")
    else:
        shutil.copy2(LOCK_DIR / "requirements-lock-v1.txt", locks / "requirements-lock-v1.txt")
    print(f"  venv2: {len(v2)} packages")

    failed = []
    if wheels:
        print("[3] Downloading the wheelhouses (needs internet, several minutes) …")
        failed += wheelhouse(locks / "requirements-lock-v2.txt", dest / "wheelhouse" / "v2")
        if v1:
            failed += wheelhouse(locks / "requirements-lock-v1.txt", dest / "wheelhouse" / "v1")
    if ollama:
        src = _ollama_models()
        if src and src.exists():
            print(f"[4] Copying the Ollama models from {src} …")
            st_ = sync_tree(src, dest / "ollama" / "models", max_file=max_file)
            too_big += st_["too_big"]
            print(f"  {st_['copied']} copied, {st_['same']} unchanged")
        else:
            print("[4] ! Ollama models folder not found – skipped")
    if arduino:
        src = _arduino15()
        if src and src.exists():
            print(f"[5] Copying the Arduino board files from {src} …")
            st_ = sync_tree(src, dest / "arduino" / "Arduino15", ignore=shutil.ignore_patterns("staging"),
                            max_file=max_file)
            print(f"  {st_['copied']} copied, {st_['same']} unchanged")
        else:
            print("[5] ! Arduino15 folder not found – skipped")
    if installers:
        print("[6] Downloading installers …")
        inst = dest / "installers"
        inst.mkdir(exist_ok=True)
        v = platform.python_version() if sys.version_info[:2] == (3, 11) else "3.11.9"
        for name, url in INSTALLERS:
            name, url = name.format(v=v), url.format(v=v)
            if not (inst / name).exists():
                ok = _download(url, inst / name)
                print(f"  {'ok' if ok else 'failed'}: {name}")

    print("[7] Writing RESTORE.md, restore.ps1 and the manifest …")
    (dest / "RESTORE.md").write_text(restore_md(wheels, v1, ollama, arduino, installers, dest), encoding="utf-8")
    (dest / "restore.ps1").write_text(RESTORE_PS1, encoding="utf-8")
    info = {"created": time.strftime("%Y-%m-%d %H:%M"), "computer": platform.node(), "python": platform.python_version(),
            "ewsv2": __version__, "segment": SEGMENT, "wheel_failures": failed,
            "parts": {"wheels": wheels, "v1": v1, "ollama": ollama, "arduino": arduino, "installers": installers}}
    (dest / "capsule_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    man = write_manifest(dest)
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min: {man['files']:,} files, {_gb(man['bytes'])}.")
    if failed:
        print(f"! {len(failed)} package(s) have no offline copy: {', '.join(failed)} (see RESTORE.md).")
    if too_big:
        print(f"! {len(too_big)} file(s) of 4 GB or more were skipped (FAT32 drive): {', '.join(too_big[:3])}. "
              "Format the pendrive as exFAT to store them.")
    print(f"Verify it any time with:  python -m ewsv2 capsule --check \"{dest}\"")
    return 0 if not failed else 1


def check(folder: Path) -> int:
    folder = folder.expanduser().resolve()
    mf = folder / "manifest.json"
    if not mf.exists():
        print(f"No manifest.json in {folder} – is this a capsule folder?")
        return 2
    man = json.loads(mf.read_text(encoding="utf-8"))
    bad, missing = [], []
    print(f"Checking {man['files']:,} files ({_gb(man['bytes'])}) from {man['created']} …")
    for rel, e in man["entries"].items():
        p = folder / rel
        if not p.exists():
            missing.append(rel)
        elif p.stat().st_size != e["bytes"] or _sha256(p) != e["sha256"]:
            bad.append(rel)
    if not bad and not missing:
        print("Capsule intact: every file is present and unchanged.")
        return 0
    for r in (missing[:10]):
        print(f"  missing: {r}")
    for r in (bad[:10]):
        print(f"  changed: {r}")
    print(f"{len(missing)} missing, {len(bad)} changed. Write the capsule again from a working laptop.")
    return 1


# ---------------------------------------------------------------------------------------------- restore texts

def restore_md(wheels: bool, v1: bool, ollama: bool, arduino: bool, installers: bool,
               dest: Path | None = None) -> str:
    manual = "\n".join(f"- {n}: {u}" for n, u in MANUAL)
    cap = str(dest) if dest is not None else "E:\\ewsv2_capsule"
    drive = cap[:2] if len(cap) > 1 and cap[1] == ":" else "the pendrive"
    ard_note = ("- Arduino board files: copy `arduino/Arduino15` to `%LOCALAPPDATA%\\Arduino15` before starting the IDE."
                if arduino else "")
    return f"""# Water Quality Early Warning System – restore without internet

Written {time.strftime('%d %b %Y')} by `python -m ewsv2 capsule` (ewsv2 {__version__}).

## What is in this folder
| Folder | Contents |
|---|---|
| `project/` | the complete repository: version 1, version 2 (`ewsv2/`), data, results, trained models |
| `locks/` | the exact package versions that were working |
| `wheelhouse/` | {"every package for Windows 64-bit, Python 3.11" if wheels else "not included"} |
| `ollama/` | {"the local AI model files" if ollama else "not included – run `ollama pull qwen2.5:3b` when online"} |
| `arduino/` | {"Arduino IDE board files" if arduino else "not included – the Arduino IDE installs them when online"} |
| `installers/` | {"Python 3.11 and Ollama installers" if installers else "not included – see the links below"} |

## Restore in one command
1. Install **Python 3.11 (64-bit)** – from `installers/` if present – and tick *Add python.exe to PATH*.
2. Open PowerShell and run (here the capsule is on {drive}):

```powershell
powershell -ExecutionPolicy Bypass -File {cap}\\restore.ps1
```

It copies the project to `Documents\\Github\\water-quality-early-warning`, creates `venv2` (and version 1's
`venv`) from the wheelhouse without internet, copies the AI model if present, and runs the system check.
To restore somewhere else: `restore.ps1 -Target D:\\projects\\water`.

## Restore by hand
```powershell
robocopy {cap}\\project "$HOME\\Documents\\Github\\water-quality-early-warning" /E
cd "$HOME\\Documents\\Github\\water-quality-early-warning"
py -3.11 -m venv venv2
venv2\\Scripts\\python -m pip install --no-index --no-deps --find-links {cap}\\wheelhouse\\v2 -r {cap}\\locks\\requirements-lock-v2.txt
venv2\\Scripts\\python -m ewsv2 doctor
venv2\\Scripts\\python -m ewsv2 dashboard
```
Version 1 (optional): the same with `venv`, `wheelhouse\\v1` and `requirements-lock-v1.txt`.

## Other software (install when needed)
- Arduino IDE 2 and the CH340 driver – only to upload the sketch to the Uno.
- Ollama – only for AI-written reports; without it the template reports are used.
{ard_note}

Download links:
{manual}

## Check the capsule
```powershell
venv2\\Scripts\\python -m ewsv2 capsule --check {cap}
```
"""


RESTORE_PS1 = r'''# Restore the Water Quality Early Warning System from this capsule (no internet needed).
# Usage:  powershell -ExecutionPolicy Bypass -File E:\ewsv2_capsule\restore.ps1 [-Target <folder>] [-SkipV1]
param(
    [string]$Target = "$HOME\Documents\Github\water-quality-early-warning",
    [switch]$SkipV1
)
$Cap = Split-Path -Parent $MyInvocation.MyCommand.Path
Write-Host "Capsule: $Cap"
Write-Host "Target : $Target"

# 1. Python 3.11
$UsePy = $false
if (Get-Command py -ErrorAction SilentlyContinue) { & py -3.11 -c "import sys" 2>$null; if ($LASTEXITCODE -eq 0) { $UsePy = $true } }
if (-not $UsePy -and -not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "Python 3.11 is not installed. Install it first (see installers\ or RESTORE.md), then run this again." -ForegroundColor Red
    exit 1
}
function New-Venv([string]$Path) {
    if ($UsePy) { & py -3.11 -m venv $Path } else { & python -m venv $Path }
}

# 2. Project files
Write-Host "`n[1/4] Copying the project ..."
robocopy "$Cap\project" "$Target" /E /NFL /NDL /NJH /NJS /NP | Out-Null
if ($LASTEXITCODE -ge 8) { Write-Host "Copy failed (robocopy code $LASTEXITCODE)" -ForegroundColor Red; exit 1 }

# 3. Version 2 environment
Write-Host "[2/4] Creating venv2 from the wheelhouse (offline) ..."
if (-not (Test-Path "$Target\venv2\Scripts\python.exe")) { New-Venv "$Target\venv2" }
& "$Target\venv2\Scripts\python.exe" -m pip install --no-index --no-deps --find-links "$Cap\wheelhouse\v2" -r "$Cap\locks\requirements-lock-v2.txt"
if ($LASTEXITCODE -ne 0) { Write-Host "venv2 install reported a problem (see above)." -ForegroundColor Yellow }

# 4. Version 1 environment (optional)
if (-not $SkipV1 -and (Test-Path "$Cap\wheelhouse\v1")) {
    Write-Host "[3/4] Creating venv (version 1) from the wheelhouse (offline) ..."
    if (-not (Test-Path "$Target\venv\Scripts\python.exe")) { New-Venv "$Target\venv" }
    & "$Target\venv\Scripts\python.exe" -m pip install --no-index --no-deps --find-links "$Cap\wheelhouse\v1" -r "$Cap\locks\requirements-lock-v1.txt"
} else { Write-Host "[3/4] Version 1 environment skipped." }

# 5. AI model (optional)
if (Test-Path "$Cap\ollama\models") {
    Write-Host "[4/4] Copying the Ollama models ..."
    robocopy "$Cap\ollama\models" "$HOME\.ollama\models" /E /NFL /NDL /NJH /NJS /NP | Out-Null
} else { Write-Host "[4/4] No Ollama models in the capsule (template reports will be used)." }

Write-Host "`nSystem check (both versions):" -ForegroundColor Cyan
Set-Location $Target
& "$Target\venv2\Scripts\python.exe" -m ewsv2 doctor --no-board
Write-Host "`nStart both dashboards (version 2 on port 8502, version 1 on port 8501):" -ForegroundColor Green
Write-Host "  cd `"$Target`"; venv2\Scripts\python -m ewsv2 dashboard" -ForegroundColor Green
Write-Host "Version 1 alone:  cd `"$Target\dashboard`"; ..\venv\Scripts\python -m streamlit run app.py --server.port 8501" -ForegroundColor Green
'''
