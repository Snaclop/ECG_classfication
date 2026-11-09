"""Download official versioned archives with HTTP resume; extract safely."""
import argparse
from pathlib import Path
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "ptb-xl": ("1.0.3", "https://physionet.org/content/ptb-xl/get-zip/1.0.3/"),
    "nstdb": ("1.0.0", "https://physionet.org/content/nstdb/get-zip/1.0.0/"),
}


def fetch(url, partial):
    for attempt in range(6):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "ECG-course-project/1.0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response:
                if response.status == 206:
                    if not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                        raise ValueError("Unexpected Content-Range")
                    mode = "ab"
                else:
                    mode, offset = "wb", 0
                total = offset + int(response.headers.get("Content-Length", 0))
                last_report = time.monotonic()
                with partial.open(mode) as output:
                    while block := response.read(64 * 1024):
                        output.write(block)
                        offset += len(block)
                        if time.monotonic() - last_report > 15:
                            print(f"Downloaded {offset / 1e6:.1f}/{total / 1e6:.1f} MB", flush=True)
                            last_report = time.monotonic()
                if total and offset != total:
                    raise IOError(f"Incomplete response: {offset}/{total}")
                return
        except (OSError, ValueError) as exc:
            print(f"Attempt {attempt + 1} failed: {exc}", flush=True)
            if attempt == 5:
                raise
            time.sleep(3)


def fetch_parallel(url, partial, workers):
    """Persist independent HTTP ranges; resume completed chunks after interruption."""
    for attempt in range(6):
        try:
            # Some official endpoints reject HEAD but support ranged GET.
            request = urllib.request.Request(url, headers={"Range": "bytes=0-0", "User-Agent": "ECG-course-project/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                if response.status != 206:
                    raise ValueError("Server does not support parallel range downloads; use --workers 1")
                total = int(response.headers["Content-Range"].rsplit("/", 1)[1])
            break
        except OSError:
            if attempt == 5:
                raise
            time.sleep(3)
    chunk_size = 8 * 1024 * 1024
    parts = partial.with_suffix(".parts")
    parts.mkdir(exist_ok=True)
    ranges = [(i, start, min(start + chunk_size, total) - 1) for i, start in enumerate(range(0, total, chunk_size))]

    def chunk(item):
        i, start, end = item
        path = parts / f"{i:05d}"
        size = end - start + 1
        if path.exists() and path.stat().st_size == size:
            return path
        temp = path.with_suffix(".part")
        for attempt in range(6):
            offset = temp.stat().st_size if temp.exists() else 0
            if offset == size:
                temp.replace(path)
                return path
            try:
                headers = {"Range": f"bytes={start + offset}-{end}", "User-Agent": "ECG-course-project/1.0"}
                with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response:
                    if response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {start + offset}-{end}/"):
                        raise ValueError("Server did not honor requested range")
                    with temp.open("ab") as output:
                        shutil.copyfileobj(response, output, length=1024 * 1024)
                if temp.stat().st_size != size:
                    raise IOError("Incomplete chunk")
                temp.replace(path)
                return path
            except (OSError, ValueError):
                if attempt == 5:
                    raise
                time.sleep(3)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for n, future in enumerate(as_completed([pool.submit(chunk, r) for r in ranges]), 1):
            future.result()
            if n % 5 == 0 or n == len(ranges):
                print(f"Downloaded chunks {n}/{len(ranges)}", flush=True)
    with partial.open("wb") as output:
        for i, _, _ in ranges:
            with (parts / f"{i:05d}").open("rb") as source:
                shutil.copyfileobj(source, output)
    assert partial.stat().st_size == total


def download(name, workers=8):
    version, url = SOURCES[name]
    archives = ROOT / "data" / "archives"
    archives.mkdir(parents=True, exist_ok=True)
    archive = archives / f"{name}-{version}.zip"
    if not archive.exists():
        partial = archive.with_suffix(".zip.part")
        print(f"Downloading {name} {version}: {url}", flush=True)
        if workers > 1:
            fetch_parallel(url, partial, workers)
        else:
            fetch(url, partial)
        # Test ZIP integrity before publishing the completed archive.
        with zipfile.ZipFile(partial) as z:
            bad = z.testzip()
            if bad:
                raise ValueError(f"Corrupt ZIP member: {bad}")
        partial.replace(archive)
    destination = ROOT / "data" / "raw" / f"{name}-{version}"
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        members = [m for m in z.infolist() if not m.is_dir()]
        # PhysioNet ZIPs use long descriptive top-level directory names.
        top = members[0].filename.split("/")[0] + "/" if members else ""
        prefix = top if top and all(m.filename.startswith(top) for m in members) else ""
        for i, member in enumerate(members, 1):
            relative = member.filename.removeprefix(prefix)
            target = (destination / relative).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError(f"Unsafe ZIP path: {member.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            # Final files are only replaced after a whole member was written.
            if not target.exists() or target.stat().st_size != member.file_size:
                partial = target.with_name(target.name + ".extracting")
                with z.open(member) as source, partial.open("wb") as output:
                    while block := source.read(1024 * 1024):
                        output.write(block)
                partial.replace(target)
            if i % 10000 == 0:
                print(f"Extracted {i}/{len(members)} members", flush=True)
    print(f"Ready for checksum validation: {destination}", flush=True)


def download_noise():
    """Fetch the three raw noise records used by this project, plus official hashes."""
    destination = ROOT / "data/raw/nstdb-1.0.0"
    destination.mkdir(parents=True, exist_ok=True)
    names = [f"{name}.{ext}" for name in ("bw", "ma", "em") for ext in ("hea", "dat")]
    names.append("SHA256SUMS.txt")
    def one(name):
        target = destination / name
        if not target.exists():
            partial = target.with_name(name + ".part")
            fetch(f"https://physionet.org/files/nstdb/1.0.0/{name}", partial)
            partial.replace(target)
        print(f"Downloaded NSTDB {name}", flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(one, names))
    print(f"Raw noise records ready for checksum validation: {destination}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*", default=list(SOURCES))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--direct", action="store_true", help="Use direct HTTPS instead of system proxy for this process")
    parser.add_argument("--full-nstdb", action="store_true", help="Download the complete NSTDB archive instead of only the three raw noise records")
    args = parser.parse_args()
    if args.direct:
        urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({})))
    if not 1 <= args.workers <= 64:
        parser.error("workers must be between 1 and 64")
    for name in args.datasets:
        if name not in SOURCES:
            parser.error(f"Unknown dataset: {name}")
        if name == "nstdb" and not args.full_nstdb:
            download_noise()
        else:
            download(name, args.workers)
