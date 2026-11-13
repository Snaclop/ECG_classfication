"""Download PTB-XL and the three NSTDB noise records used by this project."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import shutil
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def fetch(url, target):
    request = urllib.request.Request(url, headers={"User-Agent": "ECG-course-project/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response, target.open("wb") as output:
        shutil.copyfileobj(response, output)


def fetch_parallel(url, partial, workers):
    request = urllib.request.Request(url, headers={
        "Range": "bytes=0-0", "User-Agent": "ECG-course-project/1.0",
    })
    with urllib.request.urlopen(request, timeout=60) as response:
        total = int(response.headers["Content-Range"].rsplit("/", 1)[1])
    chunk_size = 8 * 1024 * 1024
    parts = partial.with_suffix(".parts")
    parts.mkdir(exist_ok=True)
    ranges = [(i, start, min(start + chunk_size, total) - 1)
              for i, start in enumerate(range(0, total, chunk_size))]

    def chunk(item):
        i, start, end = item
        path = parts / f"{i:05d}"
        if path.exists():
            return
        temporary = path.with_suffix(".part")
        request = urllib.request.Request(url, headers={
            "Range": f"bytes={start}-{end}", "User-Agent": "ECG-course-project/1.0",
        })
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        temporary.replace(path)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for count, future in enumerate(as_completed([pool.submit(chunk, item) for item in ranges]), 1):
            future.result()
            if count % 5 == 0 or count == len(ranges):
                print(f"Downloaded chunks {count}/{len(ranges)}", flush=True)
    with partial.open("wb") as output:
        for i, _, _ in ranges:
            with (parts / f"{i:05d}").open("rb") as source:
                shutil.copyfileobj(source, output)


def download_ptb(url, version, workers):
    archives = ROOT / "data/archives"
    archives.mkdir(parents=True, exist_ok=True)
    archive = archives / f"ptb-xl-{version}.zip"
    if not archive.exists():
        partial = archive.with_suffix(".zip.part")
        fetch_parallel(url, partial, workers)
        partial.replace(archive)
    destination = ROOT / "data/raw" / f"ptb-xl-{version}"
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive) as zipped:
        # The official archive has one descriptive top-level directory.
        for count, member in enumerate(zipped.infolist(), 1):
            if member.is_dir():
                continue
            relative = member.filename.split("/", 1)[1]
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with zipped.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            if count % 10000 == 0:
                print(f"Extracted {count} members", flush=True)
    print(f"PTB-XL: {destination}", flush=True)


def download_noise(url, version, workers):
    destination = ROOT / "data/raw" / f"nstdb-{version}"
    destination.mkdir(parents=True, exist_ok=False)
    names = [f"{name}.{ext}" for name in ("bw", "ma", "em") for ext in ("hea", "dat")]

    def one(name):
        target = destination / name
        partial = target.with_suffix(target.suffix + ".part")
        fetch(f"{url}/{name}", partial)
        partial.replace(target)
        print(f"Downloaded NSTDB {name}", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, names))


def main():
    ptb_url = "https://physionet.org/content/ptb-xl/get-zip/1.0.3/"
    ptb_version = "1.0.3"
    noise_url = "https://physionet.org/files/nstdb/1.0.0"
    noise_version = "1.0.0"
    workers = 8
    download_ptb(ptb_url, ptb_version, workers)
    download_noise(noise_url, noise_version, workers)


if __name__ == "__main__":
    main()
