"""Download GloVe Twitter (Stanford) en pak de gevraagde dimensies uit naar data/.

Gebruik (vanuit de repo-root):
    uv run python scripts/download_glove.py            # 50d en 100d
    uv run python scripts/download_glove.py 25 200     # andere dimensies

Het zip-bestand bevat vier .txt-bestanden (25d, 50d, 100d, 200d). part2.clslib.PretrainedEmbeddings
kan alleen een zip met één bestand lezen, dus we pakken de gewenste dimensies los uit als .txt.
data/ staat in .gitignore.
"""

import shutil
import sys
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

# Officiële Stanford-bestanden, gespiegeld door stanfordnlp op Hugging Face (nlp.stanford.edu is vaak traag/onbereikbaar)
GLOVE_TWITTER_URL = "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.twitter.27B.zip"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ZIP_PATH = DATA_DIR / "glove.twitter.27B.zip"
AVAILABLE_DIMS = (25, 50, 100, 200)
DEFAULT_DIMS = (50, 100)
CHUNK_BYTES = 8 * 1024 * 1024


def txt_path(dims: int) -> Path:
    return DATA_DIR / f"glove.twitter.27B.{dims}d.txt"


def download(url: str, destination: Path) -> None:
    if destination.is_file():
        print(f"{destination.name} staat er al, download overgeslagen")
        return
    partial = destination.with_suffix(".part")
    with urlopen(url) as response, open(partial, "wb") as out:
        total = int(response.headers.get("Content-Length", 0))
        done = 0
        while chunk := response.read(CHUNK_BYTES):
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r{done / 1e6:,.0f} / {total / 1e6:,.0f} MB", end="", flush=True)
    partial.rename(destination)  # pas hernoemen als de download compleet is
    print(f"\n{destination.name} gedownload")


def extract(zip_path: Path, dims: int) -> None:
    target = txt_path(dims)
    if target.is_file():
        print(f"{target.name} staat er al")
        return
    with ZipFile(zip_path) as archive, archive.open(target.name) as src, open(target, "wb") as dst:
        shutil.copyfileobj(src, dst, CHUNK_BYTES)
    print(f"{target.name} uitgepakt")


def parse_dims(args: list[str]) -> tuple[int, ...]:
    dims = tuple(int(a) for a in args) or DEFAULT_DIMS
    unknown = [d for d in dims if d not in AVAILABLE_DIMS]
    if unknown:
        raise SystemExit(f"Onbekende dimensie(s) {unknown}, kies uit {AVAILABLE_DIMS}")
    return dims


def main() -> None:
    dims = parse_dims(sys.argv[1:])
    DATA_DIR.mkdir(exist_ok=True)
    download(GLOVE_TWITTER_URL, ZIP_PATH)
    for d in dims:
        extract(ZIP_PATH, d)


if __name__ == "__main__":
    main()
