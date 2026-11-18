"""Download GloVe (Stanford) en pak de gevraagde dimensies uit naar data/.

Gebruik (vanuit de repo-root):
    uv run python scripts/download_glove.py                 # GloVe Twitter, 50d en 100d
    uv run python scripts/download_glove.py twitter 25 200  # GloVe Twitter, andere dimensies
    uv run python scripts/download_glove.py wiki            # GloVe 6B (Wikipedia + Gigaword), 50d

De zip-bestanden bevatten meerdere .txt-bestanden (één per dimensie). part2.clslib.PretrainedEmbeddings
kan alleen een zip met één bestand lezen, dus we pakken de gewenste dimensies los uit als .txt.
data/ staat in .gitignore.
"""

import shutil
import sys
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

# Officiële Stanford-bestanden, gespiegeld door stanfordnlp op Hugging Face (nlp.stanford.edu is vaak traag/onbereikbaar)
MIRROR = "https://huggingface.co/stanfordnlp/glove/resolve/main"
MODELS = {
    "twitter": {"zip": "glove.twitter.27B.zip", "txt": "glove.twitter.27B.{dims}d.txt",
                "dims": (25, 50, 100, 200), "default": (50, 100)},
    "wiki": {"zip": "glove.6B.zip", "txt": "glove.6B.{dims}d.txt",
             "dims": (50, 100, 200, 300), "default": (50,)},
}
DEFAULT_MODEL = "twitter"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHUNK_BYTES = 8 * 1024 * 1024


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


def extract(zip_path: Path, member: str) -> None:
    target = DATA_DIR / member
    if target.is_file():
        print(f"{target.name} staat er al")
        return
    with ZipFile(zip_path) as archive, archive.open(member) as src, open(target, "wb") as dst:
        shutil.copyfileobj(src, dst, CHUNK_BYTES)
    print(f"{target.name} uitgepakt")


def parse_args(args: list[str]) -> tuple[str, tuple[int, ...]]:
    model = args[0] if args and args[0] in MODELS else DEFAULT_MODEL
    dim_args = args[1:] if args and args[0] in MODELS else args
    spec = MODELS[model]
    dims = tuple(int(a) for a in dim_args) or spec["default"]
    unknown = [d for d in dims if d not in spec["dims"]]
    if unknown:
        raise SystemExit(f"Onbekende dimensie(s) {unknown} voor {model}, kies uit {spec['dims']}")
    return model, dims


def main() -> None:
    model, dims = parse_args(sys.argv[1:])
    spec = MODELS[model]
    DATA_DIR.mkdir(exist_ok=True)
    zip_path = DATA_DIR / spec["zip"]
    download(f"{MIRROR}/{spec['zip']}", zip_path)
    for d in dims:
        extract(zip_path, spec["txt"].format(dims=d))


if __name__ == "__main__":
    main()
