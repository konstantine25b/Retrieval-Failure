import json
from pathlib import Path
from urllib.request import urlretrieve

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "misprompt"
BASE_URL = "https://raw.githubusercontent.com/Jiayi-Zeng/mis-prompt/master/data"

FILES = ["data.json", "train.json", "dev.json", "eval.json"]


def download_misprompt():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        url = f"{BASE_URL}/{name}"
        out_path = DATA_DIR / name
        print(f"Downloading {name}...")
        urlretrieve(url, out_path)
        with out_path.open(encoding="utf-8") as f:
            rows = json.load(f)
        print(f"  -> {out_path} ({len(rows)} rows)")


def main():
    download_misprompt()
    print("Done.")


if __name__ == "__main__":
    main()
