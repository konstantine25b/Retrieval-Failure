import json
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT_ROOT / "data" / "mmlu_pro" / "eval_results"
GITHUB_API = "https://api.github.com/repos/TIGER-AI-Lab/MMLU-Pro/contents/eval_results"


def list_eval_zips() -> list[dict]:
    import urllib.request

    with urllib.request.urlopen(GITHUB_API) as response:
        items = json.load(response)
    return [item for item in items if item["name"].startswith("model_outputs_") and item["name"].endswith(".zip")]


def download_eval_results():
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    zips = list_eval_zips()
    print(f"Found {len(zips)} model eval archives")

    for item in zips:
        name = item["name"]
        out_zip = EVAL_DIR / name
        if out_zip.exists():
            print(f"Skipping {name} (already downloaded)")
            continue
        print(f"Downloading {name}...")
        urlretrieve(item["download_url"], out_zip)
        with zipfile.ZipFile(out_zip) as zf:
            zf.extractall(EVAL_DIR)
        print(f"  -> extracted to {EVAL_DIR}")


def main():
    download_eval_results()
    json_files = sorted(EVAL_DIR.glob("model_outputs_*.json"))
    print(f"Done. {len(json_files)} model result files in {EVAL_DIR}")


if __name__ == "__main__":
    main()
