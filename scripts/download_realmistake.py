import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

from datasets import load_dataset
from datasets.exceptions import DatasetNotFoundError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "realmistake"
ZIP_URL = "https://raw.githubusercontent.com/psunlpgroup/ReaLMistake/main/data.zip"
ZIP_PASSWORD = b"open-realmistake"

CONFIGS = [
    "math_word_problem_generation",
    "finegrained_fact_verification",
    "answerability_classification",
]
SPLITS = ["gpt4", "llama2"]


def download_from_huggingface():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for config in CONFIGS:
        out_dir = DATA_DIR / config
        out_dir.mkdir(parents=True, exist_ok=True)

        for split in SPLITS:
            print(f"Downloading {config} / {split}...")
            ds = load_dataset("ryokamoi/realmistake", name=config, split=split)
            out_path = out_dir / f"{split}.jsonl"
            with out_path.open("w", encoding="utf-8") as f:
                for row in ds:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"  -> {out_path} ({len(ds)} rows)")


def download_from_zip():
    print("Downloading official data.zip from GitHub...")
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "data.zip"
        urlretrieve(ZIP_URL, zip_path)

        extract_dir = Path(tmp) / "extracted"
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir, pwd=ZIP_PASSWORD)

        source_dir = extract_dir / "data"
        if DATA_DIR.exists():
            shutil.rmtree(DATA_DIR)
        shutil.copytree(source_dir, DATA_DIR)

    print(f"  -> {DATA_DIR}")


def main():
    try:
        download_from_huggingface()
    except DatasetNotFoundError:
        print("Hugging Face access unavailable. Using official zip fallback.")
        download_from_zip()
    print("Done.")


if __name__ == "__main__":
    main()
