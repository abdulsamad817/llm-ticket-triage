import argparse
import json
import random
from pathlib import Path

import yaml

from src.data.schema import TicketLabel


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_raw_records(raw_path: str) -> list[dict]:
    records = []
    path = Path(raw_path)
    if not path.exists():
        raise FileNotFoundError(
            f"No raw data found at {raw_path}. Either add your own labeled JSONL "
            f"file there, or run generate_synthetic_data.py to bootstrap a starter set."
        )
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def clean_record(record: dict) -> dict | None:
    """Basic hygiene: strip whitespace, drop empty/too-short messages, validate required keys."""
    required_keys = {"message", "intent", "sentiment", "urgency", "category", "summary"}
    if not required_keys.issubset(record.keys()):
        return None
    message = record["message"].strip()
    if len(message) < 5:
        return None
    record["message"] = message
    return record


def split_records(records: list[dict], config: dict) -> dict[str, list[dict]]:
    random.seed(config["data"]["seed"])
    shuffled = records.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * config["data"]["train_split"])
    n_val = int(n * config["data"]["val_split"])

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }


def write_jsonl(records: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main(config_path: str = "configs/train_config.yaml") -> None:
    config = load_config(config_path)
    raw_records = load_raw_records(config["data"]["raw_path"])

    cleaned = [clean_record(r) for r in raw_records]
    cleaned = [r for r in cleaned if r is not None]
    dropped = len(raw_records) - len(cleaned)
    print(f"Loaded {len(raw_records)} records, dropped {dropped} invalid, kept {len(cleaned)}.")

    splits = split_records(cleaned, config)
    processed_dir = Path(config["data"]["processed_dir"])
    for split_name, split_records_list in splits.items():
        out_path = processed_dir / f"{split_name}.jsonl"
        write_jsonl(split_records_list, out_path)
        print(f"  {split_name}: {len(split_records_list)} records -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_config.yaml")
    args = parser.parse_args()
    main(args.config)