"""
Dataset Format Converter Utility (JSON to JSONL)
==============================================
Converts legacy array-based JSON dataset files into line-delimited JSONL format
for zero-RAM byte-offset streaming (Lazy Loading).

Author: Alireza Ghafouri (Master's Thesis Project - TrajectoryLNAS)
"""

import os
import json
import sys


def convert_json_to_jsonl(json_path: str, jsonl_path: str) -> None:
    """
    Converts a standard JSON list file into a line-delimited JSONL file.
    
    Args:
        json_path: Path to the input .json file containing a list of records.
        jsonl_path: Path to the output .jsonl file.
    """
    if not os.path.exists(json_path):
        print(f"[Error] Input JSON file not found: {json_path}")
        return

    print(f"[Converter] Reading JSON dataset from: {json_path} ...")
    try:
        with open(json_path, "r", encoding="utf-8") as f_in:
            data = json.load(f_in)
    except Exception as e:
        print(f"[Error] Failed to parse JSON file {json_path}: {e}")
        return

    if not isinstance(data, list):
        print(f"[Warning] The input file {json_path} does not contain a JSON list. Found type: {type(data)}")
        return

    print(f"[Converter] Converting {len(data):,} records to JSONL streaming format at: {jsonl_path} ...")
    os.makedirs(os.path.dirname(os.path.abspath(jsonl_path)), exist_ok=True)

    with open(jsonl_path, "w", encoding="utf-8") as f_out:
        for idx, item in enumerate(data):
            f_out.write(json.dumps(item) + "\n")
            if (idx + 1) % 5000 == 0 or (idx + 1) == len(data):
                print(f"  -> Converted {idx + 1:,} / {len(data):,} records...", flush=True)

    file_size_mb = os.path.getsize(jsonl_path) / (1024 * 1024)
    print(f"[Success] Converted successfully! Output saved to: {jsonl_path} ({file_size_mb:.2f} MB)\n")


if __name__ == "__main__":
    # Default paths for TrajectoryLNAS nuScenes dataset splits
    default_base = os.path.join("data", "nuscenes", "v1.0-trainval")
    
    train_json = os.path.join(default_base, "train_sequences.json")
    train_jsonl = os.path.join(default_base, "train_sequences.jsonl")
    
    val_json = os.path.join(default_base, "val_sequences.json")
    val_jsonl = os.path.join(default_base, "val_sequences.jsonl")

    print("=" * 80)
    print(" TrajectoryLNAS Dataset Format Converter (JSON -> JSONL)")
    print("=" * 80)

    if os.path.exists(train_json):
        convert_json_to_jsonl(train_json, train_jsonl)
    else:
        print(f"[Notice] Train JSON file not found at {train_json}")

    if os.path.exists(val_json):
        convert_json_to_jsonl(val_json, val_jsonl)
    else:
        print(f"[Notice] Val JSON file not found at {val_json}")

    print("=" * 80)
    print(" Conversion utility finished. You can now use Streaming Line-by-Line loading!")
    print("=" * 80)
