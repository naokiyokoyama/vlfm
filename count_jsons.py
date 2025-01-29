import argparse
import json
import os
from pathlib import Path


def analyze_json_files(directory):
    json_files = list(Path(directory).glob("*.json"))
    count = 0
    failed = 0
    total_success = 0
    total_spl = 0

    for file in json_files:
        try:
            with open(file) as f:
                data = json.load(f)
                if data and "success" in data and "spl" in data:
                    count += 1
                    total_success += float(data["success"])
                    total_spl += float(data["spl"])
                else:
                    failed += 1
        except (json.JSONDecodeError, ValueError, KeyError):
            failed += 1
            continue

    if count > 0:
        return count, failed, total_success / count, total_spl / count
    return 0, failed, 0, 0


def main():
    parser = argparse.ArgumentParser(
        description="Count non-empty JSON files and calculate statistics"
    )
    parser.add_argument("directories", nargs="+", help="Directory paths to analyze")

    args = parser.parse_args()

    for dir_path in sorted(args.directories):
        if os.path.exists(dir_path):
            print(f"\nDirectory: {dir_path}")
            print("-" * (len(dir_path) + 11))
            subdirs = [subdir for subdir in os.scandir(dir_path) if subdir.is_dir()]
            for subdir in sorted(subdirs, key=lambda x: x.name):
                count, failed, avg_success, avg_spl = analyze_json_files(subdir.path)
                if count > 0 or failed > 0:
                    print(
                        f"  {os.path.basename(subdir.path)}: {count} valid"
                        f", {failed} blank (avg success: {avg_success:.2f},"
                        f" avg SPL: {avg_spl:.2f})"
                    )


if __name__ == "__main__":
    main()
