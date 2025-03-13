import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Dict, Tuple


class DirectoryCache:
    def __init__(self, cache_file: str = ".analysis_cache"):
        self.cache_file = cache_file
        self.cache: Dict[str, Tuple[Dict[str, float], Dict[str, float]]] = {}
        self.load_cache()

    def load_cache(self):
        try:
            with open(self.cache_file, "rb") as f:
                self.cache = pickle.load(f)
        except (FileNotFoundError, pickle.UnpicklingError):
            self.cache = {}

    def save_cache(self):
        with open(self.cache_file, "wb") as f:
            pickle.dump(self.cache, f)

    def get_directory_hash(self, directory: str) -> str:
        """Create a hash of file modification times in the directory."""
        json_files = sorted(Path(directory).glob("*.json"))
        hash_input = []
        for file in json_files:
            hash_input.append(f"{file}:{os.path.getmtime(file)}")
        return hashlib.md5("".join(hash_input).encode()).hexdigest()

    def get_cached_results(
        self, directory: str
    ) -> Tuple[int, int, float, float, float]:
        dir_hash = self.get_directory_hash(directory)
        if directory in self.cache:
            cached_hash, results = self.cache[directory]
            if cached_hash == dir_hash:
                return results
        return None

    def cache_results(
        self, directory: str, results: Tuple[int, int, float, float, float]
    ):
        dir_hash = self.get_directory_hash(directory)
        self.cache[directory] = (dir_hash, results)
        self.save_cache()


def analyze_json_files(
    directory: str, cache: DirectoryCache
) -> Tuple[int, int, float, float, float]:
    # Check cache first
    cached_results = cache.get_cached_results(directory)
    if cached_results is not None:
        return cached_results

    # If not in cache, perform analysis
    json_files = list(Path(directory).glob("*.json"))
    count = 0
    failed = 0
    total_success = 0
    total_spl = 0
    total_soft_spl = 0

    for file in json_files:
        try:
            with open(file) as f:
                data = json.load(f)
                if data and "success" in data and "spl" in data:
                    count += 1
                    total_success += float(data["success"])
                    total_spl += float(data["spl"])
                    total_soft_spl += float(data["soft_spl"])
                else:
                    failed += 1
        except (json.JSONDecodeError, ValueError, KeyError):
            failed += 1
            continue

    results = (
        count,
        failed,
        total_success / count if count > 0 else 0,
        total_spl / count if count > 0 else 0,
        total_soft_spl / count if count > 0 else 0,
    )

    # Cache the results
    cache.cache_results(directory, results)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Count non-empty JSON files and calculate statistics"
    )
    parser.add_argument("directories", nargs="+", help="Directory paths to analyze")
    parser.add_argument(
        "--cache-file",
        default=".analysis_cache",
        help="Path to cache file (default: .analysis_cache)",
    )

    args = parser.parse_args()
    cache = DirectoryCache(args.cache_file)

    grand_total_valid = 0
    grand_total_failed = 0

    for dir_path in sorted(args.directories):
        if os.path.exists(dir_path):
            subdirs = [subdir for subdir in os.scandir(dir_path) if subdir.is_dir()]
            for subdir in sorted(subdirs, key=lambda x: x.name):
                count, failed, avg_success, avg_spl, avg_soft_spl = analyze_json_files(
                    subdir.path, cache
                )
                grand_total_valid += count
                grand_total_failed += failed

    print(f"Total valid JSONs across all subdirectories: {grand_total_valid}")
    print(f"Total failed JSONs across all subdirectories: {grand_total_failed}")
    print("\nDetailed breakdown by directory:")

    for dir_path in sorted(args.directories):
        if os.path.exists(dir_path):
            print(f"\nDirectory: {dir_path}")
            print("-" * (len(dir_path) + 11))
            subdirs = [subdir for subdir in os.scandir(dir_path) if subdir.is_dir()]
            for subdir in sorted(subdirs, key=lambda x: x.name):
                count, failed, avg_success, avg_spl, avg_soft_spl = analyze_json_files(
                    subdir.path, cache
                )
                if count > 0 or failed > 0:
                    print(
                        f"  {os.path.basename(subdir.path)}: {count} valid"
                        f", {failed} blank (avg success: {avg_success:.2f},"
                        f" avg SPL: {avg_spl:.2f}, avg soft SPL: {avg_soft_spl:.2f})"
                    )


if __name__ == "__main__":
    main()
