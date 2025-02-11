import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def parse_args() -> Tuple[str, List[str]]:
    parser = argparse.ArgumentParser(description="Analyze training runs")
    parser.add_argument("key", type=str, help="JSON key to extract float values")
    parser.add_argument("dirs", nargs="+", help="Paths to training directories")
    args = parser.parse_args()
    return args.key, args.dirs


def process_directory(directory: str, key: str) -> Dict[int, Tuple[float, int]]:
    """Process a training directory and return timestep averages and file counts."""
    results = {}
    dir_path = Path(directory)

    for subdir in dir_path.iterdir():
        if not subdir.is_dir() or not subdir.name.isdigit():
            continue

        timestep = int(subdir.name)
        values = []
        file_count = 0

        for json_file in subdir.glob("*.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                    if key in data:
                        values.append(float(data[key]))
                        file_count += 1
            except (json.JSONDecodeError, ValueError, KeyError):
                continue

        if values:
            avg_value = sum(values) / len(values)
            results[timestep] = (avg_value, file_count)

    return results


def plot_results(
    all_results: Dict[str, Dict[int, Tuple[float, int]]],
    key: str,
    output_path: str = "training_analysis.png",
):
    """Create and save a line plot of the results."""
    plt.figure(figsize=(12, 6))

    for dir_name, results in all_results.items():
        timesteps = sorted(results.keys())
        values = [results[t][0] for t in timesteps]
        plt.plot(timesteps, values, label=Path(dir_name).name, marker=".")

        # Print file counts for each timestep
        print(f"\nDirectory: {dir_name}")
        for t in timesteps:
            print(f"Timestep {t}: {results[t][1]} files processed")

    plt.xlabel("Timestep")
    plt.ylabel("Average Value")
    plt.title(f"Training Analysis ({key})")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()


def main():
    key, directories = parse_args()
    all_results = {}

    for directory in directories:
        results = process_directory(directory, key)
        if results:
            all_results[directory] = results

    if all_results:
        plot_results(all_results, key)
    else:
        print("No valid data found in the provided directories")


if __name__ == "__main__":
    main()
