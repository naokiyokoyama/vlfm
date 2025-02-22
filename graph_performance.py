import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def parse_args() -> Tuple[List[str], List[str]]:
    parser = argparse.ArgumentParser(description="Analyze training runs")
    parser.add_argument(
        "keys", nargs="+", type=str, help="JSON keys to extract float values"
    )
    parser.add_argument(
        "--dirs", nargs="+", required=True, help="Paths to training directories"
    )
    args = parser.parse_args()
    return args.keys, args.dirs


def extract_batch_size(dir_name: str) -> int:
    """Extract batch size from directory name."""
    parts = dir_name.split("_")
    try:
        return int(parts[-2])
    except (IndexError, ValueError):
        return 1  # Default to 1 if batch size cannot be extracted


def process_directory(
    directory: str, keys: List[str]
) -> Dict[str, Dict[int, Tuple[float, int]]]:
    """Process a training directory and return timestep averages and file counts for each key."""
    results = {key: {} for key in keys}
    dir_path = Path(directory)

    batch_size = extract_batch_size(str(dir_path.parent.absolute()))

    for subdir in dir_path.iterdir():
        if not subdir.is_dir() or not subdir.name.isdigit():
            continue

        timestep = int(subdir.name) * batch_size

        # Initialize value collection for each key
        values = {key: [] for key in keys}
        file_counts = {key: 0 for key in keys}

        for json_file in subdir.glob("*.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                    for key in keys:
                        if key in data:
                            values[key].append(float(data[key]))
                            file_counts[key] += 1
            except (json.JSONDecodeError, ValueError, KeyError):
                continue

        # Calculate averages for each key
        for key in keys:
            if values[key]:
                avg_value = sum(values[key]) / len(values[key])
                results[key][timestep] = (avg_value, file_counts[key])

    return results


def plot_results(
    all_results: Dict[str, Dict[str, Dict[int, Tuple[float, int]]]],
    keys: List[str],
    output_path: str = "training_analysis.png",
):
    """Create and save a line plot of the results with different line styles for each key."""
    plt.figure(figsize=(12, 6))

    # Define line styles for different keys
    line_styles = ["-", "--", ":", "-."]

    # Get default color cycle
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]

    for dir_idx, (dir_name, dir_results) in enumerate(all_results.items()):
        # Cycle through colors for each directory
        base_color = colors[dir_idx % len(colors)]

        for key_idx, key in enumerate(keys):
            if key in dir_results:
                results = dir_results[key]
                timesteps = sorted(results.keys())
                values = [results[t][0] for t in timesteps]

                line_style = line_styles[key_idx % len(line_styles)]
                label = f"{Path(dir_name).parent.name} - {key}"

                plt.plot(
                    timesteps,
                    values,
                    label=label,
                    color=base_color,
                    linestyle=line_style,
                    marker=".",
                )

                # Print file counts for each timestep
                print(f"\nDirectory: {dir_name}, Key: {key}")
                for t in timesteps:
                    print(f"Timestep {t}: {results[t][1]} files processed")

    plt.xlabel("Timestep")
    plt.ylabel("Average Value")
    plt.title("Training Analysis")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def main():
    keys, directories = parse_args()
    all_results = {}

    for directory in directories:
        results = process_directory(directory, keys)
        if any(results.values()):
            all_results[directory] = results

    if all_results:
        plot_results(all_results, keys)
    else:
        print("No valid data found in the provided directories")


if __name__ == "__main__":
    main()
