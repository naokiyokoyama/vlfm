import argparse
import json
import glob
import os


def get_episode_ids_by_failure(directory, failure_cause):
    # Get all json files in the directory
    json_files = glob.glob(os.path.join(directory, '*.json'))

    # List to store matching episode IDs
    matching_episodes = []

    # Process each JSON file
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Check if failure_cause matches and append episode_id
            if data.get('failure_cause') == failure_cause:
                matching_episodes.append(data['episode_id'])
        except json.JSONDecodeError:
            print(f"Warning: Couldn't parse {json_file}")
        except KeyError:
            print(f"Warning: Missing required fields in {json_file}")

    # Sort episode IDs numerically
    matching_episodes.sort()

    # Return comma-separated string of episode IDs
    return ','.join(str(id) for id in matching_episodes)


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Extract episode IDs by failure cause from JSON files')
    parser.add_argument('directory', help='Path to directory containing JSON files')
    parser.add_argument('failure_cause', help='Failure cause to search for')

    # Parse arguments
    args = parser.parse_args()

    # Get and print results
    result = get_episode_ids_by_failure(args.directory, args.failure_cause)
    print(result)


if __name__ == '__main__':
    main()
