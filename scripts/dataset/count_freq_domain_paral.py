# BEFORE IMPORTING ANY HF RELATED MODULES
import os
os.environ['HF_HOME'] = f"{os.environ['PROJECTS']}/.cache/huggingface"

import argparse
import json
import datasets
import logging
import multiprocessing
from functools import partial
from pathlib import Path


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


domain_counts = {}
freq_counts = {}
dataset_counts = {}


def process_dataset(subset_name, subset_info):
    local_domain_counts = {}
    local_freq_counts = {}
    local_dataset_counts = 0

    logging.info(
        f"Processing subset: {subset_info['hf_dataset_name']} - {subset_info['hf_subset_name']}"
    )

    ds = datasets.load_dataset(
        subset_info["hf_dataset_name"],
        subset_info["hf_subset_name"],
        keep_in_memory=False,
        split="train"
    )

    for s in ds:
        ds_length = len(s[subset_info["target"][0]["name"]])
        local_dataset_counts += ds_length

        # frequency
        local_freq_counts[subset_info["freq"]] = (
            local_freq_counts.get(subset_info["freq"], 0) + ds_length
        )

        # domain
        for t in subset_info["target"]:
            local_domain_counts[t["domain"]] = (
                local_domain_counts.get(t["domain"], 0) + ds_length
            )

    return {
        "subset_name": subset_name,
        "domain_counts": local_domain_counts,
        "freq_counts": local_freq_counts,
        "dataset_counts": local_dataset_counts,
    }


def update_global_counts(result):
    global domain_counts, freq_counts, dataset_counts

    for domain, count in result["domain_counts"].items():
        domain_counts[domain] = domain_counts.get(domain, 0) + count

    for freq, count in result["freq_counts"].items():
        freq_counts[freq] = freq_counts.get(freq, 0) + count

    dataset_counts[result["subset_name"]] = result["dataset_counts"]


def main(json_path: str):
    global domain_counts, freq_counts, dataset_counts

    # load subsets
    with open(json_path, "r") as f:
        datasets_metadata = json.load(f)

    # Output directory = output/<json_filename_without_ext>/
    json_stem = Path(json_path).stem
    output_dir = Path("scripts/dataset/output") / json_stem
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Using output directory: {output_dir}")

    # Parallel processing
    with multiprocessing.Pool() as pool:
        args = [(name, info) for name, info in datasets_metadata.items()]
        for result in pool.starmap(process_dataset, args):
            update_global_counts(result)

    # Save files
    freq_file = output_dir / "tsmixup_condition_balancing_freq_counts.json"
    domain_file = output_dir / "tsmixup_condition_balancing_domain_counts.json"
    dataset_file = output_dir / "tsmixup_condition_balancing_dataset_counts.json"

    with freq_file.open("w") as f:
        json.dump(freq_counts, f, indent=4)
    with domain_file.open("w") as f:
        json.dump(domain_counts, f, indent=4)
    with dataset_file.open("w") as f:
        json.dump(dataset_counts, f, indent=4)

    # Logging
    logging.info("--- Frequency Counts ---")
    for freq, count in freq_counts.items():
        logging.info(f"Frequency: {freq}, Count: {count}")

    logging.info("\n--- Domain Counts ---")
    for domain, count in domain_counts.items():
        logging.info(f"Domain: {domain}, Count: {count}")

    logging.info("\n--- Dataset Counts ---")
    for dataset, count in dataset_counts.items():
        logging.info(f"Dataset: {dataset}, Count: {count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-json",
        type=str,
        required=True,
        help="Path to the subset JSON definition file."
    )
    args = parser.parse_args()
    main(args.input_json)