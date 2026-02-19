# BEFORE IMPORTING ANY HF RELATED MODULES
import os
os.environ['HF_HOME'] = f"{os.environ['PROJECTS']}/.cache/huggingface"
import json
import datasets
import logging
import argparse

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input-json",
    type=str,
    required=True,
    help="Path to the subset JSON definition file."
)
args = parser.parse_args()

datasets_metadata = json.load(open(args.input_json, "r"))

domain_counts = {} # number of time points per domain
freq_counts = {}   # number of time points per frequency

for subset in datasets_metadata.values():
    logging.info(f"Processing subset: {subset['hf_dataset_name']} - {subset['hf_subset_name']}")
    ds = datasets.load_dataset(subset['hf_dataset_name'], subset['hf_subset_name'], keep_in_memory=False, split='train')
    for s in ds:
        ds_length = len(s[subset['target'][0]['name']])
        logging.info(f"  Processing next sample of length {ds_length}")
        freq_counts[subset['freq']] = freq_counts.get(subset['freq'], 0) + ds_length
        for t in subset['target']:
            domain_counts[t['domain']] = domain_counts.get(t['domain'], 0) + ds_length
    # save json intermediate results
    os.makedirs("scripts/dataset/output", exist_ok=True)
    with open("scripts/dataset/output/tsmixup_condition_balancing_freq_counts.json", "w") as f:
        json.dump(freq_counts, f, indent=4)
    with open("scripts/dataset/output/tsmixup_condition_balancing_domain_counts.json", "w") as f:
        json.dump(domain_counts, f, indent=4)

logging.info("--- Frequency Counts ---")
for freq, count in freq_counts.items():
    logging.info(f"Frequency: {freq}, Count: {count}")
logging.info("\n--- Domain Counts ---")
for domain, count in domain_counts.items():
    logging.info(f"Domain: {domain}, Count: {count}")