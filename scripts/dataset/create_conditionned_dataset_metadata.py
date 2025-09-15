import datasets
import json
import numpy as np

def main():
    with open("scripts/dataset/subsets.json", "r") as f:
        subsets = json.load(f)

    for _, subset in subsets.items():
        print(f"Processing subsets {subset['name']}...")
        subset.pop("domain", None)
        subset.pop("target_name", None)

        ds = datasets.load_dataset(subset['hf_dataset_name'], subset['hf_subset_name'], streaming=True, split='train', trust_remote_code=True)
        sample = next(iter(ds))

        subset['target'] = []

        for col in sample.keys():
            if col not in ["timestamp", "date", "time"] and isinstance(sample[col], list):
                subset['target'].append({
                    "name": col,
                    "domain": "#TODO"
                })

    with open("scripts/dataset/subsets_new.json", "w") as f:
        json.dump(subsets, f, indent=4)

if __name__ == "__main__":
    main()