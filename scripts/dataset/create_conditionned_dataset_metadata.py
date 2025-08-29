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

        ds = datasets.load_dataset(subset['hf_dataset_name'], subset['hf_subset_name'], keep_in_memory=False, split='train', trust_remote_code=True)
        ds.set_format("numpy")

        subset['target'] = []

        for col in ds.column_names:
            if col not in ["timestamp", "date", "time"] and ds[col].dtype == np.ndarray:
                subset['target'].append({
                    "name": col,
                    "domain": "#TODO"
                })

    with open("scripts/dataset/subsets_new.json", "w") as f:
        json.dump(subsets, f, indent=4)

if __name__ == "__main__":
    main()