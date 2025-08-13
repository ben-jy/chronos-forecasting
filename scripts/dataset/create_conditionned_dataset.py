# imports
from pathlib import Path
import datasets
from gluonts.dataset.arrow.enc import into_arrow_batches
import argparse
import numpy as np
import pyarrow as pa
import json

with open("scripts/dataset/subsets.json", "r") as f:
    subsets = json.load(f)
    
def split_sequence(seq, max_len):
    seq = np.array(seq)
    # mean scaling
    # mean = np.nanmean(np.abs(seq))
    # seq = seq / mean if mean != 0 else seq
    if max_len is not None and len(seq) > max_len:
        return [seq[i:i + max_len] for i in range(0, len(seq), max_len)]
    else:
        return [seq]

def process_features(features, ndim, max_timesteps=None):
    series = []
    if ndim > 1:
        for f in features:
            series.extend(split_sequence(f, max_timesteps))
    else:
        series.extend(split_sequence(features, max_timesteps))
    return series

def process_batch(ds_features, batch, max_timesteps=None, target_name='target'):
    target_ndim = ds_features[target_name].length
    series = []
    for entry in batch[target_name]:
        series.extend(process_features(entry, target_ndim, max_timesteps))
    if 'past_feat_dynamic_real' in ds_features:
        past_feat_ndim = ds_features['past_feat_dynamic_real'].length
        for entry in batch['past_feat_dynamic_real']:
            series.extend(process_features(entry, past_feat_ndim, max_timesteps))
    return series

def batched(ds, batch_size):
    for i in range(0, len(ds), batch_size):
        yield ds[i:i+batch_size]

def main():
    parser = argparse.ArgumentParser(description="Convert a huggingface to arrow files")
    parser.add_argument("--batch_size", type=int, default=None, help="The batch size to load in memory before writing to the arrow file.")
    parser.add_argument("--max_timesteps", type=int, default=None, help="Maximum number of time steps per series. Longer series will be split.")
    parser.add_argument("--arrow_output_path", type=str, default="lotsa.arrow", help="Output Arrow file")
    args = parser.parse_args()

    writer = None
    output_path = Path(args.arrow_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    default_start = np.datetime64("2000-01-01 00:00", "s")

    for _, subset in subsets.items():
        print(f"Processing subsets {subset['name']}...")

        ds = datasets.load_dataset(subset['hf_dataset_name'], subset['hf_subset_name'], keep_in_memory=False, split='train', trust_remote_code=True)
        ds.set_format("numpy")
        
        print(f"Keys : {ds[0].keys()}")
        subset_target_name = subset['target_name']
        subset_freq = subset['freq']
        subset_domain = subset['domain']

        batch_size = args.batch_size if args.batch_size is not None else len(ds)
        
        for i, batch in enumerate(batched(ds, batch_size)):
            print(f"Processing batch {i + 1}: [{i * batch_size}:{min((i + 1) * batch_size, len(ds))}]")
            processed_series = process_batch(ds.features, batch, max_timesteps=args.max_timesteps, target_name=subset_target_name)
            dataset = [{"start": default_start, "target": np.array(ts, dtype=np.float32),
                        "frequency": subset_freq, "domain": subset_domain} for ts in processed_series]

            batches = list(into_arrow_batches(dataset, flatten_arrays=True))
            
            if not batches:
                continue

            if writer is None:
                schema = batches[0].schema
                options = pa.ipc.IpcWriteOptions(compression="lz4")
                writer = pa.RecordBatchFileWriter(output_path, schema, options=options)

            for b in batches:
                writer.write_batch(b)

    if writer is not None:
        writer.close()

    print("All batches written.")

if __name__ == "__main__":
    main()

