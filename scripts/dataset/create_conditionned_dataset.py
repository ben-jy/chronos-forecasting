# imports
from pathlib import Path
import datasets
from gluonts.dataset.arrow.enc import into_arrow_batches
import argparse
import numpy as np
import pyarrow as pa
import json
from pandas.tseries.frequencies import to_offset

DEFAULT_START = np.datetime64("2000-01-01 00:00", "s")

def get_frequency_id(freq: str) -> int:
    try:
        offset = to_offset(freq)
    except ValueError:
        raise ValueError(f"Invalid frequency string: {freq}")

    rule = offset.rule_code

    high_freq = {"ns", "us", "ms", "s", "min", "h"}
    mid_freq = {"D", "B", "W", "W-MON", "W-SUN"}

    if rule in high_freq:
        return 0
    elif rule in mid_freq:
        return 1
    else:
        return 2

def get_domain_id(domain: str) -> int:
    if domain == "transport":
        return 0
    elif domain == "weather":
        return 1
    elif domain == "energy":
        return 2
    elif domain == "web":
        return 3
    else:
        # basically means any
        # important question though : in chronos, should we set a token for "any", or should be just consider that there
        # is not conditioning ?
        return 4
    
def split_sequence(seq, max_len):
    seq = np.array(seq)
    if seq.ndim == 0:
        # empty
        return []
    if seq.ndim > 1:
        raise ValueError("Sequence must be 1D")
    # mean scaling
    # mean = np.nanmean(np.abs(seq))
    # seq = seq / mean if mean != 0 else seq
    # check if seq is only a number
    if max_len is not None and seq.size > max_len:
        return [seq[i:i + max_len] for i in range(0, len(seq), max_len)]
    else:
        return [seq]

def process_batch(batch, target, max_timesteps=None, freq=0):
    subset = []
    for t in target:
        series = []
        target_name = t['name']
        target_domain_id = get_domain_id(t['domain'])
        for entry in batch[target_name]:
            series.extend(split_sequence(entry, max_timesteps))
        subset.extend([
            {
                "start": DEFAULT_START, "target": np.array(ts, dtype=np.float32),
                "freq": freq, "domain": target_domain_id
            } for ts in series])
    return subset

def batched(ds, batch_size):
    for i in range(0, len(ds), batch_size):
        yield ds[i:i+batch_size]

def main():
    parser = argparse.ArgumentParser(description="Convert a huggingface to arrow files")
    parser.add_argument("--subset_info_file", type=str, required=True, default="subset.json")
    parser.add_argument("--batch_size", type=int, default=None, help="The batch size to load in memory before writing to the arrow file.")
    parser.add_argument("--max_timesteps", type=int, default=None, help="Maximum number of time steps per series. Longer series will be split.")
    parser.add_argument("--arrow_output_path", type=str, default="output.arrow", help="Output Arrow file")
    args = parser.parse_args()

    with open(args.subset_info_file, "r") as f:
        subsets = json.load(f)

    writer = None
    output_path = Path(args.arrow_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for _, subset in subsets.items():
        print(f"Processing subsets {subset['name']}...")

        ds = datasets.load_dataset(subset['hf_dataset_name'], subset['hf_subset_name'], keep_in_memory=False, split='train', trust_remote_code=True)
        ds.set_format("numpy")
        
        print(f"Keys : {ds[0].keys()}")
        subset_target = subset['target']
        subset_freq_id = get_frequency_id(subset['freq'])

        batch_size = args.batch_size if args.batch_size is not None else len(ds)
        
        for i, batch in enumerate(batched(ds, batch_size)):
            print(f"Processing batch {i + 1}: [{i * batch_size}:{min((i + 1) * batch_size, len(ds))}]")
            subset_batch = process_batch(batch, max_timesteps=args.max_timesteps,
                                             target=subset_target, freq=subset_freq_id)

            batches = list(into_arrow_batches(subset_batch, flatten_arrays=True))
            
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

