# imports
from pathlib import Path
import datasets
from gluonts.dataset.arrow.enc import into_arrow_batches
import argparse
import numpy as np
import pyarrow as pa
import json
from chronos.utils import get_frequency_id, get_domain_id

DEFAULT_START = np.datetime64("2000-01-01 00:00", "s")

SCHEMA = pa.schema([
    pa.field("start", pa.timestamp("s")),
    pa.field("target", pa.list_(pa.float32())),
    pa.field("freq", pa.int64()),
    pa.field("domain", pa.int64())
])

TOTAL_TIMESTEPS = {
    "weather": 13_360_714_827,
    "transport": 49_232_950,
    "energy": 808_813_081,
    "web": 274_100_000,
}

new_total_timesteps = {
    "weather": 0,
    "transport": 0,
    "energy": 0,
    "web": 0,
    "null": 0,
}

target_timesteps = 1_000_000_000

factor = {
    domain: target_timesteps / total
    for domain, total in TOTAL_TIMESTEPS.items()
}

print("Factors per domain:")
for d, p in factor.items():
    print(f"  {d}: {p:.6f}")

class Counter:
    def __init__(self, value: int = 0):
        self.value = value

    def increment(self, step: int = 1):
        self.value += step

total_counter = Counter(0)
filtered_counter = Counter(0)
    
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

def is_all_nan(arr: np.ndarray) -> bool:
    return np.isnan(arr).all()

def is_constant_ignoring_nan(arr: np.ndarray) -> bool:
    x = arr[~np.isnan(arr)]
    if x.size <= 1:
        # Treat length 0 as "all NaN" (filtered elsewhere) and length 1 as constant
        return x.size == 1
    return np.max(x) == np.min(x)

def process_batch(batch, target, max_timesteps=None, freq=0, 
                  keep_constant_prob: float = 0.01, rng: np.random.Generator = None, dataset_name: str=""):
    if rng is None:
        rng = np.random.default_rng()

    subset = []
    for t in target:
        series = []
        target_name = t['name']
        target_domain_id = get_domain_id(t['domain'])

        for entry in batch[target_name]:
            series.extend(split_sequence(entry, max_timesteps))

        # Filter and probabilistically include
        for ts in series:
            total_counter.increment()
            ts = np.array(ts, dtype=np.float32)

            # if domain is null, we keep the series
            if t['domain'] is not None:
                if factor[t['domain']] < 1.0:
                    if rng.random() > factor[t['domain']]:
                        filtered_counter.increment()
                        continue
                else:
                    # oversampling
                    n_copies = int(factor[t['domain']])
                    for _ in range(n_copies - 1):
                        subset.append({
                            "start": DEFAULT_START,
                            "target": ts,
                            "freq": freq,
                            "domain": target_domain_id
                        })
                        new_total_timesteps[t['domain']] += len(ts)

            if is_all_nan(ts):
                filtered_counter.increment()
                continue

            if is_constant_ignoring_nan(ts):
                filtered_counter.increment()
                continue

            if t['domain'] is None:
                new_total_timesteps["null"] += len(ts)
            else:
                new_total_timesteps[t['domain']] += len(ts)

            subset.append({
                "start": DEFAULT_START,
                "target": ts,
                "freq": freq,
                "domain": target_domain_id
            })

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
                                             target=subset_target, freq=subset_freq_id, dataset_name=subset['name'])

            if not subset_batch:
                continue
            
            batches = list(into_arrow_batches(subset_batch, flatten_arrays=True))
            
            if not batches:
                continue

            if writer is None:
                options = pa.ipc.IpcWriteOptions(compression="lz4")
                writer = pa.RecordBatchFileWriter(output_path, SCHEMA, options=options)

            for b in batches:
                writer.write_batch(b.cast(target_schema=SCHEMA))

    if writer is not None:
        writer.close()

    print("All batches written.")
    print(f"Total series processed: {total_counter.value}")
    print(f"Total series filtered out: {filtered_counter.value}")
    if total_counter.value:
        print(f"Proportion filtered: {filtered_counter.value / total_counter.value:.2%}")
    else:
        print("Proportion filtered: n/a (no series processed)")
    print("New total timesteps per domain:")
    for d, t in new_total_timesteps.items():
        print(f"  {d}: {t}")

if __name__ == "__main__":
    main()

