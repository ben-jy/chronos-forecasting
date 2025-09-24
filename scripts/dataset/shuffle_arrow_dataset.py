#!/usr/bin/env python3
from pathlib import Path
import argparse
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
from collections import defaultdict

def large_list_schema(schema: pa.Schema) -> pa.Schema:
    """Upcast any list<T> columns to large_list<T> to avoid 32-bit offset overflow."""
    fields = []
    for f in schema:
        t = f.type
        if pa.types.is_list(t):
            fields.append(pa.field(f.name, pa.large_list(t.value_type)))
        else:
            fields.append(f)
    return pa.schema(fields)

def main():
    p = argparse.ArgumentParser(description="Shuffle a big Arrow IPC file without loading it in RAM.")
    p.add_argument("--input_arrow_path", type=Path, required=True)
    p.add_argument("--output_arrow_path", type=Path, required=True)
    p.add_argument("--random_seed", type=int, default=None)
    p.add_argument("--write_batch_size", type=int, default=100_000,
                   help="Rows per shuffled output chunk (memory cap).")
    p.add_argument("--upcast_large_list", action="store_true", default=True,
                   help="Cast list<T> columns to large_list<T> in output (recommended).")
    args = p.parse_args()

    in_path, out_path = args.input_arrow_path, args.output_arrow_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Discover metadata & batch sizes (single lightweight pass)
    with in_path.open("rb") as f:
        reader = ipc.open_file(f)
        in_schema = reader.schema
        n_batches = reader.num_record_batches
        batch_sizes = np.fromiter(
            (reader.get_batch(i).num_rows for i in range(n_batches)),
            dtype=np.int64, count=n_batches
        )
    starts = np.concatenate(([0], np.cumsum(batch_sizes)[:-1]))
    n_rows = int(batch_sizes.sum())

    # Prepare output
    out_schema = large_list_schema(in_schema) if args.upcast_large_list else in_schema
    opts = pa.ipc.IpcWriteOptions(compression="lz4")

    if n_rows == 0:
        with out_path.open("wb") as g:
            ipc.RecordBatchFileWriter(g, out_schema, options=opts).close()
        print("Empty table. Wrote empty output.")
        return

    # Global permutation fully in RAM (int64 ~ 8*N bytes; OK for your RAM)
    rng = np.random.default_rng(args.random_seed)
    perm = rng.permutation(n_rows)

    with in_path.open("rb") as f_in, out_path.open("wb") as f_out:
        reader = ipc.open_file(f_in)
        writer = ipc.RecordBatchFileWriter(f_out, out_schema, options=opts)
        try:
            # Process permutation in windows to bound memory
            for start in range(0, n_rows, args.write_batch_size):
                end = min(start + args.write_batch_size, n_rows)
                window = perm[start:end]

                # Map global -> (batch_id, local_idx)
                batch_ids = np.searchsorted(starts, window, side="right") - 1
                local_idx = window - starts[batch_ids]

                # Group by source batch to read minimal data
                groups = defaultdict(list)
                for b, li in zip(batch_ids, local_idx):
                    groups[int(b)].append(int(li))

                partial_tables = []
                for b, li_list in groups.items():
                    if not li_list:
                        continue
                    # Read just this record batch
                    rb = reader.get_batch(b)
                    # Work as a Table to enable casting and take()
                    tbl = pa.Table.from_batches([rb])
                    if args.upcast_large_list:
                        tbl = tbl.cast(out_schema)
                    taken = tbl.take(pa.array(np.array(li_list, dtype=np.int64)))
                    partial_tables.append(taken)

                if partial_tables:
                    window_table = pa.concat_tables(partial_tables, promote=True).combine_chunks()
                    for out_rb in window_table.to_batches(max_chunksize=args.write_batch_size):
                        writer.write_batch(out_rb)
        finally:
            writer.close()

    print(f"Done: wrote shuffled file -> {out_path}")

if __name__ == "__main__":
    main()
