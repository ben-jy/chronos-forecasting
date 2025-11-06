#!/usr/bin/env python3
from pathlib import Path
import argparse
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
from collections import defaultdict
import logging
import sys
import time

def setup_logging(output_arrow_path: Path):
    """Configure logging to both stdout and a .log file next to the output file."""
    log_path = output_arrow_path.with_suffix(output_arrow_path.suffix + ".log")
    log_format = "%(asctime)s [%(levelname)s] %(message)s"

    handlers = [
        logging.FileHandler(log_path, mode="w"),
        logging.StreamHandler(sys.stdout)
    ]

    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=handlers,
        force=True,  # override default handlers
    )
    logging.info(f"Logging to {log_path}")

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

    setup_logging(args.output_arrow_path)
    t0 = time.time()

    in_path, out_path = args.input_arrow_path, args.output_arrow_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info(f"Opening input file: {in_path}")
    with in_path.open("rb") as f:
        reader = ipc.open_file(f)
        in_schema = reader.schema
        n_batches = reader.num_record_batches
        logging.info(f"{n_batches} batches in input file. Calculating total number of rows...")
        batch_sizes = np.fromiter(
            (reader.get_batch(i).num_rows for i in range(n_batches)),
            dtype=np.int64, count=n_batches
        )

    starts = np.concatenate(([0], np.cumsum(batch_sizes)[:-1]))
    n_rows = int(batch_sizes.sum())
    logging.info(f"Input file has {n_batches:,} batches, total {n_rows:,} rows")

    out_schema = large_list_schema(in_schema) if args.upcast_large_list else in_schema
    opts = pa.ipc.IpcWriteOptions(compression="lz4")

    if n_rows == 0:
        with out_path.open("wb") as g:
            ipc.RecordBatchFileWriter(g, out_schema, options=opts).close()
        logging.warning("Empty table. Wrote empty output.")
        return

    rng = np.random.default_rng(args.random_seed)
    perm = rng.permutation(n_rows)
    logging.info(f"Generated random permutation for {n_rows:,} rows")

    total_written = 0
    with in_path.open("rb") as f_in, out_path.open("wb") as f_out:
        reader = ipc.open_file(f_in)
        writer = ipc.RecordBatchFileWriter(f_out, out_schema, options=opts)
        try:
            for start in range(0, n_rows, args.write_batch_size):
                end = min(start + args.write_batch_size, n_rows)
                window = perm[start:end]

                batch_ids = np.searchsorted(starts, window, side="right") - 1
                local_idx = window - starts[batch_ids]

                groups = defaultdict(list)
                for b, li in zip(batch_ids, local_idx):
                    groups[int(b)].append(int(li))

                partial_tables = []
                for b, li_list in groups.items():
                    if not li_list:
                        continue
                    rb = reader.get_batch(b)
                    tbl = pa.Table.from_batches([rb])
                    if args.upcast_large_list:
                        tbl = tbl.cast(out_schema)
                    taken = tbl.take(pa.array(np.array(li_list, dtype=np.int64)))
                    partial_tables.append(taken)

                if partial_tables:
                    window_table = pa.concat_tables(partial_tables, promote=True).combine_chunks()
                    for out_rb in window_table.to_batches(max_chunksize=args.write_batch_size):
                        writer.write_batch(out_rb)
                        total_written += out_rb.num_rows

                if (start // args.write_batch_size) % 10 == 0:
                    pct = 100.0 * end / n_rows
                    elapsed = time.time() - t0
                    logging.info(f"Progress: {pct:.2f}% ({end:,}/{n_rows:,} rows), elapsed {elapsed/3600:.2f}h")

        finally:
            writer.close()

    total_time = time.time() - t0
    logging.info(f"✅ Done: wrote shuffled file -> {out_path}")
    logging.info(f"Total rows written: {total_written:,}")
    logging.info(f"Total time: {total_time/3600:.2f} hours ({total_time/60:.1f} minutes)")

if __name__ == "__main__":
    main()
