"""Measure cold Tokenizer construction and whole-process peak RSS in fresh processes."""

import argparse
import json
import platform
import resource
import statistics
import subprocess
import sys
from pathlib import Path
from time import perf_counter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    if args.child:
        from bn_tokenizers_embedding import Tokenizer

        started = perf_counter()
        with Tokenizer(args.model):
            elapsed = perf_counter() - started
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if platform.system() != "Darwin":
                peak *= 1024
            print(json.dumps({"seconds": elapsed, "peak_process_rss_bytes": peak}))
        return
    command = [sys.executable, str(Path(__file__).resolve()), "--model", str(args.model), "--child"]
    samples = [json.loads(subprocess.check_output(command, text=True)) for _ in range(args.repeats)]
    print(
        json.dumps(
            {
                "model": str(args.model),
                "samples": samples,
                "median_load_seconds": statistics.median(sample["seconds"] for sample in samples),
                "median_peak_process_rss_bytes": statistics.median(
                    sample["peak_process_rss_bytes"] for sample in samples
                ),
                "scope": "fresh process; Tokenizer constructor includes native library startup; "
                "RSS is whole-process peak, not isolated model memory",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
