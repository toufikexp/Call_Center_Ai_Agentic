"""argparse CLI for the batch runner.

Subcommands:
    run     Process a directory or manifest of audio files.
"""
import argparse
import sys

from src.batch.runner import BatchRunner


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.batch",
        description="Run the call analysis pipeline against many files in one process.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser(
        "run",
        help="Process a directory or manifest of audio files.",
    )
    src_group = run.add_mutually_exclusive_group(required=True)
    src_group.add_argument(
        "--input-dir",
        help="Directory containing audio files. Use --pattern to filter.",
    )
    src_group.add_argument(
        "--manifest",
        help="Text file with one absolute audio path per line "
             "(blank lines and lines starting with '#' are skipped).",
    )
    run.add_argument(
        "--pattern",
        default="*.wav",
        help="Glob pattern when --input-dir is used (default: %(default)s).",
    )
    run.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Parallel worker threads. Default: %(default)s. "
             "Increase carefully on CPU-only boxes — too high starves the model.",
    )
    run.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N files (smoke testing).",
    )
    run.add_argument(
        "--no-skip-completed",
        action="store_true",
        help="Re-process files even when a previous attempt landed in COMPLETE. "
             "Default behaviour is to skip them (idempotent resume).",
    )
    run.add_argument(
        "--batch-name",
        default=None,
        help="Free-text label persisted to batch_runs.notes.",
    )
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and print the input list, then exit. No processing.",
    )
    return parser


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)

    if args.cmd == "run":
        runner = BatchRunner(
            input_dir=args.input_dir,
            manifest=args.manifest,
            pattern=args.pattern,
            workers=args.workers,
            limit=args.limit,
            skip_completed=not args.no_skip_completed,
            batch_name=args.batch_name,
            dry_run=args.dry_run,
        )
        return runner.run()

    return 1


if __name__ == "__main__":
    sys.exit(main())
