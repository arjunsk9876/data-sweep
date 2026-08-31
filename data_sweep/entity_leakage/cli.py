import argparse
import sys
from typing import List, Optional

import pandas as pd

from data_sweep.entity_leakage.findings import EntityLeakageFinding, from_candidate_keys
from data_sweep.entity_leakage.fixes import SklearnMissingError, generate_fix_code
from data_sweep.entity_leakage.io import DatasetLoadError, load_datasets
from data_sweep.entity_leakage.keys import score_candidate_keys
from data_sweep.entity_leakage.leakage import check_cross_split_leakage, to_entity_leakage_findings
from data_sweep.entity_leakage.report import format_audit_report, format_single_file_report


def add_audit_subparser(parser: argparse.ArgumentParser) -> None:
    parser.description = (
        "Detect hidden entity/group leakage between a train and test split -- no "
        "need to say which column is the entity key, it's inferred automatically."
    )
    parser.epilog = (
        "examples:\n"
        "  data-sweep audit train.csv --test test.csv               check for leakage between the two files\n"
        "  data-sweep audit train.csv                                list candidate entity/group key columns only\n"
        "  data-sweep audit train.csv --test test.csv --fix         also print a runnable fix for the leak found\n"
        "  data-sweep audit train.csv --test test.csv --fix-file fix.py   write the fix to a file instead\n"
    )
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    parser.add_argument("input_csv", help="Path to the training/primary CSV file.")
    parser.add_argument(
        "--test",
        dest="test_csv",
        help="Path to the test CSV file, to check for entity/group leakage against the training file. "
             "Omit for single-file mode (candidate-key inference only, informational).",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Print a runnable Python snippet that fixes the leak found (requires scikit-learn).",
    )
    parser.add_argument(
        "--fix-file",
        dest="fix_file",
        help="Write the runnable fix snippet to this path instead of printing it (requires scikit-learn).",
    )


def _collect_entity_findings(
    args: argparse.Namespace, train_df: pd.DataFrame, test_df: Optional[pd.DataFrame]
) -> List[EntityLeakageFinding]:
    if test_df is None:
        candidates = score_candidate_keys(train_df)
        print(format_single_file_report(candidates))
        return from_candidate_keys(candidates)

    findings = check_cross_split_leakage(train_df, test_df)
    print(format_audit_report(findings))
    return to_entity_leakage_findings(findings)


def run_audit(args: argparse.Namespace) -> None:
    try:
        train_df, test_df = load_datasets(args.input_csv, args.test_csv)
    except DatasetLoadError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    entity_findings = _collect_entity_findings(args, train_df, test_df)

    if not (args.fix or args.fix_file):
        return

    if not entity_findings:
        print("\nNothing to fix -- no entity/group key finding to generate a fix for.")
        return

    try:
        code = generate_fix_code(entity_findings)
    except SklearnMissingError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.fix:
        print("\n" + code)

    if args.fix_file:
        with open(args.fix_file, "w") as f:
            f.write(code)
        print(f"\nWrote fix code to {args.fix_file}")
