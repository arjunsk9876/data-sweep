import argparse
import sys
from typing import List

from data_sweep.entity_leakage.findings import EntityLeakageFinding, from_candidate_keys
from data_sweep.entity_leakage.fixes import generate_fix_code
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
        "  data-sweep audit train.csv --test test.csv          check for leakage between the two files\n"
        "  data-sweep audit train.csv                           list candidate entity/group key columns only\n"
        "  data-sweep audit train.csv --test test.csv --fix    also print a runnable fix for the leak found\n"
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


def _print_fix(entity_findings: List[EntityLeakageFinding]) -> None:
    if not entity_findings:
        print("\nNothing to fix -- no entity/group key finding to generate a fix for.")
        return
    print("\n" + generate_fix_code(entity_findings))


def run_audit(args: argparse.Namespace) -> None:
    try:
        train_df, test_df = load_datasets(args.input_csv, args.test_csv)
    except DatasetLoadError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if test_df is None:
        candidates = score_candidate_keys(train_df)
        print(format_single_file_report(candidates))
        if args.fix:
            _print_fix(from_candidate_keys(candidates))
    else:
        findings = check_cross_split_leakage(train_df, test_df)
        print(format_audit_report(findings))
        if args.fix:
            _print_fix(to_entity_leakage_findings(findings))
