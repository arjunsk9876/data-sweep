import argparse


def add_audit_subparser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input_csv", help="Path to the training/primary CSV file.")
    parser.add_argument(
        "--test",
        dest="test_csv",
        help="Path to the test CSV file, to check for entity/group leakage against the training file. "
             "Omit for single-file mode (candidate-key inference only, informational).",
    )


def run_audit(args: argparse.Namespace) -> None:
    print("data-sweep audit: not yet implemented.")
