# data-sweep

A CLI tool that scans a CSV, fixes common data quality issues, and audits train/test splits for hidden leakage. Every finding comes with a plain-English explanation of what it means and why it matters.

## Setup

Requires Python 3.9+.

```bash
git clone <this-repo-url>
cd data-sweep
pip install -e .
```

This gives you a `data-sweep` command on your PATH.

## Usage

Two subcommands: `clean` and `audit`.

**`clean`** profiles a CSV, fixes common issues (duplicates, missing values, outliers, bad categorical encoding, and more), and writes a cleaned CSV plus a markdown report.

```bash
data-sweep clean yourfile.csv
data-sweep clean yourfile.csv --target label   # also flags leakage and class imbalance against the target
data-sweep clean yourfile.csv --dry-run        # report only, no cleaned CSV written
```

You'll be prompted to pick which columns to keep. Use `--columns "a,b,c"` or `--all-columns` to skip the prompt.

**`audit`** checks a train/test split for leakage.

```bash
data-sweep audit train.csv --test test.csv
```

It automatically detects entity/group key columns (customer id, device id, etc.) and flags it if the same entity appears in both files, which inflates your test score. Add `--fix` to get a runnable `GroupShuffleSplit`/`GroupKFold` snippet using the real column it found.

```bash
data-sweep audit train.csv --target churn --event-time cancel_date --record-time snapshot_date
```

This also checks for temporal leakage: computed features (like `total_purchases`) that may have used data from after the label event. This check is heuristic, not proof, so treat findings as worth investigating rather than confirmed bugs. `--event-time`/`--record-time` are optional but give a stronger signal; without them the check still runs, just at reduced confidence.

## MCP server

data-sweep also runs as an MCP server, so an agentic coding tool can call `audit_dataset` and `generate_fix` itself mid-session -- no need for a human to remember to run the CLI. Same underlying checks as the CLI, just exposed as tools. Requires Python 3.10+.

```bash
pip install -e ".[mcp]"
```

**Claude Code** (`claude mcp add`):

```bash
claude mcp add data-sweep -- data-sweep mcp
```

**Cursor** (`.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "data-sweep": {
      "command": "data-sweep",
      "args": ["mcp"]
    }
  }
}
```

Three tools: `audit_dataset` (entity/temporal leakage findings as JSON), `generate_fix` (turns entity-leakage findings into runnable fix code), and `list_checks` (what's available, for an agent deciding whether to bother). Stdio transport only, local files only -- it never writes to disk or executes generated code itself.

Example, mid-scaffolding:

```
> set up a churn model from customer_train.csv / customer_test.csv

[agent reads the files, then calls audit_dataset before writing training code]

  audit_dataset(path="customer_train.csv", test_path="customer_test.csv")
  -> 1 finding: entity_leakage, candidate_key="customer_id", severity="high",
     overlap_pct=18.4 -- 18.4% of test customers also appear in train

I found entity leakage before training anything: customer_id overlaps
18.4% between your train and test files, which would inflate your test
score. Want me to apply the suggested GroupShuffleSplit fix?

> yes

  generate_fix(findings=[...])
  -> GroupShuffleSplit snippet using customer_id

[agent applies the fix, re-splits the data, then proceeds to train]
```

## Development

```bash
pip install -e ".[dev]"
python -m pytest
mypy data_sweep run.py
```

`data_sweep/detectors/` holds the `clean` detection logic, one module per issue. `data_sweep/entity_leakage/` holds the `audit` logic: candidate key detection, cross-split leakage checks, temporal leakage checks, fix-code generation, and reporting. `data_sweep/mcp_server.py` is a thin MCP wrapper around that same logic -- no detection code lives there. CI runs pytest and mypy on every push and pull request.
