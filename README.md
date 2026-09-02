# data-sweep

You wrote `pd.read_csv()`, you're about to fit a model, and somewhere in that CSV there's a column that's secretly a copy of your target, or a customer ID that shows up in both your train and test split. Nothing in a `.describe()` or `.info()` call is going to tell you that. data-sweep is a CLI (and, as of the latest release, an MCP server) that looks for exactly that kind of thing before it costs you an afternoon of "why is my test accuracy 98%."

It does two jobs: **clean** a CSV of the ordinary stuff (duplicates, missing values, outliers, badly-encoded categoricals), and **audit** a train/test split for the kind of leakage that doesn't show up until your model mysteriously does way too well. Every finding — from either command — comes with a plain-English explanation, not just a column name and a number.

## Why this exists

Most data-quality tools (Great Expectations, YData Profiling, Deepchecks) assume a human opens a notebook, imports the library, and remembers to run it. That's a real gap: the leakage bugs that ruin a model are exactly the kind nobody thinks to check for until the model's already been shipped and the numbers don't hold up in production. data-sweep is built to be run early and often — from the terminal while you're still exploring a dataset, or automatically by an AI coding assistant that's scaffolding a project for you, before it writes a single line of `sklearn.fit()`.

## Setup

Requires Python 3.9+.

```bash
git clone <this-repo-url>
cd data-sweep
pip install -e .
```

That puts a `data-sweep` command on your PATH.

## `clean` — fix the ordinary stuff

```bash
data-sweep clean yourfile.csv
```

Scans the file, decides what to do about each issue it finds, and writes two things: a cleaned CSV and a markdown report explaining every change. It catches:

- **Duplicate rows** — exact copies, removed, keeping the first occurrence.
- **Missing values** — a column over 50% missing gets dropped; anything under that gets filled (median for numeric, mode for categorical).
- **Outliers** — flagged using the IQR method (1.5× beyond the interquartile range).
- **Mixed-type columns** — a column that's mostly numbers with a few stray strings gets coerced, the exceptions reported.
- **Constant columns** — a column with one distinct value (or none at all) carries no signal, so it's dropped.
- **Categorical encoding** — decides per-column whether a categorical is ordinal, an identifier (too many unique values to one-hot sanely), a good candidate for one-hot encoding, or worth bucketing the long tail of.

```bash
data-sweep clean yourfile.csv --target label   # also checks the target column for leakage and class imbalance
data-sweep clean yourfile.csv --dry-run        # write the report only, skip the cleaned CSV
```

With `--target` set, `clean` also flags any feature suspiciously correlated with the label (>0.95 by default — that's usually not a real predictor, it's a leak) and warns if one class dominates the target (>90% by default). Neither of those checks run without `--target`, since there's nothing to compare against.

You'll be prompted interactively to pick which columns to keep. Pass `--columns "a,b,c"` or `--all-columns` if you'd rather skip that and run non-interactively (handy in a script or CI).

## `audit` — catch leakage before you train

This is the part most tools skip entirely. Two kinds of leakage, both silent, both the kind that makes a model look great in evaluation and fall apart the moment it sees real data.

**Entity/group leakage** — the same customer, user, or device shows up in both your train and test files. The model has effectively already seen that entity, so your test score is a lie about how well it generalizes.

```bash
data-sweep audit train.csv --test test.csv
```

data-sweep infers the entity/group key on its own — you don't tell it which column is the customer ID, it figures that out from cardinality, naming patterns, and format signals (things like a UUID-shaped or `_id`-suffixed column score higher). If a candidate key overlaps between the two files, it's flagged with a severity and a percentage. Add `--fix` and it prints a runnable `GroupShuffleSplit` (two files) or `GroupKFold` (one file) snippet using the actual column it found, ready to drop into your pipeline. Use `--fix-file path.py` to write it straight to a file instead of printing it.

Run `audit` on a single file (no `--test`) and it just lists candidate entity/group keys — useful before you've even split the data, so you split around the right column from the start instead of finding out after the fact.

**Temporal leakage** — a computed feature (`total_purchases`, `avg_response_time`, anything aggregation-shaped) that was accidentally built using data from *after* the event you're trying to predict.

```bash
data-sweep audit train.csv --target churn --event-time cancel_date --record-time snapshot_date
```

This one's a heuristic, not a certainty, and the tool is upfront about that — findings are worth investigating, not proof of a bug. It combines three signals: does the column name look like a running aggregate, is it correlated with how much time elapsed past the label event, and is it suspiciously predictive of the target on its own for a single feature. `--event-time`/`--record-time` are optional but sharpen signal #2 considerably; without them the check still runs off the other two signals, just flagged as reduced-confidence.

## MCP server — for agentic coding tools

The whole reason "catch it before training" is a real goal and not just a nice idea is that an AI coding assistant scaffolding an ML project can call this itself, mid-session, without you having to remember it exists. data-sweep runs as an MCP server exposing the same checks over stdio — no separate logic, no reimplementation, it's the identical code path the CLI uses.

Requires Python 3.10+ (the MCP SDK's floor, not data-sweep's — the core CLI still only needs 3.9+).

```bash
pip install -e ".[mcp]"
```

**Claude Code:**

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

Three tools:

- **`detect_leakage_before_training`** — runs entity-leakage (always) and temporal-leakage (if you pass a target) checks, returns findings as JSON. Same `path`/`test_path`/`target`/`event_time`/`record_time` inputs as the CLI.
- **`generate_fix`** — hand it entity-leakage findings, get back the runnable fix snippet. Nothing gets executed or written to disk by the tool itself; it just returns code text.
- **`list_checks`** — a quick capability listing, for when an agent wants to know what's available before deciding whether to bother.

Guardrails, since this runs unattended inside someone else's agent session with no progress bar to watch: files over 25MB or 500K rows get rejected with a clear message pointing back at the CLI (which has no such cap), and every check has a hard 30-second timeout regardless of file size. A missing file, an unreadable path, or a malformed CSV comes back as a clean tool error, never a stack trace.

Here's roughly what it looks like in practice — an agent scaffolding a churn model, unprompted:

```
> set up a churn model from customer_train.csv / customer_test.csv

[agent reads the files, then calls detect_leakage_before_training before writing training code]

  detect_leakage_before_training(path="customer_train.csv", test_path="customer_test.csv")
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

That's from an actual test run, not a mockup — same tool, same synthetic leak, unmodified from what the agent produced on its own.

## Development

```bash
pip install -e ".[dev]"
python -m pytest
mypy data_sweep run.py
```

The codebase is organized around where the logic actually runs:

- `data_sweep/detectors/` — `clean`'s detection logic, one module per issue (duplicates, missing values, outliers, mixed types, constants, categorical encoding, class imbalance, multicollinearity, target leakage).
- `data_sweep/entity_leakage/` — `audit`'s logic: candidate-key detection (`keys.py`), cross-split overlap checks (`leakage.py`), temporal-leakage signals (`temporal.py`), fix-code generation (`fixes.py`), and report formatting.
- `data_sweep/mcp_server.py` — the MCP wrapper. No detection logic lives here; it's a thin adapter with its own guardrails (size caps, timeouts) on top of the same functions the CLI calls.

CI runs pytest and mypy on every push and pull request. 590 tests as of this writing, covering true positives, true negatives (make sure it doesn't cry wolf), and the rough edges — malformed timestamps, unparseable dates, tiny datasets where the statistics get noisy.
