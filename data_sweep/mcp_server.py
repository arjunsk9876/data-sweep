"""MCP server exposing data-sweep's audit and fix-generation logic as tools.

Thin wrapper only -- every check and fix template here is the exact function
the CLI calls (data_sweep.entity_leakage.*). No detection logic lives in this
module; it just adapts inputs/outputs to the MCP tool-call shape and adds the
guardrails (file-size cap, row cap, timeout) that only make sense for an
unattended agent session rather than a human at a terminal.

Handler functions (the `*_impl` functions) are plain, synchronous, and
mcp-typed-error-only -- they're what the `@server.tool()`-decorated functions
below delegate to, and what tests call directly to exercise the logic without
going through the MCP transport layer.
"""
import asyncio
import concurrent.futures
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from data_sweep.entity_leakage.findings import (
    EntityLeakageFinding,
    TemporalLeakageFinding,
    from_candidate_keys,
)
from data_sweep.entity_leakage.fixes import NoFindingsError, SklearnMissingError, generate_fix_code
from data_sweep.entity_leakage.io import DatasetLoadError, load_datasets
from data_sweep.entity_leakage.keys import score_candidate_keys
from data_sweep.entity_leakage.leakage import check_cross_split_leakage, to_entity_leakage_findings
from data_sweep.entity_leakage.temporal import check_temporal_leakage

# Deliberately smaller than anything the CLI enforces -- an MCP-invoked audit
# runs inside an agent session with no progress indicator, so a slow check on
# a huge file just looks like a hang. Point users at the CLI for big files.
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024
MAX_ROWS = 500_000
AUDIT_TIMEOUT_SECONDS = 30

_T = TypeVar("_T")


def _check_file_guard(path: str) -> None:
    p = Path(path)
    if not p.exists():
        raise ToolError(f"file not found: {path}")
    if not p.is_file():
        raise ToolError(f"not a file: {path}")

    size = p.stat().st_size
    if size > MAX_FILE_SIZE_BYTES:
        raise ToolError(
            f"{path} is {size / 1_048_576:.1f}MB, over the "
            f"{MAX_FILE_SIZE_BYTES / 1_048_576:.0f}MB limit for MCP-invoked audits -- "
            "run `data-sweep audit` from the CLI instead for large files."
        )


def _load_guarded(path: str, test_path: Optional[str]):
    _check_file_guard(path)
    if test_path:
        _check_file_guard(test_path)

    try:
        train_df, test_df = load_datasets(path, test_path)
    except DatasetLoadError as e:
        raise ToolError(str(e)) from None

    for df, label in [(train_df, path), (test_df, test_path)]:
        if df is not None and len(df) > MAX_ROWS:
            raise ToolError(
                f"{label} has {len(df):,} rows, over the {MAX_ROWS:,}-row limit for "
                "MCP-invoked audits -- run `data-sweep audit` from the CLI instead for large files."
            )

    return train_df, test_df


def _run_with_timeout(fn: Callable[..., _T], *args: Any, timeout: Optional[float] = None) -> _T:
    if timeout is None:
        timeout = AUDIT_TIMEOUT_SECONDS  # module global, not a bound default -- respects monkeypatching
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn, *args)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise ToolError(f"audit timed out after {timeout:.0f}s") from None


def _entity_findings_to_dicts(findings: List[EntityLeakageFinding]) -> List[Dict[str, Any]]:
    return [dict(asdict(f), check="entity_leakage") for f in findings]


def _temporal_findings_to_dicts(findings: List[TemporalLeakageFinding]) -> List[Dict[str, Any]]:
    return [dict(asdict(f), check="temporal_leakage") for f in findings]


def audit_dataset_impl(
    path: str,
    test_path: Optional[str] = None,
    target: Optional[str] = None,
    event_time: Optional[str] = None,
    record_time: Optional[str] = None,
) -> Dict[str, Any]:
    """Run entity-leakage (always) and temporal-leakage (if `target` given)
    checks on `path` (and `test_path`, if given), returning JSON-serializable
    findings. Same underlying functions and same findings the CLI's
    `data-sweep audit` command produces.
    """
    train_df, test_df = _load_guarded(path, test_path)

    for label, col in [("target", target), ("event_time", event_time), ("record_time", record_time)]:
        if col is not None and col not in train_df.columns:
            raise ToolError(f"unknown {label} column: {col}")

    if test_df is not None:
        leakage_findings = _run_with_timeout(check_cross_split_leakage, train_df, test_df)
        entity_findings = to_entity_leakage_findings(leakage_findings)
    else:
        candidates = _run_with_timeout(score_candidate_keys, train_df)
        entity_findings = from_candidate_keys(candidates)

    findings = _entity_findings_to_dicts(entity_findings)

    if target is not None:
        temporal_findings = _run_with_timeout(check_temporal_leakage, train_df, target, event_time, record_time)
        findings.extend(_temporal_findings_to_dicts(temporal_findings))

    return {"path": path, "test_path": test_path, "findings": findings, "finding_count": len(findings)}


def generate_fix_impl(findings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Turn entity-leakage findings (the `check: "entity_leakage"` dicts
    audit_dataset returns) into the same runnable fix-code text
    `fixes.generate_fix_code` produces for the CLI's `--fix` flag.
    """
    if not findings:
        raise ToolError("no findings to generate a fix for")

    entity_findings = []
    for f in findings:
        missing = [k for k in ("candidate_key", "severity", "mode") if k not in f]
        if missing:
            raise ToolError(f"finding missing required field(s): {', '.join(missing)}")
        entity_findings.append(EntityLeakageFinding(
            candidate_key=f["candidate_key"],
            uniqueness_ratio=f.get("uniqueness_ratio", 0.0),
            overlap_pct=f.get("overlap_pct", 0.0),
            severity=f["severity"],
            mode=f["mode"],
            example_values=f.get("example_values", []),
        ))

    try:
        code = generate_fix_code(entity_findings)
    except (NoFindingsError, SklearnMissingError) as e:
        raise ToolError(str(e)) from None

    return {"code": code}


def list_checks_impl() -> Dict[str, Any]:
    """Describe available checks -- lets an agent decide whether audit_dataset
    applies without guessing from the tool name alone.
    """
    return {
        "checks": [
            {
                "name": "entity_leakage",
                "description": (
                    "Detects a shared entity/group key (customer id, user id, session id, etc) "
                    "appearing in both a train and test file -- the model would have effectively "
                    "seen test entities during training. With a single file (no test_path), "
                    "infers candidate entity/group keys instead, so a split can be done correctly "
                    "the first time."
                ),
                "modes": ["two_file (path + test_path given)", "single_file (path only)"],
                "inputs": ["path", "test_path (optional)"],
            },
            {
                "name": "temporal_leakage",
                "description": (
                    "Detects computed/aggregated features that may have been built using data "
                    "from after the label event -- e.g. a 'total_purchases' column that kept "
                    "accumulating past the churn date it's meant to predict. Requires a target "
                    "column; event_time/record_time are optional but sharpen the check."
                ),
                "modes": ["enabled by passing target to audit_dataset"],
                "inputs": ["path", "target", "event_time (optional)", "record_time (optional)"],
            },
        ]
    }


server: MCPServer = MCPServer("data-sweep", version="0.4.0")


@server.tool(
    # Named for what it does, not what it's called internally -- MCP clients
    # that only surface tool names up front (no description) until a query
    # matches still get enough signal to reach for this unprompted, right
    # when "before training a model" is exactly the situation the caller is in.
    name="detect_leakage_before_training",
    description=(
        "Run this after loading a CSV and before training a model, especially when the dataset "
        "has separate train/test files -- catches hidden entity/group leakage (the same "
        "customer, user, or session appearing in both splits) and, if a target column is given, "
        "temporal leakage (features computed using data from after the label event). Returns "
        "structured findings only, nothing is written or modified -- pass entity_leakage "
        "findings to generate_fix for a runnable fix."
    ),
)
def audit_dataset(
    path: str,
    test_path: Optional[str] = None,
    target: Optional[str] = None,
    event_time: Optional[str] = None,
    record_time: Optional[str] = None,
) -> Dict[str, Any]:
    return audit_dataset_impl(path, test_path, target, event_time, record_time)


@server.tool(
    name="generate_fix",
    description=(
        "Turn entity_leakage findings from detect_leakage_before_training into a runnable "
        "Python snippet (scikit-learn GroupShuffleSplit or GroupKFold) that re-splits the data "
        "by the leaking key. Call this right after detect_leakage_before_training reports "
        "entity_leakage findings, passing those findings through unchanged. Returns code text "
        "only -- it is never executed or written to disk by this tool."
    ),
)
def generate_fix(findings: List[Dict[str, Any]]) -> Dict[str, Any]:
    return generate_fix_impl(findings)


@server.tool(
    name="list_checks",
    description=(
        "Lists the checks data-sweep can run and what inputs each needs. Call this first if "
        "you're unsure whether detect_leakage_before_training is relevant to the dataset you're "
        "working with."
    ),
)
def list_checks() -> Dict[str, Any]:
    return list_checks_impl()


def main() -> None:
    asyncio.run(server.run_stdio_async())


if __name__ == "__main__":
    main()
