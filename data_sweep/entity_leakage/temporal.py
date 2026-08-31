import re
from typing import List, Optional

import pandas as pd

from data_sweep.entity_leakage.baseline import PredictivenessResult, compute_predictiveness
from data_sweep.entity_leakage.findings import TemporalLeakageFinding

DEFAULT_AGGREGATION_KEYWORDS = [
    "total", "avg", "cumulative", "running", "lifetime", "ytd",
    "sum", "mean", "count", "last", "max", "min",
]

MIN_ROWS_FOR_ELAPSED_CORRELATION = 10

# Genuinely well-behaved historical aggregates are rarely this predictive on
# their own -- an absolute bar, not a comparison against sibling columns
# (PRD frames Signal 3 as relative to "other raw features of a similar
# type", but a fixed threshold is simpler, deterministic, and captures the
# same spirit: a lone feature scoring this high is inherently unusual).
PREDICTIVENESS_HIGH_THRESHOLD = 0.8

# A correlation weaker than this could easily be noise -- only a
# meaningfully positive relationship counts as Signal 2 "firing".
ELAPSED_CORRELATION_STRONG_THRESHOLD = 0.3

_TOKEN_SPLIT_RE = re.compile(r"[^a-zA-Z0-9]+")


def detect_aggregation_name_signal(column_name: str, keywords: List[str] = DEFAULT_AGGREGATION_KEYWORDS) -> bool:
    """Does this column name look like an aggregated/rolling feature (total_,
    avg_, cumulative_, ..._to_date, etc)?

    Weighting signal, not a gate -- plenty of legitimate, correctly-windowed
    features use these same naming conventions, so this alone never decides
    anything. It's one of three signals combined into an overall severity.

    Matches whole tokens only (split on non-alphanumeric characters), same
    approach as the entity-leakage name signal -- a naive substring check on
    "max" would false-positive on "climax_score", and one on "to_date" would
    miss the far more common underscore-separated "_to_date" suffix while
    still risking false positives like "update_date".
    """
    tokens = [t for t in _TOKEN_SPLIT_RE.split(column_name.lower()) if t]
    keyword_set = set(keywords)

    if any(token in keyword_set for token in tokens):
        return True

    # "_to_date" is a two-token pattern (e.g. "purchases_to_date") rather
    # than a single keyword -- check for "to" immediately followed by "date"
    return any(tokens[i] == "to" and tokens[i + 1] == "date" for i in range(len(tokens) - 1))


def compute_elapsed_time_correlation(
    feature: pd.Series, record_time: pd.Series, event_time: pd.Series
) -> Optional[float]:
    """Correlation between the feature's value and elapsed time (event_time
    minus record_time) -- Signal 2, only available when both timestamp
    columns are provided.

    A correctly-windowed feature's magnitude shouldn't systematically track
    how much history was available past the label event; a strong positive
    correlation here means the feature keeps growing the longer that gap
    is, a telltale sign it wasn't cut off at the event. Deliberately signed,
    not direction-agnostic like the predictiveness signal -- only a
    positive correlation tells this particular story, and the caller
    decides what counts as "strong" when combining signals into severity.

    Returns None when a correlation can't be meaningfully computed -- too
    few complete rows, a constant feature/elapsed-time with nothing to
    correlate, or either timestamp column containing values that can't be
    parsed as dates at all -- rather than fabricating a misleading number
    or letting a raw parsing exception escape to the caller.
    """
    # normalize everything to a plain Series with a fresh positional index --
    # pd.to_datetime() on a list/array returns a DatetimeIndex, not a Series
    # (no .dt accessor), and mismatched original indices between the three
    # inputs could otherwise silently misalign rows during construction
    feature = pd.Series(feature).reset_index(drop=True)
    try:
        record_time = pd.Series(pd.to_datetime(record_time)).reset_index(drop=True)
        event_time = pd.Series(pd.to_datetime(event_time)).reset_index(drop=True)
    except (ValueError, TypeError):
        # a column that isn't actually date-like (garbage strings, mixed
        # types) can't give this signal anything to correlate -- same
        # "unavailable" outcome as any other uncomputable case
        return None

    elapsed = (event_time - record_time).dt.total_seconds() / 86400
    aligned = pd.DataFrame({"feature": feature, "elapsed": elapsed}).dropna()

    if len(aligned) < MIN_ROWS_FOR_ELAPSED_CORRELATION:
        return None
    if aligned["feature"].nunique() < 2 or aligned["elapsed"].nunique() < 2:
        return None

    corr = aligned["feature"].corr(aligned["elapsed"])
    if pd.isna(corr):
        return None

    return float(corr)


def compute_predictiveness_signal(feature: pd.Series, target: pd.Series) -> Optional[PredictivenessResult]:
    """Signal 3: how predictive is this feature of the target, on its own?

    Thin wrapper around baseline.compute_predictiveness() -- the actual
    scoring logic is generic (reused later for basic feature-target
    leakage too, per the roadmap), this just names it as the temporal-
    leakage entry point so callers here don't need to know the shared
    helper lives in a different module.
    """
    return compute_predictiveness(feature, target)


def is_unusually_predictive(
    result: Optional[PredictivenessResult], threshold: float = PREDICTIVENESS_HIGH_THRESHOLD
) -> bool:
    """Does this predictiveness score clear the bar for "suspiciously high
    for a single feature"? None (score unavailable) is never suspicious --
    absence of evidence isn't evidence of a leak.
    """
    if result is None:
        return False
    return result.score >= threshold


def combine_temporal_signals(
    name_signal_matched: bool,
    elapsed_time_correlation: Optional[float],
    predictiveness: Optional[PredictivenessResult],
) -> Optional[str]:
    """Combine the three signals into an overall severity, or None if
    nothing about this feature is actually suspicious.

    None of the three signals alone is conclusive (a name match is
    extremely common on legitimate aggregates; a correlation or
    predictiveness score can't be computed at all without the right
    inputs), so severity is driven by how many independently fire:

    - 0 fired: not a finding at all -- nothing here is worth flagging
    - 1 fired: LOW
    - 2 fired: MEDIUM -- per the PRD, "two or more firing together is a
      strong flag"
    - 3 fired: HIGH

    A signal that couldn't be computed (elapsed_time_correlation or
    predictiveness is None) simply doesn't count toward the total --
    "unavailable" is never treated as "fired", regardless of whether
    that's because the right columns weren't provided or because it
    genuinely couldn't be computed from what was given.
    """
    fired_count = 0
    if name_signal_matched:
        fired_count += 1
    if elapsed_time_correlation is not None and elapsed_time_correlation > ELAPSED_CORRELATION_STRONG_THRESHOLD:
        fired_count += 1
    if is_unusually_predictive(predictiveness):
        fired_count += 1

    if fired_count == 0:
        return None
    if fired_count == 1:
        return "LOW"
    if fired_count == 2:
        return "MEDIUM"
    return "HIGH"


_SEVERITY_RANK = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}


def rank_temporal_findings(findings: List[TemporalLeakageFinding]) -> List[TemporalLeakageFinding]:
    """Sort temporal-leakage findings most-severe first.

    Severity is the primary key; ties broken by predictiveness score, then
    by elapsed-time correlation magnitude, so a finding with stronger raw
    evidence at the same severity level surfaces first even though severity
    itself is a coarse 3-level bucket.
    """
    return sorted(
        findings,
        key=lambda f: (
            _SEVERITY_RANK[f.severity],
            f.predictiveness_score if f.predictiveness_score is not None else 0.0,
            abs(f.elapsed_time_correlation) if f.elapsed_time_correlation is not None else 0.0,
        ),
        reverse=True,
    )


def check_temporal_leakage(
    df: pd.DataFrame,
    target_col: str,
    event_time_col: Optional[str] = None,
    record_time_col: Optional[str] = None,
) -> List[TemporalLeakageFinding]:
    """Scan every numeric feature column in df for signs of temporal
    leakage against target_col, combining all three signals per feature.

    Signal 2 (elapsed-time correlation) only runs when both event_time_col
    and record_time_col are given -- without them, every finding is marked
    reduced_confidence, weaker evidence from only the name and
    predictiveness signals. Never silently drops to the weaker check
    without saying so; that's what reduced_confidence is for.

    A feature only becomes a finding at all if at least one signal fires
    (see combine_temporal_signals) -- most columns in a normal dataset
    won't be flagged, and that's the expected, quiet outcome.
    """
    reduced_confidence = event_time_col is None or record_time_col is None
    target = df[target_col]

    excluded_cols = {target_col, event_time_col, record_time_col}
    findings = []

    for col in df.columns:
        if col in excluded_cols or not pd.api.types.is_numeric_dtype(df[col]):
            continue

        name_matched = detect_aggregation_name_signal(col)

        elapsed_corr = None
        if not reduced_confidence:
            elapsed_corr = compute_elapsed_time_correlation(df[col], df[record_time_col], df[event_time_col])

        predictiveness = compute_predictiveness_signal(df[col], target)

        severity = combine_temporal_signals(name_matched, elapsed_corr, predictiveness)
        if severity is None:
            continue

        findings.append(TemporalLeakageFinding(
            feature=col,
            name_signal_matched=name_matched,
            elapsed_time_correlation=elapsed_corr,
            predictiveness_score=predictiveness.score if predictiveness else None,
            predictiveness_metric=predictiveness.metric if predictiveness else None,
            severity=severity,
            reduced_confidence=reduced_confidence,
            event_time_col=event_time_col,
        ))

    return rank_temporal_findings(findings)
