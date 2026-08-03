from dataclasses import dataclass


@dataclass
class Finding:
    column: str
    issue_type: str
    confidence: float
    detail: str
    action_taken: str
