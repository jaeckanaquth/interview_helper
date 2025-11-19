# core/state.py

from dataclasses import dataclass, field
from typing import Optional, List, Dict


@dataclass
class SessionState:
    """Holds per-interview session state (questions, behavioral context, etc.)."""
    last_question: Optional[str] = None
    last_intent: Optional[str] = None
    last_behavioral_project: Optional[Dict] = None
    last_behavioral_answer: List[str] = field(default_factory=list)
