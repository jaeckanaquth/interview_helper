# core/project.py

import os
import yaml
from difflib import SequenceMatcher

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PROJECTS_PATH = os.path.join(HERE, "..", "data", "projects.yaml")


def load_projects(path: str = DEFAULT_PROJECTS_PATH):
    """Load projects from YAML into a list of dicts."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, list):
        raise ValueError("projects.yaml must contain a list of projects")
    return data


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def classify_behavior_tags(question: str) -> list:
    """Crude classifier: map question text to behavioral tags."""
    q = question.lower()
    tags = []

    # DEADLINE / PRESSURE / OWNERSHIP
    if any(w in q for w in ["deadline", "time pressure", "tight schedule", "deliver on time"]):
        tags.append("deadline")
    if any(w in q for w in ["in charge", "led", "leadership", "owned", "owner", "drove"]):
        tags.append("leadership")
        tags.append("ownership")

    # COST / OPTIMIZATION / EFFICIENCY
    if any(w in q for w in ["cost", "budget", "optimiz", "expense", "saving", "save money"]):
        tags.append("cost")
        tags.append("analysis")

    # RELIABILITY / INCIDENT / DR
    if any(w in q for w in ["incident", "outage", "downtime", "reliability", "dr", "disaster"]):
        tags.append("incident")
        tags.append("reliability")
        tags.append("risk")

    # COMPLIANCE / SECURITY
    if any(w in q for w in ["compliance", "audit", "soc2", "security", "controls", "governance"]):
        tags.append("compliance")
        tags.append("audit")
        tags.append("security")

    # MLOps / DATA / REALTIME
    if any(w in q for w in ["mlops", "model", "training", "deploying models", "prediction"]):
        tags.append("mlops")
    if any(w in q for w in ["real-time", "realtime", "stream", "sensor", "modbus", "iot"]):
        tags.append("realtime")
        tags.append("sensor_data")
    if any(w in q for w in ["pipeline", "data", "ingestion", "processing"]):
        tags.append("data_ingestion")
        tags.append("data_processing")

    # FALLBACK: if it's clearly generic “tell me about a project”
    if not tags and any(w in q for w in ["project", "situation", "example", "experience"]):
        tags.append("ownership")
        tags.append("deadline")

    return list(dict.fromkeys(tags))  # deduplicate, keep order


def _score_project_for_question(project: dict, behavior_tags: list, question: str) -> float:
    score = 0.0
    proj_tags = set(project.get("tags", []))

    # Tag overlap
    for t in behavior_tags:
        if t in proj_tags:
            score += 2.0

    # Boost for certain words in summaries (deadline-ish, etc.)
    text = (project.get("short_summary", "") + " " + project.get("impact_summary", "")).lower()

    if "deadline" in behavior_tags:
        if any(w in text for w in ["deadline", "on time", "time", "schedule"]):
            score += 1.0

    if "cost" in behavior_tags:
        if any(w in text for w in ["cost", "spend", "billing", "save", "optimiz"]):
            score += 1.0

    if "incident" in behavior_tags or "reliability" in behavior_tags:
        if any(w in text for w in ["outage", "dr", "backup", "recovery", "resilience"]):
            score += 1.0

    if "mlops" in behavior_tags:
        if "mlops" in text or "model" in text or "sagemaker" in text:
            score += 1.0

    # Tiny tie-breaker: similarity between question and project name/summary
    base = project.get("name", "") + " " + project.get("short_summary", "")
    score += 0.3 * _similarity(question, base)

    return score


def pick_best_project(question: str, projects: list, default_id: str = "ua_devops_platform") -> dict:
    """Pick the best project to answer this question.

    If nothing matches well, fallback to default_id.
    """
    if not projects:
        raise ValueError("No projects loaded")

    behavior_tags = classify_behavior_tags(question)
    scored = []

    for p in projects:
        s = _score_project_for_question(p, behavior_tags, question)
        scored.append((s, p))

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_project = scored[0]

    # If score is very low, force default
    if best_score < 1.0:
        for p in projects:
            if p.get("id") == default_id:
                return p

    return best_project
