# core/answer_llm.py

from openai import OpenAI
import yaml
import os

import dotenv
dotenv.load_dotenv(dotenv_path="config/.env")


def classify_question_intent(q: str) -> str:
    q = q.lower().strip()

    if any(p in q for p in ["tell me about yourself", "introduce yourself", "who are you"]):
        return "intro"

    if "what have you studied" in q or "education" in q or "background" in q:
        return "education"

    if "experience" in q and "what has been your" in q:
        return "experience"

    if "strength" in q:
        return "strengths"

    if "weakness" in q or "development areas" in q or "improvement areas" in q:
        return "weaknesses"

    # LLM basics
    if any(p in q for p in [
        "what is an llm",
        "what is a large language model",
        "how do llms work",
        "generative ai",
        "foundation model",
        "what is gpt",
        "what are transformers",
    ]):
        return "llm_basics"

    # ML pipeline / productionization
    if any(p in q for p in [
        "ml pipeline",
        "machine learning pipeline",
        "end to end pipeline",
        "end-to-end pipeline",
        "ml workflow",
        "machine learning workflow",
        "productionize",
        "productionalize",
        "deploy a model",
        "model deployment steps",
        "model training pipeline",
        "feature pipeline",
    ]):
        return "ml_pipeline"

    # Why this job / why this company / why here
    why_patterns = [
        "why do you want to work here",
        "why do you want to work for",
        "why are you interviewing with me today",
        "why are you interviewing with us",
        "what made you apply for this job",
        "what made you apply",
        "why this job",
        "why this company",
        "why do you want to join",
        "why do you want this role",
        "why are you interested in this position",
        "why should we hire you",               # close enough to motivation
        "why should we give this job to you",
        "why should we hire you and not someone else",
    ]
    if any(p in q for p in why_patterns):
        return "why_company"

    # Behavioral / STAR-style project questions
    behavioral_patterns = [
        "tell me about a time",
        "give me an example",
        "describe a time",
        "describe a situation",
        "situation where",
        "project where",
        "project when",
        "time when you",
        "time when you were",
        "handled a",
        "faced a",
        "dealt with",
        "how did you handle",
        "how did you meet the deadline",
        "with a deadline",
        "in charge of a project",
    ]
    if any(p in q for p in behavioral_patterns):
        return "behavioral_project"

    return "generic"

# ---------- Project loader + picker ----------

def _load_projects_from_yaml(path: str):
    """Load projects list from projects.yaml. Returns [] on failure."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if isinstance(data, list):
            return data
        else:
            return []
    except Exception:
        return []


def _classify_behavior_tags(question: str):
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

    # FALLBACK: generic behavioral project
    if not tags and any(w in q for w in ["project", "situation", "example", "experience"]):
        tags.append("ownership")
        tags.append("deadline")

    # Deduplicate, keep order
    seen = set()
    deduped = []
    for t in tags:
        if t not in seen:
            seen.add(t)
            deduped.append(t)
    return deduped


def _similarity(a: str, b: str) -> float:
    from difflib import SequenceMatcher
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _score_project_for_question(project: dict, behavior_tags, question: str) -> float:
    score = 0.0
    proj_tags = set(project.get("tags", []))

    # Tag overlap
    for t in behavior_tags:
        if t in proj_tags:
            score += 2.0

    text = (project.get("short_summary", "") + " " +
            project.get("impact_summary", "")).lower()

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

    # Tiny tie-breaker: text similarity
    base = project.get("name", "") + " " + project.get("short_summary", "")
    score += 0.3 * _similarity(question, base)

    return score


def pick_best_project(question: str, projects: list, default_id: str = "ua_devops_platform") -> dict:
    """Pick the best project for this question. Fallback to default_id."""
    if not projects:
        return {}

    behavior_tags = _classify_behavior_tags(question)
    scored = []

    for p in projects:
        s = _score_project_for_question(p, behavior_tags, question)
        scored.append((s, p))

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_project = scored[0]

    # If nothing matches well, use default UA DevOps project if present
    if best_score < 1.0:
        for p in projects:
            if p.get("id") == default_id:
                return p

    return best_project


def build_project_answer_prompt(question: str, project: dict) -> str:
    """Prompt for FIRST answer based on a chosen project."""
    name = project.get("name", "")
    role = project.get("role", "")
    company = project.get("company", "")
    short_summary = project.get("short_summary", "")
    impact_summary = project.get("impact_summary", "")

    return f"""
        You are helping me answer a behavioral interview question in first person as Anmol.

        Question:
        {question}

        Use exactly this project from my experience and do NOT switch to any other:
        - Project name: {name}
        - Role: {role}
        - Company: {company}
        - Project summary: {short_summary}
        - Project impact: {impact_summary}

        Instructions:
        - Build the answer internally using STAR structure (Situation, Task, Action, Result),
        but DO NOT write the words “Situation”, “Task”, “Action”, or “Result” anywhere.
        - Use 4–6 clean bullet points.
        - Each bullet should be a single, clear, spoken-interview-friendly sentence.
        - No headings, no labels, no bold markers.
        - Do NOT restate the question.
        - Do NOT invent any timeline, year, or date.
        - Do NOT invent details not in the project description.
    """


def build_behavioral_followup_prompt(question: str, project: dict, previous_answer_text: str | None) -> str:
    """Prompt for FOLLOW-UP questions on the same behavioral project."""
    name = project.get("name", "")
    role = project.get("role", "")
    company = project.get("company", "")

    return f"""
    You are answering a FOLLOW-UP behavioral interview question in first person as Anmol.

    Follow-up question:
    {question}

    This follow-up refers to the SAME incident described earlier. Do NOT change the story.

    Previous answer:
    {previous_answer_text}

    Project context:
    - Project name: {name}
    - Role: {role}
    - Company: {company}

    Instructions:
    - Answer ONLY what the follow-up is asking (no re-explaining the whole project).
    - Use 2–4 sharp, direct bullet points.
    - DO NOT introduce new problems, new incidents, or new timelines.
    - DO NOT restate the question.
    - DO NOT use the words “Situation”, “Task”, “Action”, or “Result”.
    - DO NOT invent years or dates.
    """


def _is_behavioral_followup(question: str, last_intent: str | None) -> bool:
    """Heuristics: detect short follow-up questions tied to a previous behavioral story."""
    if last_intent not in ("behavioral_project", "behavioral_followup"):
        return False

    q = question.lower().strip()

    # Very generic fragments / follow-up style wordings we saw in logs
    patterns = [
        "when was it",
        "which was the outcome",
        "what did you do",
        "how was it resolved",
        "what did it consist of",
        "and how was it resolved",
        "and which was the outcome",
        "okay, so tell me about the time",
        "tell me about the time",  # as standalone follow-up
        "and what did you do",
        "and how did you meet the deadline",
        "how did you meet the deadline",
    ]
    if any(p in q for p in patterns):
        return True

    # Short vague follow-ups with 'when' / 'outcome' / 'resolved'
    if len(q) < 60 and any(w in q for w in ["when", "outcome", "resolved"]):
        return True

    return False


# ---------- Answer Engine ----------

class AnswerEngine:
    def __init__(self, role: str, resume_path: str, jd_path: str):
        self.role = role

        # load resume + JD once
        try:
            self.resume_text = open(resume_path, "r", encoding="utf-8").read()
        except Exception:
            self.resume_text = ""

        try:
            self.jd_text = open(jd_path, "r", encoding="utf-8").read()
        except Exception:
            self.jd_text = ""

        # load projects.yaml once
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        projects_path = os.path.join(base_dir, "data", "projects.yaml")
        self.projects = _load_projects_from_yaml(projects_path)

        # init OpenAI client
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # simple session state for follow-ups
        self.last_question: str | None = None
        self.last_intent: str | None = None
        self.last_behavioral_project: dict | None = None
        self.last_behavioral_answer: list[str] | None = None

        # system prompt WITHOUT JD: default for most questions
        self.system_prompt_general = f"""
            You are helping Anmol in a live MLOps/DevOps interview.
            Answer ONLY in 3–6 sharp bullet points.
            Answer in first person, as Anmol.

            Use ONLY information that is consistent with the resume below.
            If something is not clearly supported by it, stay generic
            instead of inventing tools, services or companies.

            ROLE:
            {self.role}

            RESUME:
            {self.resume_text}

            Rules:
            - Do NOT restate the question.
            - Do NOT invent random tech I haven't actually used.
            - Prefer AWS, Terraform, Kubernetes, CI/CD, monitoring, drift detection,
              Airflow, PySpark, etc. ONLY if they appear in my background.
            - For 'tell me about yourself' / 'what have you studied' questions,
              include education and key experience.
            - Keep bullets short (ideally 12–20 words).
        """

        # system prompt WITH JD: only for why-company / motivation style questions
        self.system_prompt_with_jd = f"""
            You are helping Anmol in a live MLOps/DevOps interview.
            Answer ONLY in 3–6 sharp bullet points.
            Answer in first person, as Anmol.

            Use ONLY information that is consistent with the resume and job description below.
            If something is not clearly supported by them, stay generic
            instead of inventing tools, services or companies.

            ROLE:
            {self.role}

            RESUME:
            {self.resume_text}

            JOB DESCRIPTION:
            {self.jd_text}

            Rules:
            - Do NOT restate the question.
            - Do NOT invent random tech I haven't actually used.
            - You MAY refer to the company’s domain, products, or responsibilities
              only when explaining why I am a good fit for THIS ROLE or why I want
              to work at THIS COMPANY.
            - Keep bullets short (ideally 12–20 words).
        """



    def generate_answer(self, question: str):
        q = question.strip()
        base_intent = classify_question_intent(q)

        # Decide if this is a behavioral follow-up
        intent = base_intent
        is_followup = _is_behavioral_followup(q, self.last_intent)

        # Weakness/development follow-ups: keep them in weaknesses mode
        if intent == "generic" and self.last_intent == "weaknesses":
            q_low = q.lower()
            if any(p in q_low for p in [
                "what do you have identified",
                "identified as",
                "those areas",
                "these areas",
                "what have you done",
                "done to improve",
                "improve them",
                "to improve them so far",
                "how have you worked on them",
            ]):
                intent = "weaknesses"


        # For behavioral follow-ups, force a special path
        if is_followup and self.projects:
            intent = "behavioral_followup"

        # Build user_msg based on intent
        project = None
        user_msg = ""

        if intent == "behavioral_project" and self.projects:
            project = pick_best_project(q, self.projects)
            user_msg = build_project_answer_prompt(q, project)

        elif intent == "behavioral_followup" and self.projects:
            # Reuse last project if available, otherwise fall back
            project = self.last_behavioral_project or pick_best_project(q, self.projects)
            prev_text = ""
            if self.last_behavioral_answer:
                prev_text = "\n".join(f"- {b}" for b in self.last_behavioral_answer)
            user_msg = build_behavioral_followup_prompt(q, project, prev_text)
            

        else:
            base = f"Question: {q}\n"

            if intent == "intro":
                user_msg = base + (
                "Answer as a short professional self-introduction: 4–5 bullets.\n"
                "Format: **ShortTitle:** description.\n"
                "- Current role & years.\n"
                "- Core tech stack.\n"
                "- Education.\n"
                "- Value you bring.\n"
            )


            elif intent == "education":
                user_msg = base + (
                "Focus on education in 3–4 bullets.\n"
                "Format: **ShortTitle:** description.\n"
                "- Degrees.\n"
                "- Key subjects.\n"
                "- Certifications.\n"
            )


            elif intent == "experience":
                user_msg = base + (
                    "Summarise your experience in 3–5 bullets.\n"
                    "Format: **ShortTitle:** description.\n"
                )

            elif intent == "strengths":
                user_msg = base + (
                    "List 3–5 strengths.\n"
                    "Format: **StrengthName:** how it helps the role.\n"
                )

            elif intent == "weaknesses":
                user_msg = base + (
                    "List real but safe development areas.\n"
                    "Format: **WeaknessName:** what you’re doing to improve it.\n"
                )



            elif intent == "why_company":
                user_msg = base + (
                    "Answer why you applied for this role and why you want to work at this company.\n"
                    "Use the job description context above for alignment, but do NOT copy sentences.\n"
                    "Format: **Stage:** explanation.\n"
                    "- 1 bullet: what attracts you to the company/domain or team (based on JD).\n"
                    "- 2 bullets: how your past work (DevOps/MLOps/platform) matches what they need.\n"
                    "- 1 bullet: what value you will bring (impact, reliability, scalability, cost, etc.).\n"
                    "- 1 bullet: what you want to learn or grow into in this role.\n"
                    "Keep it specific to this company/role, not generic MLOps boilerplate."
                )

            elif intent == "llm_basics":
                user_msg = base + (
                    "Explain what a Large Language Model (LLM) is.\n"
                    "Give 3–5 bullets.\n"
                    "Format: **Stage:** description.\n"
                    "- 1 bullet: simple definition.\n"
                    "- 1 bullet: how transformers/attention work (high-level).\n"
                    "- 1 bullet: why LLMs are useful for automation / NLP.\n"
                    "- 1 bullet: tie to MLOps or deployment if relevant.\n"
                )

            elif intent == "ml_pipeline":
                user_msg = base + (
                    "Explain an end-to-end ML pipeline in 3–6 clear bullets.\n"
                    "Structure should cover:\n"
                    "Format: **KeyPoint:** description.\n"
                    "- Data ingestion + preprocessing.\n"
                    "- Feature engineering.\n"
                    "- Model training + evaluation.\n"
                    "- CI/CD for ML (model versioning, deployment, testing).\n"
                    "- Monitoring + retraining strategy.\n"
                    "Use AWS/Terraform/Kubernetes examples ONLY if consistent with my resume."
                )

            elif intent == "drift":  # if you fully removed drift, delete this branch entirely
                user_msg = base + (
                    "Format: **KeyPoint:** description.\n"
                    "Explain clearly how you detect and handle data/model drift in production.\n"
                    "3–5 bullets, concrete techniques and tools."
                )

            elif intent == "ml_basics":  # if you kept the old 'ml_basics' name separately
                user_msg = base + (
                    "Format: **KeyPoint:** description.\n"
                    "Explain 'what is machine learning and why is it important'.\n"
                    "Give 3–5 bullets, high level theory plus one MLOps angle."
                )

            else:
                # Generic fallback: answer the question directly, no boilerplate
                user_msg = base + (
                    "Format: **KeyPoint:** description.\n"
                    "Answer this question directly in 3–5 short bullet points.\n"
                    "- Focus only on what the question is actually asking.\n"
                    "- Do NOT just list generic responsibilities or MLOps buzzwords.\n"
                    "- Use concrete examples from your experience only if they clearly fit the question.\n"
                    "- If the question is vague, give a reasonable, honest interpretation."
                )



        try:
            # choose system prompt based on intent
            if intent == "why_company":
                system_prompt = self.system_prompt_with_jd
            else:
                system_prompt = self.system_prompt_general

            resp = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=220,
                temperature=0.3,
            )

            text = resp.choices[0].message.content.strip()

            bullets = []
            for line in text.split("\n"):
                line = line.strip()
                if line.startswith(("-", "•", "*")):
                    bullets.append(line.lstrip("-•* ").strip())

            if not bullets:
                bullets = [text]

            # ---- Update session state for next question ----
            self.last_question = q
            self.last_intent = intent

            if intent in ("behavioral_project", "behavioral_followup"):
                # Store the project used for this behavioral chain
                if self.last_intent == "behavioral_project" and self.last_project is not None:
                    # This is a follow-up to the same project
                    last_answer_text = "\n".join(self.last_answer_bullets)
                    user_msg = build_behavioral_followup_prompt(q, self.last_project, last_answer_text)
                else:
                    # first behavioral question
                    project = pick_best_project(q, self.projects)
                    self.last_project = project
                    user_msg = build_project_answer_prompt(q, project)
                self.last_behavioral_project = project
                self.last_behavioral_answer = bullets
            

            return bullets

        except Exception as e:
            return [f"(LLM error: {e})"]
