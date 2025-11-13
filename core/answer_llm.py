# core/answer_llm.py

from openai import OpenAI
import yaml
import os

import dotenv
dotenv.load_dotenv(dotenv_path="config/.env")

def classify_question_intent(q: str) -> str:
    q = q.lower()

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

    if "data drift" in q or "concept drift" in q or "detect drift" in q:
        return "drift"

    if "what is machine learning" in q:
        return "ml_basics"

    return "generic"

class AnswerEngine:
    def __init__(self, role: str, resume_path: str, jd_path: str):
        self.role = role

        # load resume + JD once
        try:
            self.resume_text = open(resume_path, "r", encoding="utf-8").read()
        except:
            self.resume_text = ""

        try:
            self.jd_text = open(jd_path, "r", encoding="utf-8").read()
        except:
            self.jd_text = ""

        # init OpenAI client
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # system prompt for consistent behaviour
        self.system_prompt = f"""
            You are helping Anmol in a live MLOps interview.
            Answer ONLY in 3–5 sharp bullet points.
            Answer in first person, as Anmol.

            Use ONLY information that is consistent with the resume and JD below.
            If something is not clearly supported by them, stay generic instead of inventing tools, services or companies.

            ROLE:
            {self.role}

            RESUME:
            {self.resume_text}

            JOB DESCRIPTION:
            {self.jd_text}

            Rules:
            - Do NOT restate the question.
            - Do NOT invent random tech I haven't actually used.
            - Prefer AWS, Terraform, Kubernetes, CI/CD, monitoring, drift detection, Airflow, PySpark, etc. ONLY if they appear in my background.
            - For 'tell me about yourself' / 'what have you studied' questions, include education and key experience.
            - Keep bullets short (ideally 12–20 words).
        """




    def generate_answer(self, question: str):
        q = question.strip()
        intent = classify_question_intent(q)
        base = f"Question: {q}\n"

        if intent == "intro":
            user_msg = base + (
                "Answer as a short professional self-introduction: 4–5 bullets.\n"
                "- 1 bullet: current role and years of experience.\n"
                "- 1–2 bullets: core skills relevant to this JD.\n"
                "- 1 bullet: education/degree.\n"
                "- 1 bullet: what value I bring.\n"
            )

        elif intent == "education":
            user_msg = base + (
                "Focus on education and key technical areas you have studied.\n"
                "Mention degrees, majors and 1–2 relevant certifications.\n"
                "Keep 3–4 bullets, concise."
            )

        elif intent == "experience":
            user_msg = base + (
                "Summarise your experience in 3–5 bullets.\n"
                "Cover domains, main responsibilities, and impact (MLOps/DevOps/data)."
            )

        elif intent == "strengths":
            user_msg = base + (
                "Answer: 'What are your biggest strengths and how would our company benefit?'\n"
                "Give 3–5 strengths, each tied to MLOps/platform work and value for the company."
            )

        elif intent == "weaknesses":
            user_msg = base + (
                "Answer about development/improvement areas.\n"
                "Give 2–3 real, safe weaknesses and 2–3 bullets on how you're improving them."
            )

        elif intent == "drift":
            user_msg = base + (
                "Explain clearly how you detect and handle data/model drift in production.\n"
                "3–5 bullets, concrete techniques and tools."
            )

        elif intent == "ml_basics":
            user_msg = base + (
                "Explain 'what is machine learning and why is it important'.\n"
                "Give 3–5 bullets, high level theory plus one MLOps angle."
            )

        else:
            user_msg = base + (
                "Provide 3–5 concise, technical bullets rooted in your real experience.\n"
                "Prioritize MLOps, ML platforms, DevOps and production systems."
            )

        try:
            resp = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self.system_prompt},
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

            return bullets if bullets else [text]

        except Exception as e:
            return [f"(LLM error: {e})"]


