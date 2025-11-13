# core/answer_suggester.py

import re
import yaml

class AnswerSuggester:

    def __init__(self, jd_path="data/jd.yaml", resume_path="data/resume_topics.yaml"):
        # Load JD-specific topics (if provided)
        try:
            self.jd_topics = yaml.safe_load(open(jd_path))
        except:
            self.jd_topics = {}

        # Load resume fixed topics
        self.resume_topics = yaml.safe_load(open(resume_path))

    def _score_topic(self, text, topic):
        score = 0
        for kw in topic.get("keywords", []):
            if kw in text:
                score += 1
        return score

    def suggest(self, question_text):
        """
        Returns a list of 3–5 bullet-sized answers
        aligned with the detected question.
        """
        q = question_text.lower()

        # 1) Score resume-based topics
        resume_scores = []
        for name, t in self.resume_topics.items():
            resume_scores.append((name, self._score_topic(q, t)))
        resume_scores.sort(key=lambda x: x[1], reverse=True)

        # 2) Score JD-based topics
        jd_scores = []
        for name, t in self.jd_topics.items():
            jd_scores.append((name, self._score_topic(q, t)))
        jd_scores.sort(key=lambda x: x[1], reverse=True)

        # pick top resume and top JD topics
        best_resume = resume_scores[0][0] if resume_scores[0][1] > 0 else None
        best_jd = jd_scores[0][0] if jd_scores and jd_scores[0][1] > 0 else None

        bullets = []

        if best_resume:
            bullets.extend(self.resume_topics[best_resume]["bullets"][:3])

        if best_jd:
            bullets.extend(self.jd_topics[best_jd]["bullets"][:2])

        # fallback if nothing matched
        if not bullets:
            bullets = [
                "I follow a structured MLOps approach: reproducibility, automation, and observability.",
                "I rely heavily on CI/CD, containerization, and IaC to ensure stable deployments.",
                "I incorporate monitoring and drift-detection to keep models reliable in production."
            ]

        return bullets
