# core/answer_retriever.py

import os
import json
from typing import List, Dict, Optional, Tuple
from difflib import SequenceMatcher


class AnswerRetriever:
    def __init__(self, path: str):
        self.path = path
        self.qa_list: List[Dict] = []
        self._load()

    def _load(self):
        if not os.path.exists(self.path):
            print(f"[INFO] No answer_bank found at {self.path}, history reuse disabled.")
            self.qa_list = []
            return

        qa_list: List[Dict] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if "question" in obj and "bullets" in obj:
                        qa_list.append(obj)
                except json.JSONDecodeError:
                    continue

        self.qa_list = qa_list
        print(f"[INFO] Loaded {len(self.qa_list)} historical Q&A entries from {self.path}")

    @staticmethod
    def _similarity(a: str, b: str) -> float:
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def find_best(
        self, question: str, threshold: float = 0.8
    ) -> Optional[Tuple[List[str], str, float]]:
        """Return (bullets, matched_question, score) or None if no good match."""
        if not self.qa_list:
            return None

        best = None
        best_score = 0.0
        q = question.strip()

        for entry in self.qa_list:
            stored_q = entry.get("question", "")
            score = self._similarity(q, stored_q)
            if score > best_score:
                best_score = score
                best = entry

        if best is None or best_score < threshold:
            return None

        bullets = best.get("bullets", [])
        return bullets, best.get("question", ""), best_score
