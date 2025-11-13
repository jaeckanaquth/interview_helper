# core/question_finder.py

import re
from typing import List

QUESTION_WORDS = {
    "who", "what", "when", "where", "why", "how",
    "can", "could", "would",
    "do", "does", "did",
    "are", "is", "was", "were",
    "have", "has", "had",
    "should", "shall", "may", "might",
}
# (notice: no "will")


QUESTION_PHRASES = [
    "tell me about",
    "walk me through",
    "can you explain",
    "could you explain",
    "would you explain",
    "could you walk me through",
    "how would you",
    "how did you",
    "what did you do",
    "what would you do",
    "how do you handle",
    "how did you handle",
    "give me an example",
    "could you give me an example",
    "explain a time when",
    "explain the time when",
]

# filler at the beginning we want to strip for normalization
LEADING_FILLERS = [
    "so", "okay", "ok", "right", "well", "alright", "yeah", "look",
    "so uh", "so um", "so like",
]


class QuestionFinder:
    """
    Stateful question detector over a rolling transcript.

    Usage:
        qf = QuestionFinder()
        new_questions = qf.process(new_text_chunk)
    """

    def __init__(self, buffer_limit: int = 4000):
        self.buffer_limit = buffer_limit
        self.text_tail = ""          # rolling text buffer
        self.seen_questions = set()  # normalized strings

    def _normalize_question(self, s: str) -> str:
        s = s.strip()

        # remove trailing punctuation
        s = re.sub(r"[?!.\s]+$", "", s)
        s = s.lower()

        # strip “let's start with our first question…” boilerplate
        boilerplate_patterns = [
            r"^so\s+let'?s\s+begin\s+with\s+our\s+very\s+first\s+question\s+which\s+is\s+",
            r"^so\s+let'?s\s+begin\s+with\s+our\s+first\s+question\s+which\s+is\s+",
            r"^so\s+let'?s\s+begin\s+with\s+the\s+first\s+question\s+which\s+is\s+",
            r"^let'?s\s+begin\s+with\s+our\s+very\s+first\s+question\s+which\s+is\s+",
            r"^let'?s\s+begin\s+with\s+the\s+first\s+question\s+which\s+is\s+",
            r"^let'?s\s+start\s+with\s+the\s+tell\s+me\s+about\s+yourself\s+question\s*",
        ]
        for pat in boilerplate_patterns:
            s = re.sub(pat, "", s)

        # strip leading fillers like "so", "okay", etc.
        for filler in LEADING_FILLERS:
            pattern = r"^" + re.escape(filler) + r"\s+"
            s = re.sub(pattern, "", s)

        # keep from the first question word onwards
        tokens = s.split()
        if not tokens:
            return ""
        first_q_idx = None
        for i, t in enumerate(tokens):
            if t in QUESTION_WORDS:
                first_q_idx = i
                break
        if first_q_idx is not None:
            tokens = tokens[first_q_idx:]
        s = " ".join(tokens)

        # collapse whitespace
        s = re.sub(r"\s+", " ", s)
        return s.strip()


    def _looks_like_question(self, s: str) -> bool:
        if not s:
            return False

        raw = s.strip()
        lower = raw.lower()

        # obvious: explicit '?'
        if "?" in raw:
            return True

        # interrogative phrases
        for phrase in QUESTION_PHRASES:
            if phrase in lower:
                return True

        # first token heuristic
        tokens = lower.split()
        if not tokens:
            return False

        first = tokens[0]
        if first in QUESTION_WORDS:
            return True

        # heuristic: ends with "right", "correct", etc. AND contains "you"
        if any(lower.endswith(end) for end in [" right", " correct", " yeah", " okay"]):
            if "you" in tokens or "your" in tokens:
                return False

        # filter explanatory "X is Y, right?" with no 'you/your'
        if lower.endswith(" right") and "you" not in tokens and "your" not in tokens:
            return False


        return False

    def _split_sentences(self, text: str) -> List[str]:
        # split on ., ?, ! but keep text reasonably intact
        parts = re.split(r"(?<=[.?!])\s+", text)
        return [p.strip() for p in parts if p.strip()]

    def process(self, new_text: str) -> List[str]:
        """
        Feed new transcript text. Returns a list of *new* questions detected.
        """
        if not new_text:
            return []

        # extend rolling buffer
        self.text_tail += " " + new_text
        if len(self.text_tail) > self.buffer_limit:
            self.text_tail = self.text_tail[-self.buffer_limit:]

        candidates = self._split_sentences(self.text_tail)
        new_questions: List[str] = []

        for cand in candidates:
            if len(cand.split()) < 4:
                # too short to be a real interview question (most of the time)
                continue

            if not self._looks_like_question(cand):
                continue


            norm = self._normalize_question(cand)
            if not norm:
                continue

            word_count = len(norm.split())
            # require at least 5 words (avoids "What have you seen?")
            if word_count < 5 or word_count > 40:
                continue

            # aggressive dedupe: substring / superstring
            duplicate = False
            for seen in self.seen_questions:
                if norm in seen or seen in norm:
                    duplicate = True
                    break
            if duplicate:
                continue

            self.seen_questions.add(norm)
            new_questions.append(cand.strip())

            # length filter
            word_count = len(norm.split())
            if word_count < 4 or word_count > 40:
                continue

            if norm in self.seen_questions:
                continue

            self.seen_questions.add(norm)
            new_questions.append(cand.strip())

        return new_questions
