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

LEADING_FILLERS = [
    "so", "okay", "ok", "right", "well", "alright", "yeah", "look",
    "so uh", "so um", "so like", "um", "uh",
]

# Meeting noise and obvious non-interview chatter to ignore
IGNORED_PATTERNS = [
    r"\bgive me a minute\b",
    r"\bjoin in a minute\b",
    r"\bplease show me your id\b",
    r"\bshow me your id\b",
    r"\bshow me your id proof\b",
    r"\bcan you show me your id\b",
    r"\b(on mute|you are on mute)\b",
    r"\bwill be joining\b",
    r"\bfollow question\b",
    r"\bfollow up\b",
    r"\bthanks\b",
    r"\bthank you\b",
    r"\bwe will be\b",
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

        # strip common boilerplate
        boilerplate_patterns = [
            r"^so\s+let'?s\s+begin\s+with\s+our\s+very\s+first\s+question\s+which\s+is\s+",
            r"^so\s+let'?s\s+begin\s+with\s+our\s+first\s+question\s+which\s+is\s+",
            r"^let'?s\s+begin\s+with\s+our\s+very\s+first\s+question\s+which\s+is\s+",
            r"^let'?s\s+start\s+with\s+the\s+tell\s+me\s+about\s+yourself\s+question\s*",
        ]
        for pat in boilerplate_patterns:
            s = re.sub(pat, "", s)

        # strip leading fillers like "so", "okay", etc.
        for filler in LEADING_FILLERS:
            pattern = r"^" + re.escape(filler) + r"[\s,]+"
            s = re.sub(pattern, "", s)

        # collapse repeated filler tokens and repeated words
        s = re.sub(r'\b(\w+)( \1){2,}\b', r'\1', s)

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

        # ignore obvious meeting chatter / commands
        for pat in IGNORED_PATTERNS:
            if re.search(pat, lower):
                return False

        # explicit '?'
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
            # require at least a verb or a question word + noun to avoid short fragments
            if len(tokens) >= 5:
                return True
            # short questions like "Can you explain X?" are OK if they contain a verb after the first token
            if len(tokens) >= 3 and any(t in tokens for t in ["explain", "describe", "show", "tell", "walk", "do", "did", "have", "has"]):
                return True

        # filter explanatory "X is Y, right?" with no 'you/your' (likely not an interview question)
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
            # skip tiny fragments
            if len(cand.split()) < 4:
                continue

            if not self._looks_like_question(cand):
                continue

            norm = self._normalize_question(cand)
            if not norm:
                continue

            word_count = len(norm.split())
            # require at least 5 words, at most 50
            if word_count < 5 or word_count > 50:
                continue

            # dedupe: substring / superstring similarity on normalized form
            duplicate = False
            for seen in self.seen_questions:
                if norm in seen or seen in norm:
                    duplicate = True
                    break
            if duplicate:
                continue

            self.seen_questions.add(norm)
            new_questions.append(cand.strip())

        return new_questions
