# tools/build_answer_bank.py

import os
import glob
import json
from typing import List, Dict


def parse_qa_markdown(path: str) -> List[Dict]:
    """
    Parse a qa_log.md file in the format:

    ## Q1. Question text

    - bullet 1
    - bullet 2

    ---
    """
    qa_pairs: List[Dict] = []

    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]

    current_q = None
    current_bullets: List[str] = []

    def flush():
        nonlocal current_q, current_bullets
        if current_q and current_bullets:
            qa_pairs.append({
                "question": current_q.strip(),
                "bullets": [b.strip() for b in current_bullets if b.strip()],
                "source_file": path,
            })
        current_q = None
        current_bullets = []

    for line in lines:
        line_stripped = line.strip()

        # Question header
        if line_stripped.startswith("## "):
            # flush previous Q/A
            flush()

            header = line_stripped[3:].strip()  # drop "## "
            # expected forms:
            # Q1. Question text
            # Q1: Question text
            # Q1 Question text
            parts = header.split(maxsplit=1)
            if len(parts) == 2 and parts[0].lower().startswith("q"):
                # remove possible trailing '.' or ':' from the Q token
                q_text = parts[1].lstrip(".:").strip()
                current_q = q_text
                current_bullets = []
            else:
                current_q = header
                current_bullets = []
            continue

        # separator
        if line_stripped.startswith("---"):
            flush()
            continue

        # bullet
        if line_stripped.startswith("- "):
            bullet_text = line_stripped[2:].strip()
            if current_q is not None:
                current_bullets.append(bullet_text)
            continue

        # other lines are ignored (blank or headings, etc.)

    # final flush
    flush()
    return qa_pairs


def main():
    base_dir = os.path.join("data", "sessions")
    pattern = os.path.join(base_dir, "*", "qa_log.md")
    paths = glob.glob(pattern)

    all_pairs: List[Dict] = []
    print(f"[INFO] Found {len(paths)} qa_log.md files")
    for p in sorted(paths):
        all_pairs.extend(parse_qa_markdown(p))
        print(f"[INFO] Parsed {len(all_pairs)} Q&A pairs from {p}")
    out_path = os.path.join("data", "answer_bank.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for qa in all_pairs:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")

    print(f"[INFO] Extracted {len(all_pairs)} Q&A pairs into {out_path}")


if __name__ == "__main__":
    main()
