# transcripts/transcript_writer.py

import os
from datetime import datetime
from typing import List, Dict


def write_session_transcript(qa_log: List[Dict], base_dir: str = "data/sessions") -> str:
    """
    Write a single session's Q&A log to a markdown file.

    Format:
    # Q&A Transcript

    ## Q1. <question text>

    - bullet 1
    - bullet 2

    ---
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(base_dir, ts)
    os.makedirs(session_dir, exist_ok=True)

    out_path = os.path.join(session_dir, "qa_log.md")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Q&A Transcript\n\n")
        for i, item in enumerate(qa_log, 1):
            q_text = item.get("q", "").strip()
            bullets = item.get("bullets", []) or []

            f.write(f"## Q{i}. {q_text}\n\n")
            for b in bullets:
                b = str(b).strip()
                if not b:
                    continue
                # you want just bullet points, no bold, no tags
                f.write(f"- {b}\n")
            f.write("\n---\n\n")

    return out_path
