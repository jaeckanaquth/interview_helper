# 🔧 Interview Helper — Phase 1

**Real-time Interview Assistant with Live Transcription, Question Detection & GPT-4o-mini Answers**

Interview Helper is a Python-based tool designed to assist during **live technical interviews**.
It listens to system audio (WASAPI loopback), transcribes the conversation using **Whisper**, detects questions in real-time, and instantly generates **bullet-point answers** tailored to:

* Your **resume**
* The **current job description**
* The **role you are interviewing for**

Phase 1 focuses on the **core pipeline**, delivering stable, accurate, real-time Q&A assistance.

---

## ✨ Features (Phase 1)

### 🎤 1. Real-Time System Audio Capture

* Captures **desktop audio only** (no mic required)
* Uses **WASAPI loopback** → works with Zoom, Meet, Teams, YouTube practice videos
* Highly stable, low-latency stream

### 🧠 2. Whisper-Based Streaming STT

* Uses **faster-whisper**
* Sliding window transcription (10s window / 3s step)
* VAD gating
* Duplicate suppression
* Output is fast and high-quality

### ❓ 3. Smart Question Detection

* Extracts the **real core question** the interviewer asks
* Removes filler, boilerplate (“let’s begin with our first question…”)
* Deduplicates variations of the same question
* Filters out explanation statements (“X is part of Y, right?”)

### 🤖 4. GPT-4o-mini Answer Engine

* Answers every detected question in **3–5 sharp bullet points**
* Uses:

  * Your **resume.md**
  * The **current_jd.md**
  * The **role** you're applying for
* Zero hallucinations thanks to strict system prompt constraints
* Intent-aware responses:

  * “Tell me about yourself”
  * “What have you studied?”
  * “What is machine learning?”
  * “Strengths/weaknesses”
  * “Experience”
  * “Data drift”
  * And generic MLOps/DevOps questions

### 📝 5. Automatic Q&A Logging

Every real question + generated answer is saved to:

```
/data/sessions/<timestamp>/qa_log.md
```

Useful for review, reflection, and improving your preparation.

---

# 📁 Project Structure

```
interview_helper/
│
├── main.py                          # Main live pipeline
├── config/
│   └── settings.yaml                # Audio + streaming configuration
│
├── core/
│   ├── audio_capture.py             # WASAPI loopback capture
│   ├── stt_whisper_stream.py        # Whisper streaming pipeline
│   ├── question_finder.py           # Smart question extraction & dedupe
│   ├── answer_llm.py                # GPT-4o-mini answer engine
│   └── __init__.py
│
├── data/
│   ├── resume.md                    # Summary of your resume
│   ├── current_jd.md                # JD for *this* interview (editable)
│   └── sessions/                    # STT & Q&A logs
│
└── README.md
```

---

# 🚀 Getting Started

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

(If you don’t have one yet, create it from your current environment.)

Ensure you also install:

```bash
pip install openai pyaudio numpy scipy faster-whisper pyyaml
```

## 2. Set your OpenAI API key

```
export OPENAI_API_KEY="your-key"
```

or on Windows:

```
setx OPENAI_API_KEY "your-key"
```

## 3. Edit your resume and JD summaries

`data/resume.md` — short 1–2 paragraph summary of your real resume
`data/current_jd.md` — paste/summarize the job description for the next call

These heavily shape your interview answers.

## 4. Run the system

```bash
python main.py
```

You should see:

```
[INFO] Capturing system audio...
[INFO] Starting live transcription + question finder...
🗣️ <transcribed audio>
❓ Q: <detected question>
➡ bullet answer 1
➡ bullet answer 2
...
```

Press **Ctrl+C** to stop.

Your transcript and Q&A log will save automatically.

---

# 🧩 How It Works (Core Pipeline)

```
Desktop audio → WASAPI loopback → Whisper stream → 
Sliding window → Question Finder → GPT-4o-mini →
Bullet-point answer → Console + Q&A log
```

## Whisper Sliding Window

* 10s context
* 3s step
* Keeps Whisper accurate while staying near-real-time

## Question Finder

* Extracts the real core question
* Filters duplicates + explanations
* Normalizes phrasing
* Only fires when an **actual** question is detected

## GPT-4o-mini Answer Engine

* Uses role + resume + JD
* Intent-aware prompts
* 3–5 crisp bullets
* Zero rambling, zero paragraphs

---

# 🗒️ Output Example

```
❓ Q: tell me about yourself

➡ I’m a Senior MLOps/DevOps engineer with experience across AWS and GCP.
➡ I built end-to-end MLOps platforms for data ingestion, model training and deployment.
➡ Strong in Terraform, Kubernetes, CI/CD and cloud security.
➡ Experienced with predictive maintenance, document extraction, and ML observability.
➡ I focus on productionizing ML systems at scale.
```

---

# 🎯 Phase 1 Complete

Your system is now a functional **real-time interview assistant**:

* Live transcript
* Live question detection
* Live answer generation (LLM)
* Session logging
* Resume/JD-aware reasoning

Excellent foundation.

---

# ▶️ Phase 2 Preview (coming next)

You asked for:

1. **Auto-merged resume/JD summaries**
2. **Context memory of previous Q&A**
3. **A real UI overlay (always-on-top window)**
4. **End-of-session transcript export** (already added)

Phase 2 will add these without changing your Phase 1 pipeline.

---

If you'd like, I can generate:

* A **requirements.txt**
* A **sample JD summary**
* A **sample resume summary**
* Or even a GIF-style step-by-step demo sequence for the README.

Just say the word.
