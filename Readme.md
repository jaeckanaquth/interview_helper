### Interview Helper

Real-time interview assistant that:
- Captures system audio
- Transcribes speech with Whisper (sliding window)
- Detects actual interview questions
- Generates concise, resume/JD-aware bullet answers via OpenAI
- Logs Q&A per session


### Key Features
- Real-time WASAPI loopback capture (no mic required)
- Streaming STT with faster-whisper (10s window / 3s step, VAD, dedupe)
- Smart question extraction and deduplication
- Intent-aware answer engine (intro, education, strengths/weaknesses, experience, ML/LLM basics, ML pipeline, why-company, behavioral + follow-ups)
- Session logging to `data/sessions/<timestamp>/qa_log.md`


### Project Structure
```
interview_helper/
├─ main.py                       # Live pipeline: audio → STT → question finder → LLM → log
├─ config/
│  ├─ settings.yaml              # Audio + streaming params
│  └─ .env                       # OPENAI_API_KEY (create this)
├─ core/
│  ├─ audio_capture.py           # System audio capture (WASAPI loopback)
│  ├─ stt_whisper_stream.py      # Faster-Whisper streaming
│  ├─ question_finder.py         # Extracts real questions
│  └─ answer_llm.py              # OpenAI-driven answer engine
├─ data/
│  ├─ resume.md                  # Your resume summary (edit)
│  ├─ current_jd.md              # Current job description summary (edit)
│  └─ sessions/                  # Auto-saved logs
└─ documents/
   └─ Phase 1.md                 # Detailed design/notes
```


### Prerequisites
- Python 3.10+ recommended
- Windows (WASAPI loopback). Works best with Zoom/Meet/Teams playback.
- OpenAI API key


### Installation
1) Create/activate your environment (as per your workflow):

```bash
conda activate snow
```

2) Install dependencies:

```bash
pip install -r requirements.txt
# plus libraries used in code but not pinned in requirements.txt:
pip install openai python-dotenv pyyaml
```


### Configure
1) OpenAI key (recommended via `.env` the code already loads):
- Create `config/.env` with:

```bash
OPENAI_API_KEY=your_openai_key_here
```

2) Audio/streaming parameters:
- Edit `config/settings.yaml` to match your system input device and preferences:
  - `audio.input_device`: device index (WASAPI loopback device)
  - `audio.rate`, `audio.channels`
  - `streaming.window_s`, `streaming.step_s`, `vad_rms_thresh`, `min_speech_ms`

3) Personalize content:
- `data/resume.md`: concise resume summary
- `data/current_jd.md`: summary/paste of the target role’s JD


### Run
```bash
conda activate snow
python main.py
```
You should see live transcript snippets, detected questions, and bullet answers. Press Ctrl+C to stop.

Session logs are saved to:
```
data/sessions/<timestamp>/qa_log.md
```


### Troubleshooting
- No audio detected:
  - Set the correct `audio.input_device` index in `config/settings.yaml`
  - Ensure system sound is playing and WASAPI loopback device is available
- STT quality:
  - Tune `streaming.window_s`/`step_s` and `vad_rms_thresh`
  - Try different faster-whisper model sizes via `stt.model`
- OpenAI errors:
  - Verify `config/.env` exists and contains `OPENAI_API_KEY`
  - Network/proxy issues may block API calls


### Notes
- By design, the LLM answers are short, interview-spoken bullets and constrained to your resume/JD context to avoid hallucinations.
- For “behavioral” follow-ups, the engine tries to keep the same project thread consistent across questions.


### License
Proprietary/Personal use unless specified otherwise.

