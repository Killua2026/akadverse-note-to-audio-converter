"""
AkadVerse - Note-to-Audio Converter
Tier 5 | Microservice Port: 8011
========================================================================
v1.0 - Initial build - used Google Cloud TTS synthesis in stage 2, which was simple but incurred costs and required API keys.

v2.0 - Open-Source Pivot with Robust Shell.

What this service does:
  Converts any text content (notes, summaries, study guides) into a
  natural-sounding MP3 audio file that students can listen to while
  commuting, exercising, or resting.

  Two-stage pipeline:
    Stage 1 - Gemini text pre-processing:
      Raw notes contain markdown, bullet points, code snippets,
      abbreviations (BST, DSA, O(n log n)), and LaTeX. None of this
      reads naturally aloud. Gemini transforms the raw input into
      clean, spoken-word prose:
        - Strips all markdown formatting
        - Expands abbreviations to full words
        - Converts notation (e.g. O(n log n) -> "O of n log n")
        - Rewrites bullet lists as flowing sentences
        - Adds natural speech transitions ("Moving on to...", "As we noted...")
        - Structures the text to sound like a lecture, not a recitation

    Stage 2 - Edge-TTS synthesis (Microsoft Azure Neural):
      The cleaned prose is sent to the edge-tts engine, which accesses
      Microsoft Azure's Read Aloud API. This requires NO billing and
      NO API keys. Multiple voice options are supported:
        - en-NG (English, Nigeria) - default, for local context
        - en-GB (English, British)
        - en-US (English, American)
        - Male and female variants for each locale

  Input options:
    - Direct text paste (via request body)
    - PDF upload (PyPDF text extraction + Gemini Vision OCR fallback
      for scanned PDFs, ensuring maximum robustness)

  Output:
    - MP3 file saved to audio_output/ folder (simulates GCS cache)
    - Job metadata stored in SQLite (simulates PostgreSQL)
    - Audio streamed directly to student via GET /audio/{job_id}
    - Kafka mock event 'audio.generated' published on completion

Architecture notes:
  - We use asyncio.to_thread() for synchronous Gemini API calls to
    prevent blocking the FastAPI event loop during heavy OCR tasks.
  - Job states: pending -> processing -> completed / failed
  - Port: 8011

Endpoints:
  POST /convert              - Submit text for audio conversion
  POST /convert-pdf          - Upload PDF for audio conversion
  GET  /audio/{job_id}       - Stream the MP3 audio file
  GET  /jobs                 - List conversion jobs
  GET  /voices               - List available voice options
  GET  /health               - Service status
"""

import io
import json
import os
import sqlite3
import threading
import asyncio
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from typing import AsyncIterator, List, Optional
from uuid import uuid4

import pypdf
import edge_tts
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Query, BackgroundTasks, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from google import genai


# Load .env before any constants are resolved
load_dotenv()


# =========================================================
# CONSTANTS
# =========================================================

# Local directory for saved MP3 files (simulates GCS)
AUDIO_OUTPUT_DIR = "audio_output"

# SQLite database for job metadata
DB_PATH = "akadverse_audio.db"

# Maximum characters of PDF text forwarded to the TTS pipeline.
# Beyond this, the audio becomes impractically long.
MAX_PDF_TEXT_CHARS = 50_000

# Minimum text length to proceed with conversion
MIN_TEXT_LENGTH = 50

# For scanned PDF OCR: only process this many pages via Gemini Vision
MAX_OCR_PAGES = 5

# Thread lock for SQLite writes
db_lock = threading.Lock()


# =========================================================
# VOICE CATALOGUE
# =========================================================

# Available voices keyed by a user-friendly label.
# These map to high-quality Microsoft Azure Neural voices via edge-tts.
VOICE_OPTIONS = {
    "nigerian_female": {
        "label":         "Nigerian English - Female (Ezinne)",
        "language_code": "en-NG",
        "voice_name":    "en-NG-EzinneNeural",
        "gender":        "FEMALE",
    },
    "nigerian_male": {
        "label":         "Nigerian English - Male (Abeo)",
        "language_code": "en-NG",
        "voice_name":    "en-NG-AbeoNeural",
        "gender":        "MALE",
    },
    "british_female": {
        "label":         "British English - Female (Sonia)",
        "language_code": "en-GB",
        "voice_name":    "en-GB-SoniaNeural",
        "gender":        "FEMALE",
    },
    "british_male": {
        "label":         "British English - Male (Ryan)",
        "language_code": "en-GB",
        "voice_name":    "en-GB-RyanNeural",
        "gender":        "MALE",
    },
    "american_female": {
        "label":         "American English - Female (Aria)",
        "language_code": "en-US",
        "voice_name":    "en-US-AriaNeural",
        "gender":        "FEMALE",
    },
    "american_male": {
        "label":         "American English - Male (Christopher)",
        "language_code": "en-US",
        "voice_name":    "en-US-ChristopherNeural",
        "gender":        "MALE",
    },
}

# Default voice: Nigerian English female (matches majority of user base)
DEFAULT_VOICE = "nigerian_female"


# =========================================================
# PYDANTIC SCHEMAS
# =========================================================

class ConvertTextRequest(BaseModel):
    """Request body for submitting raw text for audio conversion."""
    text: str = Field(
        ...,
        min_length=MIN_TEXT_LENGTH,
        description="The text content to convert. Can be notes, summaries, or any study material."
    )
    title: str = Field(
        default="Untitled Note",
        description="A title for this audio file, used in the filename and metadata."
    )
    voice: str = Field(
        default=DEFAULT_VOICE,
        description=(
            "Voice selection key. Available: nigerian_female, nigerian_male, "
            "british_female, british_male, american_female, american_male."
        )
    )
    student_id: str = Field(
        default="anonymous",
        description="Student identifier for tracking and Insight Engine events."
    )
    google_api_key: str = Field(
        description="Your Google Gemini API key (used for Stage 1 text pre-processing)."
    )
    speaking_rate: float = Field(
        default=1.0,
        ge=0.75,
        le=1.5,
        description=(
            "Speaking rate multiplier. 1.0 is natural speed. "
            "0.75 is slower (good for complex material). "
            "1.25 is faster (good for review)."
        )
    )


class ConvertResponse(BaseModel):
    """Response returned immediately after a conversion job is submitted."""
    status: str
    job_id: str
    message: str
    title: str
    voice_label: str
    estimated_duration_seconds: Optional[int]
    audio_url: str


class JobRecord(BaseModel):
    """A single job record returned by GET /jobs."""
    job_id: str
    student_id: str
    title: str
    voice: str
    state: str
    char_count: int
    error_message: Optional[str]
    created_at: str
    completed_at: Optional[str]
    audio_url: Optional[str]


# =========================================================
# DATABASE
# =========================================================

def init_db() -> None:
    """
    Creates the SQLite jobs table on startup if it does not exist.

    Table: audio_jobs
      - job_id:          UUID hex, primary key
      - student_id:      Student identifier
      - title:           Human-readable title of the audio file
      - voice:           Voice key used (e.g. 'nigerian_female')
      - state:           pending / processing / completed / failed
      - input_type:      'text' or 'pdf'
      - char_count:      Character count of the processed text
      - audio_path:      Local filesystem path to the MP3 file
      - error_message:   Set if state = failed
      - created_at:      ISO timestamp
      - completed_at:    ISO timestamp (null until completion)
    """
    try:
        with get_db() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audio_jobs (
                    job_id          TEXT PRIMARY KEY,
                    student_id      TEXT NOT NULL,
                    title           TEXT NOT NULL,
                    voice           TEXT NOT NULL,
                    state           TEXT NOT NULL DEFAULT 'pending',
                    input_type      TEXT NOT NULL,
                    char_count      INTEGER NOT NULL DEFAULT 0,
                    audio_path      TEXT,
                    error_message   TEXT,
                    created_at      TEXT NOT NULL,
                    completed_at    TEXT
                )
            """)
            conn.commit()
        print("[DB] Audio jobs database initialised successfully.")
    except sqlite3.Error as e:
        print(f"[DB ERROR] Initialisation failed: {e}")
        raise


@contextmanager
def get_db():
    """
    Context manager for SQLite connections.
    Guarantees connection is always closed even on exception.
    Row factory enables column-name access: row['job_id'].
    """
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        yield conn
    except sqlite3.Error as e:
        if conn:
            conn.rollback()
        print(f"[DB ERROR] {e}")
        raise
    finally:
        if conn:
            conn.close()


def create_job(
    job_id: str,
    student_id: str,
    title: str,
    voice: str,
    input_type: str
) -> None:
    """Inserts a new job row in 'pending' state."""
    with db_lock:
        with get_db() as conn:
            conn.execute(
                """
                INSERT INTO audio_jobs
                  (job_id, student_id, title, voice, state, input_type, created_at)
                VALUES (?, ?, ?, ?, 'pending', ?, ?)
                """,
                (job_id, student_id, title, voice, input_type,
                 datetime.now(timezone.utc).isoformat())
            )
            conn.commit()


def update_job_state(
    job_id: str,
    state: str,
    audio_path: Optional[str] = None,
    char_count: int = 0,
    error_message: Optional[str] = None
) -> None:
    """
    Updates a job's state. Called at each stage of the pipeline.
    Sets completed_at only when state is 'completed' or 'failed'.
    """
    now = datetime.now(timezone.utc).isoformat()
    completed_at = now if state in ("completed", "failed") else None

    with db_lock:
        with get_db() as conn:
            conn.execute(
                """
                UPDATE audio_jobs
                SET state = ?, audio_path = ?, char_count = ?,
                    error_message = ?, completed_at = ?
                WHERE job_id = ?
                """,
                (state, audio_path, char_count, error_message, completed_at, job_id)
            )
            conn.commit()


# =========================================================
# MODEL DISCOVERY
# =========================================================

def get_valid_model_name(api_key_str: str) -> str:
    """
    Dynamically discovers the best available Gemini generative model.
    Consistent with the pattern used across all Tier 5 microservices.
    Falls back to 'gemini-1.5-flash' if discovery fails.
    """
    try:
        client = genai.Client(api_key=api_key_str)
        all_models = [
            m.name.replace("models/", "")
            for m in client.models.list()
            if m.name
        ]
        priority = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-pro"]
        for preferred in priority:
            if preferred in all_models:
                print(f"[Model] Using: {preferred}")
                return preferred
        if all_models:
            return all_models[0]
    except Exception as e:
        print(f"[Model WARNING] Discovery failed ({e}). Defaulting to gemini-1.5-flash.")
    return "gemini-1.5-flash"


# =========================================================
# STAGE 1: GEMINI TEXT PRE-PROCESSING
# =========================================================

TEXT_PREP_PROMPT = """You are preparing academic text to be read aloud as a natural-sounding audio lecture.
Transform the input text so it sounds like a knowledgeable lecturer speaking to students.

TRANSFORMATION RULES:
1. Remove ALL markdown formatting: no ##, **, *, -, backticks, or underscores.
2. Convert bullet lists into flowing prose sentences. Join related points naturally.
3. Expand common academic abbreviations:
   - BST -> Binary Search Tree
   - DSA -> Data Structures and Algorithms
   - LLM -> Large Language Model
   - API -> Application Programming Interface
   - DB -> database
   - O(n) -> O of n
   - O(n log n) -> O of n log n
   - O(n²) -> O of n squared
   - etc. -> and so on
4. Convert LaTeX and math notation to spoken form:
   - ∑ -> the sum of
   - ∫ -> the integral of
   - √x -> the square root of x
5. Add natural lecture transitions between sections:
   - Use phrases like "Moving on to", "As we have seen", "It is important to note that",
     "Let us now consider", "To summarise this section"
6. Convert headings into spoken section introductions:
   - "## 3.2 Time Complexity" -> "In this section, we examine time complexity."
7. Numbers: spell out 1-10 in narrative text. Keep larger numbers as digits.
8. Do NOT add any content that was not in the original text.
9. Do NOT add a title or introduction unless one was present in the original.
10. Output ONLY the transformed spoken text. No commentary, no formatting.

INPUT TEXT:
{raw_text}

SPOKEN TEXT OUTPUT:"""


def preprocess_text_with_gemini(raw_text: str, api_key: str) -> str:
    """
    Stage 1 of the pipeline: sends raw notes to Gemini for transformation
    into natural spoken-word prose suitable for text-to-speech synthesis.

    If Gemini pre-processing fails for any reason, the raw text is returned
    as a fallback so the TTS stage can still proceed with something.

    Args:
        raw_text:  The original notes text, possibly containing markdown.
        api_key:   Gemini API key.

    Returns:
        Cleaned, spoken-word text ready for TTS synthesis.
    """
    model_name = get_valid_model_name(api_key)
    prompt = TEXT_PREP_PROMPT.format(raw_text=raw_text[:20000])  # Cap for safety

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        processed = (response.text or "").strip()

        if not processed:
            print("[Stage 1 WARNING] Gemini returned empty output. Using raw text as fallback.")
            return raw_text

        print(
            f"[Stage 1] Text pre-processing complete. "
            f"Input: {len(raw_text)} chars -> Output: {len(processed)} chars."
        )
        return processed

    except Exception as e:
        print(f"[Stage 1 WARNING] Gemini pre-processing failed: {e}. Using raw text.")
        return raw_text


# =========================================================
# PDF TEXT EXTRACTION
# =========================================================

def extract_text_from_pdf_bytes(pdf_bytes: bytes, api_key: str) -> str:
    """
    Extracts text from a PDF using the same robust two-stage strategy as the
    Sample Questions Generator.

    Stage A - PyPDF (fast, works on digitally created PDFs):
        Reads the embedded text layer directly. If this yields
        more than 100 characters, it is used directly.

    Stage B - Gemini Vision OCR (for scanned / image-based PDFs):
        Converts pages to JPEG and sends them to Gemini Vision
        if PyPDF returns insufficient text.

    Returns the extracted text, truncated to MAX_PDF_TEXT_CHARS.
    """
    MIN_CHARS_FOR_DIRECT = 100

    # - Stage A: PyPDF -
    try:
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        pages_text = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text.strip())
        full_text = "\n\n".join(pages_text).strip()

        if len(full_text) >= MIN_CHARS_FOR_DIRECT:
            truncated = full_text[:MAX_PDF_TEXT_CHARS]
            print(f"[PDF] Stage A (PyPDF): Extracted {len(truncated)} chars.")
            return truncated
        else:
            print(
                f"[PDF] Stage A (PyPDF): Only {len(full_text)} chars. "
                "PDF is likely scanned. Escalating to Stage B (Gemini OCR)..."
            )
    except Exception as e:
        print(f"[PDF] Stage A (PyPDF) failed: {e}. Escalating to Stage B...")

    # - Stage B: Gemini Vision OCR -
    try:
        import base64
        from pdf2image import convert_from_bytes

        print("[PDF] Stage B: Converting pages to images for OCR...")
        images = convert_from_bytes(
            pdf_bytes,
            first_page=1,
            last_page=MAX_OCR_PAGES,
            dpi=200,
            fmt="jpeg"
        )

        model_name = get_valid_model_name(api_key)
        client = genai.Client(api_key=api_key)

        ocr_parts = []
        for i, image in enumerate(images, start=1):
            img_buf = io.BytesIO()
            image.save(img_buf, format="JPEG", quality=85)
            img_b64 = base64.b64encode(img_buf.getvalue()).decode("utf-8")

            from google.genai.types import Content, Part, Blob
            response = client.models.generate_content(
                model=model_name,
                contents=[
                    Content(parts=[
                        Part(inline_data=Blob(
                            mime_type="image/jpeg",
                            data=base64.b64decode(img_b64)
                        )),
                        Part(text=(
                            "Extract all text from this academic document page exactly as it appears. "
                            "Preserve headings, bullet points, and mathematical notation."
                        ))
                    ])
                ]
            )
            page_text = (response.text or "").strip()
            print(f"[PDF] Stage B: Page {i} OCR returned {len(page_text)} chars.")
            if page_text:
                ocr_parts.append(page_text)

        combined = "\n\n".join(ocr_parts).strip()
        if not combined:
            print("[PDF WARNING] Stage B: No text extracted from scanned PDF.")
            return ""

        truncated = combined[:MAX_PDF_TEXT_CHARS]
        print(f"[PDF] Stage B: Successfully extracted {len(truncated)} chars.")
        return truncated

    except ImportError:
        print(
            "[PDF WARNING] pdf2image not installed. Install: pip install pdf2image. "
            "Also requires poppler on system PATH."
        )
        return ""
    except Exception as e:
        print(f"[PDF WARNING] Stage B OCR failed: {e}.")
        return ""


# =========================================================
# STAGE 2: EDGE-TTS SYNTHESIS (Open Source)
# =========================================================

def format_speaking_rate(rate_float: float) -> str:
    """
    Converts a float multiplier (e.g., 1.25) to Edge-TTS string format (e.g., '+25%').
    """
    percentage = int(round((rate_float - 1.0) * 100))
    return f"{percentage:+d}%"

async def synthesise_audio_edge(
    text: str,
    voice_key: str,
    speaking_rate: float,
    output_path: str
) -> None:
    """
    Stage 2 of the pipeline: synthesises spoken audio using the open-source
    edge-tts library, bypassing Google Cloud billing constraints.

    Edge-TTS automatically chunks large texts and streams the response
    securely to disk, so we do not need to manually chunk it anymore.
    """
    if voice_key not in VOICE_OPTIONS:
        raise ValueError(f"Unknown voice key: '{voice_key}'.")

    voice_name = VOICE_OPTIONS[voice_key]["voice_name"]
    rate_str = format_speaking_rate(speaking_rate)

    try:
        communicate = edge_tts.Communicate(text, voice_name, rate=rate_str)
        await communicate.save(output_path)
    except Exception as e:
        raise RuntimeError(f"Edge-TTS synthesis failed: {e}") from e


# =========================================================
# CONVERSION PIPELINE (runs in background)
# =========================================================

async def run_conversion_pipeline(
    job_id: str,
    raw_text: str,
    title: str,
    voice_key: str,
    speaking_rate: float,
    google_api_key: str,
    input_type: str
) -> None:
    """
    Executes the full two-stage conversion pipeline asynchronously.

    This function is called via FastAPI's BackgroundTasks. Because Stage 1
    (Gemini prep) makes a synchronous network call, we wrap it in asyncio.to_thread
    so it does not block the FastAPI event loop. Stage 2 (edge-tts) runs natively async.

    Pipeline:
      1. Update job state to 'processing'
      2. Stage 1: Gemini text pre-processing (run in thread)
      3. Stage 2: Edge-TTS synthesis (run async)
      4. Update job state to 'completed' with audio_path
      5. Publish Kafka mock event

    On any unhandled exception, job state is set to 'failed'.
    """
    update_job_state(job_id, "processing")
    print(f"[Pipeline] Job '{job_id}' processing started.")

    try:
        # --- Stage 1: Gemini text pre-processing (Threaded to prevent blocking) ---
        print(f"[Pipeline] Stage 1: Pre-processing {len(raw_text)} chars with Gemini...")
        spoken_text = await asyncio.to_thread(preprocess_text_with_gemini, raw_text, google_api_key)
        char_count = len(spoken_text)

        if char_count < MIN_TEXT_LENGTH:
            raise ValueError(
                f"Pre-processed text is too short ({char_count} chars). "
                "The input may not contain readable text."
            )

        # --- Stage 2: Edge-TTS Synthesis (Native Async) ---
        print(f"[Pipeline] Stage 2: Synthesising {char_count} chars with Edge-TTS ({voice_key})...")
        
        os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)
        safe_title = "".join(
            c if c.isalnum() or c in " _-" else ""
            for c in title
        ).strip().replace(" ", "_")[:40]
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{job_id}_{safe_title}_{timestamp}.mp3"
        audio_path = os.path.join(AUDIO_OUTPUT_DIR, filename)

        await synthesise_audio_edge(spoken_text, voice_key, speaking_rate, audio_path)

        print(f"[Pipeline] MP3 saved: '{audio_path}'.")

        # --- Mark job complete ---
        update_job_state(
            job_id,
            state="completed",
            audio_path=audio_path,
            char_count=char_count
        )

        # --- Kafka mock event ---
        print(
            f"[KAFKA MOCK] Published event 'audio.generated' - "
            f"job: '{job_id}', title: '{title}', "
            f"voice: '{voice_key}', chars: {char_count}"
        )

    except Exception as e:
        error_msg = str(e)
        print(f"[Pipeline ERROR] Job '{job_id}' failed: {error_msg}")
        update_job_state(job_id, state="failed", error_message=error_msg)


# =========================================================
# FASTAPI APPLICATION
# =========================================================

@asynccontextmanager
async def lifespan(_: "FastAPI") -> AsyncIterator[None]:
    """Startup: initialise DB and create output folder."""
    print("[Startup] AkadVerse Note-to-Audio Converter initialising...")

    os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)
    init_db()

    print("[Startup] TTS Engine: edge-tts (Open Source Azure Neural).")
    print(
        "[Startup] Ready. Run with: "
        "uvicorn note_to_audio_converter:app --host 127.0.0.1 --port 8011 --reload"
    )
    yield
    print("[Shutdown] AkadVerse Note-to-Audio Converter stopped.")


app = FastAPI(
    title="AkadVerse - Note-to-Audio Converter API",
    description=(
        "Tier 5 student tool. Converts lecture notes, summaries, and study guides "
        "into natural-sounding MP3 audio files using Gemini + Edge-TTS."
    ),
    version="2.0",
    lifespan=lifespan
)


# =========================================================
# ENDPOINT 1: Convert text to audio
# =========================================================

@app.post("/convert", response_model=ConvertResponse, tags=["Conversion"])
async def convert_text(
    request: ConvertTextRequest,
    background_tasks: BackgroundTasks
):
    """
    Submits text for audio conversion. Returns immediately with a job_id.
    The two-stage pipeline (Gemini pre-processing + TTS synthesis) runs
    in the background. Poll GET /jobs or GET /audio/{job_id} to check status.
    """
    if request.voice not in VOICE_OPTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown voice '{request.voice}'. Valid: {list(VOICE_OPTIONS.keys())}"
        )

    job_id = uuid4().hex[:16]
    voice_info = VOICE_OPTIONS[request.voice]

    # Rough duration estimate
    estimated_words = len(request.text) / 5
    estimated_seconds = int((estimated_words / 130) * 60 / request.speaking_rate)

    create_job(
        job_id=job_id,
        student_id=request.student_id,
        title=request.title,
        voice=request.voice,
        input_type="text"
    )

    background_tasks.add_task(
        run_conversion_pipeline,
        job_id=job_id,
        raw_text=request.text,
        title=request.title,
        voice_key=request.voice,
        speaking_rate=request.speaking_rate,
        google_api_key=request.google_api_key,
        input_type="text"
    )

    print(
        f"[Convert] Job '{job_id}' queued: '{request.title}' | "
        f"Voice: {voice_info['label']} | "
        f"Input: {len(request.text):,} chars"
    )

    return ConvertResponse(
        status="accepted",
        job_id=job_id,
        message=(
            f"Conversion job accepted. The pipeline is running in the background. "
            f"Use GET /audio/{job_id} to download when ready."
        ),
        title=request.title,
        voice_label=voice_info["label"],
        estimated_duration_seconds=estimated_seconds,
        audio_url=f"/audio/{job_id}"
    )


# =========================================================
# ENDPOINT 2: Convert PDF to audio
# =========================================================

@app.post("/convert-pdf", response_model=ConvertResponse, tags=["Conversion"])
async def convert_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="PDF file to convert to audio."),
    title: str = "Uploaded Note",
    voice: str = DEFAULT_VOICE,
    student_id: str = "anonymous",
    speaking_rate: float = 1.0,
    google_api_key: str = Form(...)
):
    """
    Accepts a PDF upload, extracts the text (PyPDF or Gemini OCR for scanned
    PDFs), then runs the same two-stage pipeline as /convert.
    """
    if not google_api_key:
        raise HTTPException(
            status_code=400,
            detail="google_api_key is required for PDF conversion."
        )
    if voice not in VOICE_OPTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown voice '{voice}'. Valid: {list(VOICE_OPTIONS.keys())}"
        )

    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    resolved_title = (title or "").strip() or filename or "Uploaded Note"

    pdf_bytes = await file.read()

    print(f"[PDF] Extracting text from '{filename}'...")
    # Wrap synchronous extraction in to_thread to prevent blocking the worker
    raw_text = await asyncio.to_thread(extract_text_from_pdf_bytes, pdf_bytes, google_api_key)

    if not raw_text or len(raw_text) < MIN_TEXT_LENGTH:
        raise HTTPException(
            status_code=422,
            detail=(
                "Could not extract sufficient text from the PDF. "
                "If this is a scanned document, ensure poppler and pdf2image "
                "are installed for Gemini Vision OCR fallback."
            )
        )

    job_id = uuid4().hex[:16]
    voice_info = VOICE_OPTIONS[voice]
    estimated_words = len(raw_text) / 5
    estimated_seconds = int((estimated_words / 130) * 60 / max(speaking_rate, 0.1))

    create_job(
        job_id=job_id,
        student_id=student_id,
        title=resolved_title,
        voice=voice,
        input_type="pdf"
    )

    background_tasks.add_task(
        run_conversion_pipeline,
        job_id=job_id,
        raw_text=raw_text,
        title=resolved_title,
        voice_key=voice,
        speaking_rate=speaking_rate,
        google_api_key=google_api_key,
        input_type="pdf"
    )

    print(
        f"[Convert-PDF] Job '{job_id}' queued: '{filename}' | "
        f"Extracted: {len(raw_text):,} chars | Voice: {voice_info['label']}"
    )

    return ConvertResponse(
        status="accepted",
        job_id=job_id,
        message=(
            f"PDF text extracted ({len(raw_text):,} chars). "
            f"Conversion pipeline running. Use GET /audio/{job_id} when ready."
        ),
        title=resolved_title,
        voice_label=voice_info["label"],
        estimated_duration_seconds=estimated_seconds,
        audio_url=f"/audio/{job_id}"
    )


# =========================================================
# ENDPOINT 3: Stream the MP3 audio file
# =========================================================

@app.get("/audio/{job_id}", tags=["Audio"])
async def get_audio(job_id: str):
    """
    Streams the generated MP3 audio file for a completed job.
    """
    with get_db() as conn:
        row = conn.execute(
            "SELECT state, audio_path, error_message FROM audio_jobs WHERE job_id = ?",
            (job_id,)
        ).fetchone()

    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"Job '{job_id}' not found."
        )

    state = row["state"]

    if state in ("pending", "processing"):
        raise HTTPException(
            status_code=202,
            detail=f"Job is still {state}. Please try again in a few seconds."
        )

    if state == "failed":
        raise HTTPException(
            status_code=500,
            detail=f"Conversion failed: {row['error_message']}"
        )

    audio_path = row["audio_path"]
    if not audio_path or not os.path.exists(audio_path):
        raise HTTPException(
            status_code=404,
            detail="Audio file not found on disk. It may have been deleted."
        )

    def audio_stream():
        with open(audio_path, "rb") as f:
            while chunk := f.read(65536):   # 64KB chunks
                yield chunk

    filename = os.path.basename(audio_path)
    return StreamingResponse(
        audio_stream(),
        media_type="audio/mpeg",
        headers={
            "Content-Disposition": f"attachment; filename={filename}",
            "Content-Length": str(os.path.getsize(audio_path))
        }
    )


# =========================================================
# ENDPOINT 4: List jobs
# =========================================================

@app.get("/jobs", tags=["Jobs"])
async def list_jobs(
    student_id: Optional[str] = Query(default=None, description="Filter by student ID."),
    state: Optional[str] = Query(
        default=None,
        description="Filter by state: pending, processing, completed, failed."
    ),
    limit: int = Query(default=10, ge=1, le=50),
    offset: int = Query(default=0, ge=0)
):
    """
    Lists audio conversion jobs, most recent first.
    Includes a direct audio URL for completed jobs.
    """
    try:
        conditions = []
        params = []

        if student_id:
            conditions.append("student_id = ?")
            params.append(student_id)
        if state:
            conditions.append("state = ?")
            params.append(state.lower())

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.extend([limit, offset])

        with get_db() as conn:
            rows = conn.execute(
                f"""
                SELECT job_id, student_id, title, voice, state, input_type,
                       char_count, audio_path, error_message, created_at, completed_at
                FROM audio_jobs
                {where}
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                params
            ).fetchall()

        if not rows:
            return {"jobs": [], "message": "No jobs found."}

        return {
            "jobs": [
                {
                    "job_id":       row["job_id"],
                    "student_id":   row["student_id"],
                    "title":        row["title"],
                    "voice":        row["voice"],
                    "voice_label":  VOICE_OPTIONS.get(row["voice"], {}).get("label", row["voice"]),
                    "state":        row["state"],
                    "input_type":   row["input_type"],
                    "char_count":   row["char_count"],
                    "error_message": row["error_message"],
                    "created_at":   row["created_at"],
                    "completed_at": row["completed_at"],
                    "audio_url":    f"/audio/{row['job_id']}" if row["state"] == "completed" else None
                }
                for row in rows
            ],
            "total_returned": len(rows)
        }

    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")


# =========================================================
# ENDPOINT 5: List available voices
# =========================================================

@app.get("/voices", tags=["Configuration"])
async def list_voices():
    """Returns all available voice options with their labels and language codes."""
    return {
        "default_voice": DEFAULT_VOICE,
        "voices": [
            {
                "key":           key,
                "label":         v["label"],
                "language_code": v["language_code"],
                "gender":        v["gender"],
            }
            for key, v in VOICE_OPTIONS.items()
        ]
    }


# =========================================================
# ENDPOINT 6: Health check
# =========================================================

@app.get("/health", tags=["System"])
async def health_check():
    """Reports service status, TTS engine state, and job statistics."""
    try:
        with get_db() as conn:
            total      = conn.execute("SELECT COUNT(*) FROM audio_jobs").fetchone()[0]
            pending    = conn.execute("SELECT COUNT(*) FROM audio_jobs WHERE state='pending'").fetchone()[0]
            processing = conn.execute("SELECT COUNT(*) FROM audio_jobs WHERE state='processing'").fetchone()[0]
            completed  = conn.execute("SELECT COUNT(*) FROM audio_jobs WHERE state='completed'").fetchone()[0]
            failed     = conn.execute("SELECT COUNT(*) FROM audio_jobs WHERE state='failed'").fetchone()[0]
    except Exception:
        total = pending = processing = completed = failed = 0

    return {
        "status":                  "ok",
        "version":                 "2.0",
        "tts_engine":              "edge-tts",
        "audio_output_directory":  AUDIO_OUTPUT_DIR,
        "output_dir_exists":       os.path.exists(AUDIO_OUTPUT_DIR),
        "jobs_total":              total,
        "jobs_pending":            pending,
        "jobs_processing":         processing,
        "jobs_completed":          completed,
        "jobs_failed":             failed,
        "available_voices":        list(VOICE_OPTIONS.keys()),
        "endpoints": {
            "POST /convert":           "Submit text for audio conversion.",
            "POST /convert-pdf":       "Upload a PDF for audio conversion.",
            "GET  /audio/{job_id}":    "Stream the MP3 once completed.",
            "GET  /jobs":              "List conversion jobs with status.",
            "GET  /voices":            "List available voice options.",
            "GET  /health":            "This endpoint."
        }
    }


# =========================================================
# Run: uvicorn note_to_audio_converter:app --host 127.0.0.1 --port 8011 --reload
# =========================================================