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

v3.0 - Multimodal PDF Smart Router & Edge-TTS

What this service does:
  Converts study materials into natural-sounding MP3 audio files.
  Features a highly intelligent PDF extraction pipeline that translates
  visual diagrams into spoken-word descriptions.

  The Smart Pipeline:
    1. PDF Ingestion (The Smart Router):
       - Uses PyMuPDF to scan the document.
       - If NO images are found: Fast-tracks pure text extraction.
       - If IMAGES are found (or it is a scanned doc): Uploads the entire
         PDF to the Gemini File API. Gemini acts as an accessibility
         expert, transcribing the text AND writing rich, spoken
         descriptions of every diagram, chart, and graph.
         
    2. Gemini Text Polish (Stage 1):
       - Cleans formatting, expands abbreviations (e.g. BST, O(n)),
         and adds natural lecture transitions.
         
    3. Edge-TTS Synthesis (Stage 2):
       - Uses Microsoft Azure Neural voices (completely free, no billing)
         to synthesize the final MP3 file.

Architecture notes:
  - Uses `tempfile` to securely handle the Gemini File API requirements.
  - Background tasks ensure the API remains non-blocking during heavy OCR.
  - Job states: pending -> processing -> completed / failed
  - Port: 8011

v3.1 - PDF Title Resolution Refinement

What changed:
    - The /convert-pdf endpoint now derives a clean filename from the uploaded PDF
        (for example, "CSC333_Lecture5.pdf" -> "CSC333_Lecture5").
    - If title is left as the default "Uploaded Note", the API now uses the clean
        filename as the effective title.
    - If a custom title is provided, the API merges both values as:
        "{title} - {clean_filename}".
    - This prevents generic defaults from hiding the true source document name.
"""

import io
import os
import json
import sqlite3
import threading
import asyncio
import tempfile
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from typing import AsyncIterator, List, Optional
from uuid import uuid4

import fitz  # PyMuPDF: pip install pymupdf
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
MAX_PDF_TEXT_CHARS = 50_000

# Minimum text length to proceed with conversion
MIN_TEXT_LENGTH = 50

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
        description="Voice selection key."
    )
    student_id: str = Field(
        default="anonymous",
        description="Student identifier for tracking and Insight Engine events."
    )
    google_api_key: str = Field(
        description="Your Google Gemini API key."
    )
    speaking_rate: float = Field(
        default=1.0,
        ge=0.75,
        le=1.5,
        description="Speaking rate multiplier. 1.0 is natural speed."
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


# =========================================================
# DATABASE
# =========================================================

def init_db() -> None:
    """Creates the SQLite jobs table on startup if it does not exist."""
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
    """Context manager for SQLite connections."""
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


def create_job(job_id: str, student_id: str, title: str, voice: str, input_type: str) -> None:
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


def update_job_state(job_id: str, state: str, audio_path: Optional[str] = None, char_count: int = 0, error_message: Optional[str] = None) -> None:
    """Updates a job's state. Called at each stage of the pipeline."""
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
    """Dynamically discovers the best available Gemini generative model."""
    try:
        client = genai.Client(api_key=api_key_str)
        all_models = [m.name.replace("models/", "") for m in client.models.list() if m.name]
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
# STAGE 0: MULTIMODAL PDF EXTRACTION (The Smart Router)
# =========================================================

MULTIMODAL_EXTRACTION_PROMPT = """You are an expert academic accessibility transcriber.
Your task is to transcribe this entire PDF document into text for a visually impaired student 
who will listen to it as an audio file.

INSTRUCTIONS:
1. Transcribe all standard text accurately, maintaining the logical flow.
2. When you encounter a diagram, chart, graph, or image, DO NOT skip it. Instead, insert a highly 
   detailed, plain-English paragraph describing exactly what the visual shows. 
   (e.g., "The diagram illustrates a Binary Search Tree. At the root node is the number 50...")
3. Ensure mathematical formulas and code blocks are transcribed cleanly.

Output ONLY the complete transcribed text with your visual descriptions seamlessly integrated. 
Do not add any conversational filler like "Here is the transcription."
"""

def extract_text_from_pdf_bytes(pdf_bytes: bytes, api_key: str) -> str:
    """
    The Smart Router: Uses PyMuPDF to analyze the PDF first.
    If no images are found, it extracts text quickly. 
    If images are found, it leverages the Gemini File API for multimodal transcription.
    """
    MIN_CHARS_FOR_DIRECT = 100
    
    # --- Check 1: Analyze Document with PyMuPDF ---
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_images = 0
        text_parts = []
        
        for page in doc:
            text_parts.append(page.get_text())
            total_images += len(page.get_images())
            
        full_text = "\n\n".join(text_parts).strip()
        doc.close()
        
        # Smart Routing Decision
        if total_images == 0 and len(full_text) >= MIN_CHARS_FOR_DIRECT:
            print(f"[PDF] Smart Router: 0 images detected. Using fast-track text extraction ({len(full_text)} chars).")
            return full_text[:MAX_PDF_TEXT_CHARS]
        else:
            print(f"[PDF] Smart Router: {total_images} images detected (or scanned doc). Escalating to Gemini File API for visual transcription...")
            
    except Exception as e:
        print(f"[PDF WARNING] PyMuPDF analysis failed: {e}. Escalating to Gemini File API...")

    # --- Check 2: Gemini File API (Multimodal Extraction) ---
    temp_file_path = ""
    try:
        # The Gemini SDK prefers working with physical files for the File API
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(pdf_bytes)
            temp_file_path = temp_pdf.name

        client = genai.Client(api_key=api_key)
        model_name = get_valid_model_name(api_key)
        
        print("[PDF] Uploading document to Google servers for multimodal analysis...")
        uploaded_file = client.files.upload(
            file=temp_file_path, 
            config={'mime_type': 'application/pdf'}
        )
        
        print("[PDF] Document uploaded. Generating rich audio transcription...")
        response = client.models.generate_content(
            model=model_name,
            contents=[uploaded_file, MULTIMODAL_EXTRACTION_PROMPT]
        )
        
        # Cleanup the file from Google's servers
        try:
            if uploaded_file.name:
                client.files.delete(name=uploaded_file.name)
        except Exception as cleanup_err:
            print(f"[PDF WARNING] Failed to delete file from Gemini servers: {cleanup_err}")

        extracted_text = (response.text or "").strip()
        
        if not extracted_text:
            print("[PDF WARNING] Gemini returned empty transcription.")
            return ""
            
        truncated = extracted_text[:MAX_PDF_TEXT_CHARS]
        print(f"[PDF] Gemini File API successfully extracted {len(truncated)} chars with image descriptions.")
        return truncated

    except Exception as e:
        print(f"[PDF ERROR] Gemini File API extraction failed: {e}")
        return ""
    finally:
        # Always clean up the local temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)


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
   - O(n) -> O of n
   - etc. -> and so on
4. Convert LaTeX and math notation to spoken form:
   - ∑ -> the sum of
   - √x -> the square root of x
5. Add natural lecture transitions between sections:
   - Use phrases like "Moving on to", "As we have seen", "Let us now consider".
6. Output ONLY the transformed spoken text. No commentary.

INPUT TEXT:
{raw_text}

SPOKEN TEXT OUTPUT:"""


def preprocess_text_with_gemini(raw_text: str, api_key: str) -> str:
    """Polishes the raw or extracted text into flowing speech."""
    model_name = get_valid_model_name(api_key)
    prompt = TEXT_PREP_PROMPT.format(raw_text=raw_text[:20000])

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(model=model_name, contents=prompt)
        processed = (response.text or "").strip()

        if not processed:
            return raw_text
        print(f"[Stage 1] Pre-processing complete. Input: {len(raw_text)} chars -> Output: {len(processed)} chars.")
        return processed
    except Exception as e:
        print(f"[Stage 1 WARNING] Gemini pre-processing failed: {e}. Using raw text.")
        return raw_text


# =========================================================
# STAGE 2: EDGE-TTS SYNTHESIS
# =========================================================

def format_speaking_rate(rate_float: float) -> str:
    """Converts a float multiplier to Edge-TTS format (e.g. '+25%')."""
    percentage = int(round((rate_float - 1.0) * 100))
    return f"{percentage:+d}%"

async def synthesise_audio_edge(text: str, voice_key: str, speaking_rate: float, output_path: str) -> None:
    """Synthesises spoken audio using the open-source edge-tts library."""
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
    Executes the full pipeline asynchronously. Stage 1 (Gemini) is wrapped 
    in a thread to prevent blocking. Stage 2 (TTS) runs natively async.
    """
    update_job_state(job_id, "processing")
    print(f"[Pipeline] Job '{job_id}' processing started.")

    try:
        # --- Stage 1: Text pre-processing (Threaded) ---
        print(f"[Pipeline] Stage 1: Polishing text with Gemini...")
        spoken_text = await asyncio.to_thread(preprocess_text_with_gemini, raw_text, google_api_key)
        char_count = len(spoken_text)

        if char_count < MIN_TEXT_LENGTH:
            raise ValueError(f"Pre-processed text is too short ({char_count} chars).")

        # --- Stage 2: Edge-TTS Synthesis (Native Async) ---
        print(f"[Pipeline] Stage 2: Synthesising audio with Edge-TTS ({voice_key})...")
        os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)
        safe_title = "".join(c if c.isalnum() or c in " _-" else "" for c in title).strip().replace(" ", "_")[:40]
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{job_id}_{safe_title}_{timestamp}.mp3"
        audio_path = os.path.join(AUDIO_OUTPUT_DIR, filename)

        await synthesise_audio_edge(spoken_text, voice_key, speaking_rate, audio_path)
        print(f"[Pipeline] MP3 saved: '{audio_path}'.")

        # --- Mark complete ---
        update_job_state(job_id, state="completed", audio_path=audio_path, char_count=char_count)
        print(f"[KAFKA MOCK] Published event 'audio.generated' - job: '{job_id}'")

    except Exception as e:
        error_msg = str(e)
        print(f"[Pipeline ERROR] Job '{job_id}' failed: {error_msg}")
        update_job_state(job_id, state="failed", error_message=error_msg)


# =========================================================
# FASTAPI APPLICATION
# =========================================================

@asynccontextmanager
async def lifespan(_: "FastAPI") -> AsyncIterator[None]:
    print("[Startup] AkadVerse Note-to-Audio Converter initialising...")
    os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)
    init_db()
    print("[Startup] Ready. Run with: uvicorn note_to_audio_converter:app --host 127.0.0.1 --port 8011 --reload")
    yield
    print("[Shutdown] AkadVerse Note-to-Audio Converter stopped.")

app = FastAPI(
    title="AkadVerse - Note-to-Audio Converter API",
    description="Tier 5 student tool with Smart Multimodal Routing and Azure Neural Voices.",
    version="3.0",
    lifespan=lifespan
)


# =========================================================
# ENDPOINTS
# =========================================================

@app.post("/convert", response_model=ConvertResponse, tags=["Conversion"])
async def convert_text(request: ConvertTextRequest, background_tasks: BackgroundTasks):
    if request.voice not in VOICE_OPTIONS:
        raise HTTPException(status_code=400, detail=f"Unknown voice '{request.voice}'.")

    job_id = uuid4().hex[:16]
    voice_info = VOICE_OPTIONS[request.voice]
    estimated_seconds = int(((len(request.text) / 5) / 130) * 60 / request.speaking_rate)

    create_job(job_id, request.student_id, request.title, request.voice, "text")

    background_tasks.add_task(
        run_conversion_pipeline,
        job_id, request.text, request.title, request.voice, 
        request.speaking_rate, request.google_api_key, "text"
    )

    return ConvertResponse(
        status="accepted",
        job_id=job_id,
        message="Pipeline running in background. Use GET /audio/{job_id} when ready.",
        title=request.title,
        voice_label=voice_info["label"],
        estimated_duration_seconds=estimated_seconds,
        audio_url=f"/audio/{job_id}"
    )

@app.post("/convert-pdf", response_model=ConvertResponse, tags=["Conversion"])
async def convert_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str = "Uploaded Note",
    voice: str = DEFAULT_VOICE,
    student_id: str = "anonymous",
    speaking_rate: float = 1.0,
    google_api_key: str = Form(...)
):
    if not google_api_key:
        raise HTTPException(status_code=400, detail="google_api_key is required.")

    # Keep filename None-safe because UploadFile.filename can be missing.
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    pdf_bytes = await file.read()
    print(f"[PDF] Initiating Smart Router for '{filename}'...")
    
    # Execute extraction in a thread to prevent blocking
    raw_text = await asyncio.to_thread(extract_text_from_pdf_bytes, pdf_bytes, google_api_key)

    if not raw_text or len(raw_text) < MIN_TEXT_LENGTH:
        raise HTTPException(status_code=422, detail="Extraction failed or text too short.")

    # --- TITLE RESOLUTION (v3.0 refinement) ---
    # Why: when title stays at the default "Uploaded Note", we should surface the
    # real PDF name instead of keeping a generic label.
    # Behavior:
    # 1) "Uploaded Note" + "CSC333_Lecture5.pdf" -> "CSC333_Lecture5"
    # 2) "Midterm Review" + "CSC333_Lecture5.pdf" -> "Midterm Review - CSC333_Lecture5"
    # 3) Missing/empty filename -> fallback to user title (or default title)
    clean_filename = filename.rsplit(".", 1)[0] if "." in filename else filename
    if title == "Uploaded Note":
        effective_title = clean_filename or "Uploaded Note"
    else:
        effective_title = f"{title} - {clean_filename}" if clean_filename else title
    # -----------------------------------------

    job_id = uuid4().hex[:16]
    voice_info = VOICE_OPTIONS[voice]
    estimated_seconds = int(((len(raw_text) / 5) / 130) * 60 / max(speaking_rate, 0.1))

    create_job(job_id, student_id, effective_title, voice, "pdf")

    background_tasks.add_task(
        run_conversion_pipeline,
        job_id, raw_text, effective_title, voice,
        speaking_rate, google_api_key, "pdf"
    )

    return ConvertResponse(
        status="accepted",
        job_id=job_id,
        message="PDF extracted. Pipeline running in background.",
        title=effective_title,
        voice_label=voice_info["label"],
        estimated_duration_seconds=estimated_seconds,
        audio_url=f"/audio/{job_id}"
    )

@app.get("/audio/{job_id}", tags=["Audio"])
async def get_audio(job_id: str):
    with get_db() as conn:
        row = conn.execute("SELECT state, audio_path, error_message FROM audio_jobs WHERE job_id = ?", (job_id,)).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Job not found.")
    
    state = row["state"]
    if state in ("pending", "processing"):
        raise HTTPException(status_code=202, detail=f"Job is {state}. Try again shortly.")
    if state == "failed":
        raise HTTPException(status_code=500, detail=f"Conversion failed: {row['error_message']}")

    audio_path = row["audio_path"]
    if not audio_path or not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="Audio file missing.")

    def audio_stream():
        with open(audio_path, "rb") as f:
            while chunk := f.read(65536):
                yield chunk

    return StreamingResponse(
        audio_stream(),
        media_type="audio/mpeg",
        headers={
            "Content-Disposition": f"attachment; filename={os.path.basename(audio_path)}",
            "Content-Length": str(os.path.getsize(audio_path))
        }
    )

@app.get("/jobs", tags=["Jobs"])
async def list_jobs(student_id: Optional[str] = None, state: Optional[str] = None, limit: int = 10, offset: int = 0):
    try:
        conditions, params = [], []
        if student_id:
            conditions.append("student_id = ?")
            params.append(student_id)
        if state:
            conditions.append("state = ?")
            params.append(state.lower())

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.extend([limit, offset])

        with get_db() as conn:
            rows = conn.execute(f"SELECT * FROM audio_jobs {where} ORDER BY created_at DESC LIMIT ? OFFSET ?", params).fetchall()

        return {"jobs": [dict(row) for row in rows]}
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"DB Error: {e}")

@app.get("/voices", tags=["Configuration"])
async def list_voices():
    return {"default_voice": DEFAULT_VOICE, "voices": VOICE_OPTIONS}

@app.get("/health", tags=["System"])
async def health_check():
    return {"status": "ok", "tts_engine": "edge-tts", "version": "3.0"}