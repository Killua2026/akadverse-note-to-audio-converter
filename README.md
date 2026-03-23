# AkadVerse: Note-to-Audio Converter
### Tier 5 Learning AI Tool | Microservice Port: `8011`

> An intelligent accessibility and study tool. Converts lecture notes,
> summaries, and PDF documents into natural-sounding MP3 audio files using
> a two-stage AI pipeline. Features a Smart Router that automatically detects
> visual diagrams in PDFs and uses Gemini's multimodal capability to translate
> those images into rich, spoken-word descriptions -- so auditory learners
> miss nothing.

---

## Table of Contents

1. [What This Microservice Does](#what-this-microservice-does)
2. [The Smart Router Pipeline](#the-smart-router-pipeline)
3. [Architecture Overview](#architecture-overview)
4. [Prerequisites](#prerequisites)
5. [Getting Your API Key](#getting-your-api-key)
6. [Installation](#installation)
7. [Running the Server](#running-the-server)
8. [API Endpoints](#api-endpoints)
   - [POST /convert](#1-post-convert)
   - [POST /convert-pdf](#2-post-convert-pdf)
   - [GET /audio/{job\_id}](#3-get-audiojob_id)
   - [GET /jobs](#4-get-jobs)
   - [GET /voices](#5-get-voices)
   - [GET /health](#6-get-health)
9. [Testing with Swagger UI](#testing-with-swagger-ui)
10. [Example Test Inputs](#example-test-inputs)
11. [Understanding the Responses](#understanding-the-responses)
12. [Available Voices](#available-voices)
13. [Generated Files](#generated-files)
14. [Common Errors and Fixes](#common-errors-and-fixes)
15. [Project Structure](#project-structure)

---

## What This Microservice Does

This service is a **Tier 5 component** of the AkadVerse AI-first e-learning
platform, living inside the *My Learning* module as a passive study tool.

When a student submits text or uploads a PDF, the service runs a two-stage
AI pipeline:

1. **Gemini text polish (Stage 1):** Strips markdown, expands abbreviations
   like BST and DSA, converts notation like `O(n log n)` to "O of n log n",
   rewrites bullet points as flowing prose, and adds natural lecture
   transitions so the output sounds like a real class, not a robotic
   text dump.
2. **Edge-TTS synthesis (Stage 2):** Microsoft Azure Neural voices synthesise
   the polished text into a high-quality MP3 file. No billing, no API keys,
   no quota limits for TTS.

Both stages run in the background. The API returns immediately with a
`job_id` so the server never times out on long documents.

---

## The Smart Router Pipeline

The most significant architectural feature of this service is how it handles
PDFs. When a PDF is uploaded, PyMuPDF acts as a traffic controller before
any AI is involved:

**Path A - Fast Track (text-only PDFs):**
If the PDF contains zero images, PyMuPDF extracts the text locally in
milliseconds. No AI is used at this stage. The extracted text goes
straight to Stage 1 for polishing.

**Path B - Multimodal Escalation (PDFs with images):**
If the PDF contains any images -- charts, flowcharts, UML diagrams,
circuit schematics -- the entire document is uploaded to the Gemini
File API. Gemini acts as an accessibility expert: it transcribes all
text accurately and, when it encounters a visual, inserts a rich
plain-English paragraph describing exactly what the diagram shows.

For example, a BST diagram becomes: "The diagram illustrates a Binary
Search Tree. At the root node is the number 50, with 30 as the left
child and 70 as the right child..."

This path also handles scanned PDFs that have no embedded text layer,
since the Gemini File API reads the document visually rather than
trying to extract a non-existent text layer.

---

## Architecture Overview

```
Student submits text or uploads PDF
        |
        v
Is it a PDF?
  |-- NO:  Skip to Stage 1
  |-- YES: PyMuPDF scans document
        |
        |-- 0 images: Fast-track local text extraction
        |-- 1+ images or scanned doc:
                Upload to Gemini File API
                Multimodal OCR + image descriptions
        |
        v
Stage 1: Gemini Text Polish (runs in thread to avoid blocking)
  - Strip markdown, expand abbreviations, add speech transitions
        |
        v
Stage 2: Edge-TTS Synthesis (native async)
  - Microsoft Azure Neural voices, completely free, no billing
        |
        v
MP3 saved to audio_output/
Job state updated to 'completed' in SQLite
Kafka mock event: audio.generated published
```

**Key design decisions:**

- **Edge-TTS instead of Google Cloud TTS:** The original v1.0 used Google
  Cloud Standard voices and required a service account JSON credential file.
  The current version uses `edge-tts`, which accesses Microsoft Azure Neural
  voices for free with zero configuration. The voice quality is significantly
  better -- named personalities like Ezinne, Abeo, Sonia, and Ryan.
- **PyMuPDF as the Smart Router:** Rather than sending every PDF to an AI
  model, PyMuPDF first inspects the document cheaply. Text-only PDFs never
  touch any external API for extraction, saving Gemini quota.
- **Gemini File API for visual PDFs:** The File API receives the entire PDF
  as a document object, not as page-by-page images. This gives Gemini full
  document context for more coherent transcriptions.
- **`asyncio.to_thread()` for synchronous Gemini calls:** Wrapping the
  synchronous Gemini client calls in `asyncio.to_thread()` prevents them
  from blocking the FastAPI event loop during heavy extraction.
- **`tempfile` for secure PDF handling:** The Gemini File API requires a
  physical file path. Uploaded bytes are written to a secure temporary file,
  uploaded, then both the local temp file and the uploaded copy on Google's
  servers are deleted in a `finally` block.
- **Dynamic title resolution:** If no custom title is given, the PDF's
  actual filename is used. If a custom title is given, it merges with the
  filename as `"{title} - {filename}"`, preventing generic defaults from
  hiding the true document source.

---

## Prerequisites

- **Python 3.10 or higher**
- **pip** (Python package manager)
- A **Google Gemini API key** (free tier is sufficient for all features)
- An **internet connection** (required for both Gemini and Edge-TTS)

> **No Google Cloud project, no service account, no billing setup required.**
> All TTS synthesis uses Edge-TTS, which accesses Microsoft Azure Neural
> voices completely free.

---

## Getting Your API Key

1. Go to [https://aistudio.google.com/apikey](https://aistudio.google.com/apikey)
2. Sign in with a Google account.
3. Click **Create API Key**.
4. Copy the key -- you will paste it into the Swagger UI request body.

> The Gemini API key is used for Stage 1 text polishing and for multimodal
> PDF transcription (Path B). It is not involved in TTS synthesis.

---

## Installation

### Step 1 - Set up your project folder

```
akadverse-note-to-audio-converter/
|-- note_to_audio_converter.py
`-- requirements.txt
```

### Step 2 - Create and activate a virtual environment

```bash
# Create
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate
```

### Step 3 - Install dependencies

```bash
pip install -r requirements.txt
```

Full dependency reference:

| Package | Purpose |
|---|---|
| `fastapi` | Web framework for the API |
| `uvicorn` | ASGI server to run FastAPI |
| `google-genai>=1.67.0` | Gemini SDK for text polish and multimodal PDF extraction |
| `edge-tts` | Free Microsoft Azure Neural TTS voices |
| `pymupdf` | PDF analysis (Smart Router) and fast text extraction |
| `python-dotenv` | Loads `.env` for configuration |
| `pydantic` | Data validation and response schemas |
| `python-multipart` | Required by FastAPI for PDF file upload handling |

---

## Running the Server

From inside your project folder with the virtual environment activated:

```bash
uvicorn note_to_audio_converter:app --host 127.0.0.1 --port 8011 --reload
```

**Expected startup output:**

```
[Startup] AkadVerse Note-to-Audio Converter initialising...
[DB] Audio jobs database initialised successfully.
[Startup] TTS Engine: edge-tts (Open Source Azure Neural).
[Startup] Ready. Run with: uvicorn note_to_audio_converter:app --host 127.0.0.1 --port 8011 --reload
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8011 (Press CTRL+C to quit)
```

Two items are created automatically on first startup:

- `audio_output/` -- output folder for generated MP3 files
- `akadverse_audio.db` -- SQLite job tracking database

---

## API Endpoints

### 1. `POST /convert`

**What it does:** Accepts raw text, runs the two-stage pipeline in the
background, and returns immediately with a `job_id`.

**Request body (JSON):**

| Field | Required | Default | Description |
|---|---|---|---|
| `text` | Yes | -- | The text to convert (minimum 50 characters) |
| `title` | No | `"Untitled Note"` | Title used in the filename and metadata |
| `voice` | No | `"nigerian_female"` | Voice key -- see `/voices` for all options |
| `student_id` | No | `"anonymous"` | Student identifier for tracking |
| `google_api_key` | Yes | -- | Your Gemini API key |
| `speaking_rate` | No | `1.0` | Speed multiplier: 0.75 (slow) to 1.5 (fast) |

**Success response (200 OK):**

```json
{
  "status": "accepted",
  "job_id": "a3f1b2c4d5e6f708",
  "message": "Pipeline running in background. Use GET /audio/{job_id} when ready.",
  "title": "BST Overview",
  "voice_label": "Nigerian English - Female (Ezinne)",
  "estimated_duration_seconds": 42,
  "audio_url": "/audio/a3f1b2c4d5e6f708"
}
```

**Expected terminal output during processing:**

```
[Model] Using: gemini-2.5-flash
[Stage 1] Pre-processing complete. Input: 412 chars -> Output: 538 chars.
[Pipeline] Stage 2: Synthesising audio with Edge-TTS (nigerian_female)...
[Pipeline] MP3 saved: 'audio_output/a3f1b2c4_BST_Overview_20260321_120000.mp3'.
[KAFKA MOCK] Published event 'audio.generated' - job: 'a3f1b2c4d5e6f708'
```

---

### 2. `POST /convert-pdf`

**What it does:** Accepts a PDF upload, routes it through the Smart Router,
extracts text (with image descriptions where applicable), then runs the
same two-stage pipeline as `/convert`.

**Form fields:**

| Field | Required | Default | Description |
|---|---|---|---|
| `file` | Yes | -- | The PDF file to convert |
| `google_api_key` | Yes | -- | Your Gemini API key |
| `title` | No | `"Uploaded Note"` | Custom title, merged with filename if provided |
| `voice` | No | `"nigerian_female"` | Voice key |
| `student_id` | No | `"anonymous"` | Student identifier |
| `speaking_rate` | No | `1.0` | Speed multiplier (0.75 to 1.5) |

**Title resolution behaviour:**

| Scenario | Result |
|---|---|
| No custom title + `lecture5.pdf` | `"lecture5"` |
| `"Midterm Review"` + `lecture5.pdf` | `"Midterm Review - lecture5"` |
| No custom title + no filename | `"Uploaded Note"` |

**Smart Router terminal output (text-only PDF):**

```
[PDF] Smart Router: 0 images detected. Using fast-track text extraction (3,241 chars).
```

**Smart Router terminal output (PDF with diagrams):**

```
[PDF] Smart Router: 4 images detected. Escalating to Gemini File API for visual transcription...
[PDF] Uploading document to Google servers for multimodal analysis...
[PDF] Gemini File API successfully extracted 5,872 chars with image descriptions.
```

---

### 3. `GET /audio/{job_id}`

**What it does:** Streams the generated MP3 file once the job is completed.

**Responses by status code:**

| Code | Meaning |
|---|---|
| `200 OK` | Job complete. MP3 streams as a download. |
| `202 Accepted` | Job still pending or processing. Try again in a few seconds. |
| `500 Internal Server Error` | Job failed. Error message explains the cause. |
| `404 Not Found` | Job ID does not exist. |

**How to download the audio:** Open this URL directly in your browser:

```
http://127.0.0.1:8011/audio/YOUR_JOB_ID
```

Your browser will play or download the MP3 immediately.

---

### 4. `GET /jobs`

**What it does:** Lists conversion jobs with optional filters, most recent
first.

**Query parameters:**

| Parameter | Default | Description |
|---|---|---|
| `student_id` | -- | Filter by student ID |
| `state` | -- | Filter by state: `pending`, `processing`, `completed`, `failed` |
| `limit` | `10` | Max records returned |
| `offset` | `0` | Records to skip for pagination |

**Success response (200 OK):**

```json
{
  "jobs": [
    {
      "job_id": "a3f1b2c4d5e6f708",
      "student_id": "23CE034397",
      "title": "BST Overview",
      "voice": "nigerian_female",
      "state": "completed",
      "input_type": "text",
      "char_count": 538,
      "audio_path": "audio_output/a3f1b2c4_BST_Overview_20260321_120000.mp3",
      "error_message": null,
      "created_at": "2026-03-21T12:00:00+00:00",
      "completed_at": "2026-03-21T12:00:18+00:00"
    }
  ]
}
```

---

### 5. `GET /voices`

**What it does:** Returns all available voice options with their labels,
language codes, and gender.

---

### 6. `GET /health`

**What it does:** Reports service status and TTS engine configuration.

**Success response (200 OK):**

```json
{
  "status": "ok",
  "tts_engine": "edge-tts",
  "version": "3.0"
}
```

---

## Testing with Swagger UI

With the server running, open:

```
http://127.0.0.1:8011/docs
```

---

## Example Test Inputs

Run these tests in order for a complete end-to-end verification.

---

### Test 1 - Health check

`GET /health` -- confirm `"tts_engine": "edge-tts"` and `"status": "ok"`.

---

### Test 2 - List voices

`GET /voices` -- verify all six voices are listed. Note the Nigerian English
voices (`en-NG-EzinneNeural`, `en-NG-AbeoNeural`) which are the default
for the AkadVerse user base.

---

### Test 3 - Convert text

`POST /convert` with the following JSON body:

```json
{
  "text": "Binary Search Trees (BST) are a fundamental data structure in computer science. The time complexity for search operations is O(log n) in the average case for a balanced tree, but degrades to O(n) in the worst case when the tree becomes skewed. The in-order traversal of a BST yields all elements in sorted ascending order.",
  "title": "BST Overview",
  "voice": "nigerian_female",
  "student_id": "23CE034397",
  "google_api_key": "YOUR_KEY",
  "speaking_rate": 1.0
}
```

**Expected:** `200 OK` with `status: accepted` and a `job_id`. Watch the
terminal to see Gemini expand "BST" to "Binary Search Tree" and "O(log n)"
to "O of log n" before Edge-TTS synthesises the audio. Copy the `job_id`
for Test 5.

---

### Test 4 - Convert PDF (Smart Router)

`POST /convert-pdf`:

- `file`: Upload any PDF that contains a visual diagram (flowchart, UML
  diagram, circuit, or similar).
- `google_api_key`: Enter your key.
- Leave all other fields at their defaults.

**Expected (text-only PDF):**
```
[PDF] Smart Router: 0 images detected. Using fast-track text extraction.
```

**Expected (PDF with diagrams):**
```
[PDF] Smart Router: 2 images detected. Escalating to Gemini File API...
[PDF] Gemini File API successfully extracted 4,213 chars with image descriptions.
```

When you listen to the audio, you will hear Gemini's descriptions of the
diagrams woven seamlessly into the spoken text.

---

### Test 5 - Stream the audio

Once a job from Test 3 or Test 4 shows `state: completed` in `/jobs`,
open this URL in your browser:

```
http://127.0.0.1:8011/audio/YOUR_JOB_ID
```

---

### Test 6 - Verify job states

`GET /jobs?student_id=23CE034397` -- confirm both completed jobs appear
with `state: completed`, `char_count` populated, and `audio_path` set.

---

### Test 7 - Title resolution

`POST /convert-pdf` twice with `CSC332_Lecture5.pdf`:

- First call: leave `title` as the default `"Uploaded Note"`.
  Expected title in response: `"CSC332_Lecture5"`
- Second call: set `title` to `"Midterm Review"`.
  Expected title in response: `"Midterm Review - CSC332_Lecture5"`

---

## Understanding the Responses

### Why the API returns immediately

Audio synthesis for a full set of lecture notes takes 15 to 60 seconds.
A synchronous endpoint would time out in most HTTP clients. The service
uses FastAPI `BackgroundTasks` to run the pipeline after the response is
sent. Poll `GET /audio/{job_id}` -- it returns `202` while processing and
`200` with the MP3 stream when done.

### The `202 Accepted` response on `/audio/{job_id}`

This is not an error. It means the job is still running. Wait a few seconds
and try again. The terminal shows when processing completes.

### Why Edge-TTS requires internet

`edge-tts` streams synthesis from Microsoft Azure's servers. It requires
an active internet connection. If you are offline, Stage 2 fails and the
job moves to `failed` state with the error message stored.

### The Kafka mock event

After every successful conversion you will see in the terminal:

```
[KAFKA MOCK] Published event 'audio.generated' - job: 'a3f1b2c4d5e6f708'
```

In production this event notifies the student and feeds the Insight Engine.
During development it is a log-only simulation.

---

## Available Voices

All voices are Microsoft Azure Neural voices accessed via `edge-tts` at
no cost.

| Key | Voice Name | Language | Gender |
|---|---|---|---|
| `nigerian_female` | Ezinne | English (Nigeria) | Female |
| `nigerian_male` | Abeo | English (Nigeria) | Male |
| `british_female` | Sonia | English (UK) | Female |
| `british_male` | Ryan | English (UK) | Male |
| `american_female` | Aria | English (US) | Female |
| `american_male` | Christopher | English (US) | Male |

The default is `nigerian_female` (Ezinne). For a slower pace on complex
material, set `speaking_rate` to `0.85`. For a faster review session,
use `1.25`.

---

## Generated Files

The following are created at runtime. Do not commit them to version control.

| File / Folder | What it is |
|---|---|
| `audio_output/` | Folder containing all generated MP3 files |
| `audio_output/*.mp3` | One MP3 per completed conversion job |
| `akadverse_audio.db` | SQLite job tracking database |

**Suggested `.gitignore`:**

```
audio_output/
akadverse_audio.db
__pycache__/
*.pyc
.env
venv/
.vscode/
```

---

## Common Errors and Fixes

**`ModuleNotFoundError: No module named 'fitz'`**

PyMuPDF is not installed.

```bash
pip install pymupdf
```

**`ModuleNotFoundError: No module named 'multipart'`**

Required by FastAPI for file upload handling.

```bash
pip install python-multipart
```

**`ModuleNotFoundError: No module named 'edge_tts'`**

```bash
pip install edge-tts
```

**`422 Unprocessable Entity` on `/convert-pdf`**

The `google_api_key` field is missing or blank. It is a required form
field on the PDF endpoint. Enter your key in the Swagger UI form.

**Edge-TTS synthesis failed / connection error**

You are offline or Microsoft's Azure endpoint is temporarily unreachable.
Edge-TTS requires an active internet connection. Resubmit the job once
your connection is restored.

**Job stays in `processing` state indefinitely**

A background task may have silently failed. Check the terminal for
`[Pipeline ERROR]` lines. Common causes are a Gemini quota error during
Stage 1 or a network error during Edge-TTS. The job moves to `failed`
state with the error message stored once the exception is caught.

**`Address already in use` on startup**

Port 8011 is occupied. Run on a different port:

```bash
uvicorn note_to_audio_converter:app --host 127.0.0.1 --port 8012 --reload
```

---

## Project Structure

```
akadverse-note-to-audio-converter/
|
|-- note_to_audio_converter.py   # Main microservice - Smart Router and TTS pipeline
|-- requirements.txt             # Python dependencies
|-- README.md                    # This file
|-- .gitignore
|
|-- audio_output/                # Generated on first run - DO NOT COMMIT
|   `-- {job_id}_{title}_{timestamp}.mp3
|
`-- akadverse_audio.db           # Generated on first run - DO NOT COMMIT
```

---

*AkadVerse AI Architecture v1.0*