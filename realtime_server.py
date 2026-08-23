import asyncio
import base64
import json
import os
import numpy as np
from fastapi import FastAPI, WebSocket, Request, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
import uvicorn
import logging
from logging.handlers import RotatingFileHandler
from prompts import PROMPTS
from openai_realtime_client import OpenAIRealtimeAudioTextClient
from starlette.websockets import WebSocketState
import wave
import datetime
import time
import scipy.signal
from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel, Field
from typing import Generator
from llm_processor import get_llm_processor
from datetime import datetime, timedelta
import soundfile as sf
import io
import re
import tempfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        RotatingFileHandler(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "realtime_server.log"),
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger(__name__)

# Get recordings directory from environment variable or use default
RECORDINGS_DIR = os.getenv("BRAINWAVE_RECORDINGS_DIR", "recordings")

# Validate and create recordings directory
try:
    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    # Test write permissions
    test_file = os.path.join(RECORDINGS_DIR, "test_write_permission")
    with open(test_file, 'w') as f:
        f.write("test")
    os.remove(test_file)
    logger.info(f"Using recordings directory: {os.path.abspath(RECORDINGS_DIR)}")
except Exception as e:
    logger.error(f"Error setting up recordings directory {RECORDINGS_DIR}: {str(e)}")
    # Only log the error, do not raise or stop the server
    logger.error(f"Warning: Cannot access or write to recordings directory: {RECORDINGS_DIR}. Audio and transcript saving will be disabled.")

TIMESTAMP_PATTERN = re.compile(r"\d{8}_\d{6}")


def extract_time_tag_from_filename(filename: str) -> str:
    """Extract timestamp token from a filename or fallback to current time."""
    base_name = os.path.splitext(os.path.basename(filename))[0]
    match = TIMESTAMP_PATTERN.search(base_name)
    if match:
        return match.group(0)
    return datetime.now().strftime("%Y%m%d_%H%M%S")

# Pydantic models for request and response schemas
class ReadabilityRequest(BaseModel):
    text: str = Field(..., description="The text to improve readability for.")

class ReadabilityResponse(BaseModel):
    enhanced_text: str = Field(..., description="The text with improved readability.")

class CorrectnessRequest(BaseModel):
    text: str = Field(..., description="The text to check for factual correctness.")

class CorrectnessResponse(BaseModel):
    analysis: str = Field(..., description="The factual correctness analysis.")

class AskAIRequest(BaseModel):
    text: str = Field(..., description="The question to ask AI.")

class AskAIResponse(BaseModel):
    answer: str = Field(..., description="AI's answer to the question.")

app = FastAPI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY is not set in environment variables.")
    raise EnvironmentError("OPENAI_API_KEY is not set.")

# Xiaomi MiMo ASR (shadow testing). OpenAI-compatible API; docs:
# https://mimo.mi.com/models/en-US/mimo-v2.5-asr
# Optional: when the key is absent the shadow transcription is skipped silently.
MIMO_API_KEY = os.getenv("MIMO_API_KEY") or os.getenv("XIAOMI_ASR_API_KEY")
MIMO_BASE_URL = os.getenv("MIMO_BASE_URL", "https://api.xiaomimimo.com/v1")
MIMO_ASR_MODEL = os.getenv("MIMO_ASR_MODEL", "mimo-v2.5-asr")
if MIMO_API_KEY:
    logger.info("MiMo ASR shadow transcription enabled")
else:
    logger.info("MIMO_API_KEY not set; MiMo ASR shadow transcription disabled")

# Aliyun Paraformer (second shadow). File-transcription API via the DashScope
# SDK; docs: https://help.aliyun.com/zh/isi/developer-reference/api-details
# Optional: skipped when the key or the dashscope package is missing.
PARAFORMER_API_KEY = os.getenv("QWEN_Prepaid_10RMB_per_Month") or os.getenv("DASHSCOPE_API_KEY")
PARAFORMER_ASR_MODEL = os.getenv("PARAFORMER_ASR_MODEL", "paraformer-mtl-v1")
try:
    from dashscope.audio.asr import Transcription as DashscopeTranscription
    from dashscope.utils.oss_utils import check_and_upload_local as dashscope_upload_local
    import requests as dashscope_requests
    DASHSCOPE_SDK_AVAILABLE = True
except ImportError:
    DASHSCOPE_SDK_AVAILABLE = False
if PARAFORMER_API_KEY and DASHSCOPE_SDK_AVAILABLE:
    logger.info(f"Paraformer shadow transcription enabled ({PARAFORMER_ASR_MODEL})")
elif PARAFORMER_API_KEY:
    logger.info("dashscope package not installed; Paraformer shadow disabled (uv pip install dashscope)")
else:
    logger.info("No DashScope API key; Paraformer shadow transcription disabled")

# Aliyun qwen3-asr-flash-realtime (third shadow). Streaming WebSocket ASR, used
# here in BATCH-AT-STOP mode: we replay the buffered audio through the socket
# after Stop and collect only the FINAL (committed) segments — never the
# revisable partials — so the user sees one clean, corrected transcript.
# Docs: https://help.aliyun.com/zh/model-studio/qwen3-asr-flash-realtime
# Shares the DashScope key with Paraformer. ~¥0.02/min (far cheaper than
# OpenAI realtime). Optional: skipped when the key is missing.
QWEN_REALTIME_API_KEY = PARAFORMER_API_KEY
QWEN_REALTIME_MODEL = os.getenv("QWEN_REALTIME_ASR_MODEL", "qwen3-asr-flash-realtime")
# Default host is the standard DashScope realtime endpoint; the docs also show a
# workspace-scoped form (wss://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/...).
QWEN_REALTIME_WS_URL = os.getenv(
    "QWEN_REALTIME_WS_URL",
    "wss://dashscope.aliyuncs.com/api-ws/v1/realtime",
)
if QWEN_REALTIME_API_KEY:
    logger.info(f"Qwen realtime shadow transcription enabled ({QWEN_REALTIME_MODEL})")
else:
    logger.info("No DashScope API key; Qwen realtime shadow transcription disabled")

# Initialize with a default model
llm_processor = get_llm_processor("gpt-4o")  # Default processor

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    # Browsers request this automatically; return 204 instead of a noisy 404.
    from fastapi.responses import Response
    return Response(status_code=204)


@app.get("/", response_class=HTMLResponse)
async def get_realtime_page(request: Request):
    return FileResponse("static/realtime.html")

class AudioProcessor:
    def __init__(self, target_sample_rate=24000):
        self.target_sample_rate = target_sample_rate
        self.source_sample_rate = 48000  # Most common sample rate for microphones
        self.current_session_id = None
        self.current_transcription = []
        self.audio_buffer = []  # Add audio buffer as instance variable
        self.current_filename = None  # Cache for generated filename
        self._transcript_header_line = "下面是语音识别转录结果："
        self._expected_header = f"{self._transcript_header_line}\n\n"
        self._header_buffer = ""
        self._header_removed = False
        self.session_active = False
        self.saved_paths = {}
        self.latest_audio_path = None
        self.latest_transcription_path = None
        self.transcription_model = None  # Which model produced current_transcription
        self.transcription_elapsed = None  # End-to-end seconds the model took

    def process_audio_chunk(self, audio_data):
        # Convert binary audio data to Int16 array
        pcm_data = np.frombuffer(audio_data, dtype=np.int16)

        # Convert to float32 for better precision during resampling
        float_data = pcm_data.astype(np.float32) / 32768.0

        # Resample from 48kHz to 24kHz
        resampled_data = scipy.signal.resample_poly(
            float_data,
            self.target_sample_rate,
            self.source_sample_rate
        )

        # Convert back to int16 while preserving amplitude
        resampled_int16 = (resampled_data * 32768.0).clip(-32768, 32767).astype(np.int16)
        processed_audio = resampled_int16.tobytes()

        # Store the processed audio in our buffer
        self.audio_buffer.append(processed_audio)

        return processed_audio

    def start_new_session(self):
        """Start a new recording session with a unique timestamp-based ID"""
        self.current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_transcription = []
        self.audio_buffer = []  # Clear audio buffer for new session
        self.current_filename = None  # Reset cached filename for new session
        self._header_buffer = ""
        self._header_removed = False
        self.session_active = True
        self.saved_paths = {}
        self.latest_audio_path = None
        self.latest_transcription_path = None
        self.transcription_model = None
        self.transcription_elapsed = None
        return self.current_session_id

    def has_active_session(self):
        return self.session_active and self.current_session_id is not None

    def end_session(self):
        self.session_active = False
        self.current_session_id = None
        self.current_transcription = []
        self.audio_buffer = []
        self.current_filename = None
        self._header_buffer = ""
        self._header_removed = False
        self.saved_paths = {}
        self.latest_audio_path = None
        self.latest_transcription_path = None
        self.transcription_model = None
        self.transcription_elapsed = None

    def add_transcription_text(self, text):
        """Add transcription text to the current session"""
        if not self.current_session_id:
            logger.warning(f"No session ID when trying to add text: '{text}'")
            return text

        cleaned_text = self._strip_transcript_header(text)
        if cleaned_text:
            self.current_transcription.append(cleaned_text)
            logger.debug(f"Added text to transcription: '{cleaned_text}', total pieces: {len(self.current_transcription)}")
        return cleaned_text

    @staticmethod
    def strip_transcript_header(text: str) -> str:
        """Remove preamble line (contains 转录结果) and the blank line after it."""
        lines = text.split('\n')
        if lines and "转录结果" in lines[0]:
            rest = '\n'.join(lines[1:])
            return rest.lstrip('\r\n')
        return text

    def _strip_transcript_header(self, text: str) -> str:
        """Buffer until first non-empty line; discard it if it contains 转录结果, then strip leading blank lines."""
        if self._header_removed:
            return text

        combined = f"{self._header_buffer}{text}"
        # Skip any leading blank lines before checking the first real line
        stripped_start = combined.lstrip("\r\n")

        if not stripped_start:
            self._header_buffer = combined
            return ""

        newline_pos = stripped_start.find('\n')

        if newline_pos == -1:
            self._header_buffer = combined
            return ""

        first_line = stripped_start[:newline_pos]
        rest = stripped_start[newline_pos + 1:]
        self._header_removed = True
        self._header_buffer = ""

        if "转录结果" in first_line:
            return rest.lstrip("\r\n")
        return stripped_start

    def generate_content_filename(self, text_content):
        """Generate a descriptive filename from transcribed text content, caching the result for reuse."""
        if self.current_filename:
            return self.current_filename
        logger.info(f"Generating content filename for text: {text_content[:100]}...")

        if not text_content or len(text_content.strip()) < 10:
            # Fallback for very short content
            fallback_name = f"{self.current_session_id}_recording-too-short"
            logger.info(f"Text too short, using fallback name: {fallback_name}")
            self.current_filename = fallback_name
            return fallback_name

        try:
            # Use LLM to generate a short, descriptive title
            llm_processor = get_llm_processor("gpt-4o")
            prompt = (
                "Generate a short, descriptive filename (max 5-8 words) for this transcribed content. "
                "Start with the main topic or keywords (nouns), followed by verbs or clarifying words if needed. "
                "Do NOT include numbers, bullet points, or formatting at the beginning. "
                "Use only alphanumeric characters and underscores. "
                "Return only the filename, no quotes or extra text."
            )

            # Get the first 500 characters to avoid too long prompts
            content_sample = text_content[:500]
            if len(text_content) > 500:
                content_sample += "..."

            full_prompt = f"{prompt}\n\nContent: {content_sample}"
            logger.info(f"Generating filename with prompt: {full_prompt[:200]}...")

            # Generate filename synchronously
            filename = llm_processor.process_text_sync(full_prompt, "", model="gpt-4o")
            logger.info(f"Raw LLM response for filename: '{filename}'")

            # Clean up the filename
            filename = filename.strip().strip('"').strip("'")
            # Remove any leading numbers, dots, or bullet points
            filename = re.sub(r'^[\d\.\-\s]+', '', filename)
            # Replace spaces and special characters with underscores
            filename = re.sub(r'[^\w\s-]', '', filename)
            filename = re.sub(r'[\s-]+', '_', filename)
            filename = filename.lower()

            # Limit length and ensure it's not empty
            if len(filename) > 50:
                filename = filename[:50]
            if not filename:
                filename = "transcription"

            final_name = f"{self.current_session_id}_{filename}"
            logger.info(f"Final generated filename: {final_name}")
            self.current_filename = final_name
            return final_name

        except Exception as e:
            logger.error(f"Error generating content filename: {str(e)}", exc_info=True)
            # Fallback to timestamp with generic label
            fallback_name = f"{self.current_session_id}_transcription"
            logger.info(f"Using fallback filename: {fallback_name}")
            self.current_filename = fallback_name
            return fallback_name

    def save_audio_buffer(self, session_id=None, strategy="content"):
        """Save the audio buffer as a WAV file.

        strategy: 'content' for descriptive filenames, 'timestamp' for fail-safe saves.
        """
        if strategy not in ("content", "timestamp"):
            raise ValueError(f"Unknown save strategy: {strategy}")

        if not session_id:
            session_id = self.current_session_id

        if not session_id:
            logger.warning("No session ID provided for audio save")
            return

        if not self.audio_buffer:
            logger.warning("No audio data to save")
            return

        filename = session_id
        if strategy == "content":
            full_text = ''.join(self.current_transcription)
            logger.info(f"Full transcription text for audio save: {full_text[:200]}...")
            logger.info(f"Transcription length: {len(full_text)} characters")
            filename = self.generate_content_filename(full_text)

        wav_path = os.path.join(RECORDINGS_DIR, f"{filename}.wav")

        existing_path = self.saved_paths.get(strategy)
        if existing_path:
            if os.path.exists(existing_path):
                logger.info(f"Audio already saved using strategy '{strategy}' at {existing_path}, skipping duplicate write")
                return existing_path
            logger.warning(f"Previously saved path {existing_path} missing, rewriting audio file")

        logger.info(f"Saving audio with filename: {filename}")
        logger.info(f"Full audio path: {wav_path}")

        self._write_audio_file(wav_path)

        self.saved_paths[strategy] = wav_path
        self.latest_audio_path = wav_path
        logger.info(f"Saved audio recording to {wav_path}")
        return wav_path

    def _write_audio_file(self, wav_path):
        with wave.open(wav_path, 'wb') as wf:
            wf.setnchannels(1)  # Mono audio
            wf.setsampwidth(2)  # 2 bytes per sample (16-bit)
            wf.setframerate(self.target_sample_rate)
            wf.writeframes(b''.join(self.audio_buffer))

    def save_transcription(self, session_id=None):
        """Save the transcription as a text file with content-based naming and UTF-8-BOM encoding"""
        if not session_id:
            session_id = self.current_session_id

        if not session_id or not self.current_transcription:
            logger.warning("No session ID or transcription available")
            return

        # Generate content-based filename
        full_text = ''.join(self.current_transcription)
        logger.info(f"Full transcription text for text save: {full_text[:200]}...")
        logger.info(f"Transcription length: {len(full_text)} characters")

        filename = self.generate_content_filename(full_text)
        txt_path = os.path.join(RECORDINGS_DIR, f"{filename}.txt")

        logger.info(f"Saving transcription with filename: {filename}")
        logger.info(f"Full text path: {txt_path}")

        elapsed_note = f" in {self.transcription_elapsed:.1f}s" if self.transcription_elapsed else ""
        model_note = f"\n\n---\n_Transcribed by: {self.transcription_model or 'unknown'}{elapsed_note}_\n"
        with open(txt_path, 'wb') as f:  # Open in binary mode
            # Write UTF-8 BOM
            f.write(b'\xef\xbb\xbf')
            # Write content encoded as UTF-8
            f.write(full_text.encode('utf-8'))
            # Footer noting which model produced the transcript
            f.write(model_note.encode('utf-8'))
        logger.info(f"Saved transcription to {txt_path} with UTF-8-BOM encoding (model: {self.transcription_model})")
        self.latest_transcription_path = txt_path
        return txt_path

    def cleanup_timestamp_backup(self):
        """Remove the pure timestamp WAV once descriptive copies exist."""
        timestamp_path = self.saved_paths.get("timestamp")
        content_path = self.saved_paths.get("content")
        transcription_path = self.latest_transcription_path

        if not timestamp_path:
            return

        if (
            content_path and os.path.exists(content_path)
            and transcription_path and os.path.exists(transcription_path)
        ):
            try:
                os.remove(timestamp_path)
                logger.info(f"Removed fail-safe timestamp recording {timestamp_path} after successful save")
                self.saved_paths.pop("timestamp", None)
            except Exception as e:
                logger.error(f"Failed to remove timestamp recording {timestamp_path}: {e}", exc_info=True)

# Shared vocabulary hint for the transcription endpoint. Deliberately does NOT
# pin a language: the user speaks English, Mandarin, or a code-switched mix, so
# we let gpt-4o-transcribe auto-detect and keep each language as spoken.
# Bad-roll detection for gpt-4o-transcribe. Thresholds calibrated on 332
# historical AudioWrite clips >= 10s (see analysis_tmp/chars_per_sec.py):
# English-only transcripts run ~10.8 chars/sec median (p10 = 7.2), while
# mixed Chinese/English run ~3.6 (p10 = 2.1) because CJK packs a word per
# character. So the plausibility floor depends on the script of the returned
# text: below it, the model very likely dropped part of the audio. False
# positives just cost one extra API call. Short clips are exempt — a 3s
# "LLM" is legitimate.
MIN_CHARS_PER_SEC_ENGLISH = 5.0
MIN_CHARS_PER_SEC_CJK = 2.0
MIN_RETRY_DURATION_SEC = 10.0
_CJK_RE = re.compile(r'[一-鿿]')


def min_plausible_transcript_chars(text: str, duration_sec: float) -> float:
    """Minimum plausible transcript length for a clip, by script of `text`."""
    rate = MIN_CHARS_PER_SEC_CJK if _CJK_RE.search(text) else MIN_CHARS_PER_SEC_ENGLISH
    return duration_sec * rate

TRANSCRIBE_HINT = (
    "Mandarin and/or English speech, possibly code-switched, may include "
    "technical terms and product names (e.g., LLM, GPT, agent, DB, Cursor). "
    "Transcribe verbatim in the original language(s); do not translate."
)




async def transcribe_with_rest_api(audio_data: bytes, websocket: WebSocket, audio_processor: AudioProcessor):
    """Transcribe audio via REST using gpt-4o-transcribe (non-streaming).

    We deliberately do NOT stream: streaming emits text token-by-token, and a
    multi-byte UTF-8 character (e.g. Chinese, 3 bytes) split across two delta
    chunks gets decoded as a U+FFFD replacement character. Non-streaming decodes
    the whole response as one complete UTF-8 string, so the transcript is clean.
    Restful mode is batch anyway (audio is buffered and sent only after Stop),
    so streaming provided no real latency benefit.
    """
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    # gpt-4o-transcribe occasionally returns only a fragment of a longer clip
    # (nondeterministic; re-sending the identical audio usually recovers the
    # full text). We know the clip duration from the PCM byte count, so a
    # transcript far shorter than plausible speech density flags a bad roll:
    # retry once (two tries total) and keep the longer transcript. A separate,
    # stronger detector (shadow cross-check) plus a structural recovery (chunked
    # re-transcription) handles the silent TRAILING-segment drop that the
    # chars/sec floor cannot see; see the mitigation notes above.
    duration_sec = len(audio_data) / (24000 * 2)
    t_start = time.monotonic()

    async def _transcribe_pcm(pcm: bytes) -> str:
        """Transcribe one PCM buffer (16-bit mono @ 24kHz) via gpt-4o-transcribe,
        managing its own temp WAV. Reused for the full clip and for each chunk."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            tmp_path = f.name
            with wave.open(f, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(24000)
                wf.writeframes(pcm)
        try:
            with open(tmp_path, 'rb') as audio_file:
                response = await client.audio.transcriptions.create(
                    model="gpt-4o-transcribe",
                    file=audio_file,
                    response_format="text",
                    prompt=TRANSCRIBE_HINT,
                )
            # response_format="text" returns the transcript string directly
            return (response if isinstance(response, str) else getattr(response, "text", "")).strip()
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    try:
        logger.info(f"Calling REST API transcription ({duration_sec:.1f}s)")

        full_text = await _transcribe_pcm(audio_data)
        logger.info(f"REST API transcription complete, length: {len(full_text)}")
        model_label = "gpt-4o-transcribe"

        # Detector 1 (existing): chars/sec floor -> one identical re-run.
        min_plausible_chars = min_plausible_transcript_chars(full_text, duration_sec)
        if duration_sec >= MIN_RETRY_DURATION_SEC and len(full_text) < min_plausible_chars:
            logger.warning(
                f"Transcript suspiciously short ({len(full_text)} chars for "
                f"{duration_sec:.1f}s audio, expected >= {min_plausible_chars:.0f}); retrying once"
            )
            try:
                retry_text = await _transcribe_pcm(audio_data)
                logger.info(f"Retry transcription complete, length: {len(retry_text)}")
                if len(retry_text) > len(full_text):
                    full_text = retry_text
            except Exception as e:
                logger.error(f"Retry transcription failed; keeping first result: {e}", exc_info=True)

        if full_text:
            audio_processor.current_transcription = [full_text]
            audio_processor.transcription_model = model_label
            audio_processor.transcription_elapsed = time.monotonic() - t_start
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_text(json.dumps({
                    "type": "text",
                    "content": full_text,
                    "isNewResponse": False
                }))
        return full_text

    except Exception as e:
        logger.error(f"Error in REST API transcription: {e}", exc_info=True)
        raise


def pcm_to_wav_bytes(pcm_audio: bytes, sample_rate: int = 24000) -> bytes:
    """Wrap raw 16-bit mono PCM into an in-memory WAV container (no temp file)."""
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_audio)
    return buf.getvalue()


# MiMo rejects audio whose decoded bytes exceed 10 MB (observed: HTTP 400
# 'input_audio.data exceeds maximum size of 10MB'; the docs claim the limit is
# on the base64 string, but 13 MB base64 payloads went through — the enforced
# cap is on the decoded audio). Raw 24 kHz WAV hits it at ~3.5 min, so we
# transcode to MP3 (~6x smaller), lifting the ceiling to ~20+ min — past the
# ~10-15 min where MiMo's 2K output-token cap truncates transcripts anyway.
MIMO_MAX_AUDIO_BYTES = 10 * 1024 * 1024
# Per-request chunk length. MiMo's 2K output-token cap truncates transcripts
# past ~10-15 min of dense speech, so ~10 min chunks keep every chunk's
# transcript complete; the merged text is still written as ONE txt file.
MIMO_CHUNK_SECONDS = 600


def _mimo_prepare_chunks(wav_bytes: bytes):
    """Split WAV into <=MIMO_CHUNK_SECONDS pieces and encode each to MP3
    (blocking; run via to_thread). Returns a list of (audio_bytes, format).
    Falls back to the original WAV as a single chunk if decoding fails."""
    try:
        data, samplerate = sf.read(io.BytesIO(wav_bytes), dtype='int16')
    except Exception as e:
        logger.warning(f"MiMo chunk prep: WAV decode failed ({e}); sending WAV as-is")
        if len(wav_bytes) > MIMO_MAX_AUDIO_BYTES:
            logger.warning(f"WAV exceeds MiMo 10MB cap ({len(wav_bytes)} bytes); shadow skipped")
            return []
        return [(wav_bytes, "wav")]

    chunks = []
    samples_per_chunk = MIMO_CHUNK_SECONDS * samplerate
    for start in range(0, len(data), samples_per_chunk):
        segment = data[start:start + samples_per_chunk]
        buf = io.BytesIO()
        try:
            sf.write(buf, segment, samplerate, format='MP3')
        except Exception as e:
            logger.warning(f"MP3 encode failed ({e}); using WAV for this chunk")
            buf = io.BytesIO()
            sf.write(buf, segment, samplerate, format='WAV', subtype='PCM_16')
            audio = buf.getvalue()
            if len(audio) > MIMO_MAX_AUDIO_BYTES:
                logger.warning(f"WAV chunk exceeds MiMo 10MB cap ({len(audio)} bytes); chunk dropped")
                continue
            chunks.append((audio, "wav"))
            continue
        audio = buf.getvalue()
        if len(audio) > MIMO_MAX_AUDIO_BYTES:
            logger.warning(f"MP3 chunk exceeds MiMo 10MB cap ({len(audio)} bytes); chunk dropped")
            continue
        chunks.append((audio, "mp3"))

    total = sum(len(a) for a, _ in chunks)
    logger.info(
        f"MiMo audio prepared: {len(wav_bytes)} WAV bytes -> "
        f"{len(chunks)} chunk(s), {total} bytes total"
    )
    return chunks


async def _mimo_chat(audio_bytes: bytes, fmt: str) -> str:
    """One MiMo chat-completions call for one audio chunk."""
    audio_b64 = base64.b64encode(audio_bytes).decode('ascii')
    client = AsyncOpenAI(api_key=MIMO_API_KEY, base_url=MIMO_BASE_URL)
    response = await client.chat.completions.create(
        model=MIMO_ASR_MODEL,
        messages=[{
            "role": "user",
            "content": [{
                "type": "input_audio",
                "input_audio": {"data": audio_b64, "format": fmt},
            }],
        }],
        extra_body={"language": "auto"},
    )
    return (response.choices[0].message.content or "").strip()


async def mimo_transcribe(wav_bytes: bytes) -> str:
    """Transcribe WAV bytes with Xiaomi MiMo ASR; returns text ('' on empty/error).

    MiMo exposes an OpenAI-compatible chat-completions API where the audio is
    sent as base64 `input_audio` (docs: mimo.mi.com/models/en-US/mimo-v2.5-asr).
    Audio is transcoded to MP3 and, past MIMO_CHUNK_SECONDS, split into chunks
    transcribed concurrently — the pieces are merged into one transcript.
    """
    if not MIMO_API_KEY:
        return ""
    try:
        chunks = await asyncio.to_thread(_mimo_prepare_chunks, wav_bytes)
        if not chunks:
            return ""
        results = await asyncio.gather(
            *(_mimo_chat(audio, fmt) for audio, fmt in chunks),
            return_exceptions=True,
        )
        texts = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                logger.error(f"MiMo chunk {i + 1}/{len(chunks)} failed: {r}")
                texts.append(f"[chunk {i + 1} failed]")
            elif r:
                texts.append(r)
        # All-failed => treat as no transcript rather than a file of markers
        if not any(not t.startswith("[chunk") for t in texts):
            return ""
        return "\n".join(texts).strip()
    except Exception as e:
        logger.error(f"MiMo ASR shadow transcription failed: {e}", exc_info=True)
        return ""


def paraformer_enabled() -> bool:
    return bool(PARAFORMER_API_KEY and DASHSCOPE_SDK_AVAILABLE)


def qwen_realtime_enabled() -> bool:
    return bool(QWEN_REALTIME_API_KEY)


async def _close_ws_quietly(ws):
    """Background goodbye for a WebSocket we've already finished with."""
    try:
        await ws.close()
    except Exception:
        pass


async def qwen_realtime_transcribe(pcm_24k: bytes) -> str:
    """Batch-at-Stop transcription via qwen3-asr-flash-realtime (WebSocket).

    Replays the buffered 24 kHz PCM (resampled to the 16 kHz the model expects)
    through the realtime socket in manual-commit mode, then commits + finishes.
    Only `...transcription.completed` (FINAL) events are collected; the
    revisable `...transcription.text` partials are ignored on purpose, so the
    caller gets one clean, corrected transcript rather than flickering drafts.
    Returns '' on empty/error (never raises) so it's safe as a shadow.
    """
    if not QWEN_REALTIME_API_KEY or not pcm_24k:
        return ""
    import websockets  # local import: optional dependency of the shadow path
    try:
        # 24 kHz int16 -> 16 kHz int16 (model's expected rate)
        x = np.frombuffer(pcm_24k, dtype=np.int16).astype(np.float32) / 32768.0
        y = scipy.signal.resample_poly(x, 16000, 24000)
        pcm_16k = (y * 32768.0).clip(-32768, 32767).astype(np.int16).tobytes()

        url = f"{QWEN_REALTIME_WS_URL}?model={QWEN_REALTIME_MODEL}"
        headers = {
            "Authorization": f"Bearer {QWEN_REALTIME_API_KEY}",
            "OpenAI-Beta": "realtime=v1",
        }
        finals = []
        # websockets renamed the kwarg: `extra_headers` (<=13) -> `additional_headers` (>=14).
        import inspect
        hdr_kw = ("additional_headers"
                  if "additional_headers" in inspect.signature(websockets.connect).parameters
                  else "extra_headers")
        # Measured on a 13.5s clip: connect 0.06s, audio 0.06s, model 0.36s —
        # then a ~1.0s wait for the server to answer our close frame. Since we
        # already hold the full transcript at `session.finished`, we return
        # immediately and do the goodbye in the background (see `finally`).
        ws = await websockets.connect(url, open_timeout=15, max_size=None,
                                      close_timeout=0.2, **{hdr_kw: headers})
        try:
            await ws.send(json.dumps({
                "event_id": "evt_session",
                "type": "session.update",
                "session": {
                    "modalities": ["text"],
                    "input_audio_format": "pcm",
                    "sample_rate": 16000,
                    "turn_detection": None,  # manual mode: we commit explicitly
                },
            }))
            # Stream audio in ~100 ms chunks (16 kHz * 2 bytes * 0.1 s = 3200 B)
            chunk = 3200
            for i in range(0, len(pcm_16k), chunk):
                await ws.send(json.dumps({
                    "event_id": f"evt_a{i}",
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(pcm_16k[i:i + chunk]).decode("ascii"),
                }))
            await ws.send(json.dumps({"event_id": "evt_commit", "type": "input_audio_buffer.commit"}))
            await ws.send(json.dumps({"event_id": "evt_finish", "type": "session.finish"}))

            # Collect finals until the server says the session is finished.
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=30)
                except asyncio.TimeoutError:
                    logger.warning("Qwen realtime: no event for 30s; giving up")
                    break
                try:
                    ev = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    continue
                et = ev.get("type", "")
                if et == "conversation.item.input_audio_transcription.completed":
                    t = (ev.get("transcript") or "").strip()
                    if t:
                        finals.append(t)
                elif et in ("session.finished", "session.finish"):
                    break
                elif et == "error":
                    logger.error(f"Qwen realtime error event: {ev}")
                    break
        finally:
            # Don't block on the ~1s close handshake — hand it off.
            asyncio.create_task(_close_ws_quietly(ws))
        return " ".join(finals).strip()
    except Exception as e:
        logger.error(f"Qwen realtime shadow transcription failed: {e}", exc_info=True)
        return ""


def _paraformer_transcribe_sync(wav_bytes: bytes) -> str:
    """Blocking DashScope flow (the SDK is sync; run via asyncio.to_thread):
    temp WAV -> DashScope temp OSS bucket (readable only by this API key) ->
    async transcription task -> download transcript JSON."""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        f.write(wav_bytes)
        tmp_path = f.name
    try:
        _, file_url, _ = dashscope_upload_local(
            PARAFORMER_ASR_MODEL, f'file://{tmp_path}', PARAFORMER_API_KEY)
        task = DashscopeTranscription.async_call(
            model=PARAFORMER_ASR_MODEL,
            file_urls=[file_url],
            api_key=PARAFORMER_API_KEY,
            headers={'X-DashScope-OssResourceResolve': 'enable'},
        )
        result = DashscopeTranscription.wait(task=task.output.task_id, api_key=PARAFORMER_API_KEY)
        if result.status_code != 200:
            logger.error(f"Paraformer task failed: {result.status_code} {result.output}")
            return ""
        texts = []
        for r in result.output.get('results', []):
            if r.get('subtask_status') != 'SUCCEEDED':
                logger.error(f"Paraformer subtask failed: {json.dumps(r, ensure_ascii=False)}")
                continue
            detail = dashscope_requests.get(r['transcription_url'], timeout=30).json()
            texts += [t.get('text', '') for t in detail.get('transcripts', [])]
        return '\n'.join(t for t in texts if t).strip()
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# The DashScope SDK's OSS upload has no timeout — a stalled upload blocks the
# worker thread forever with zero log output (observed 2026-08-12: an 8.4 MB
# upload silently never completed). wait_for can't kill the thread, but it
# bounds our wait and makes the failure visible in the log.
PARAFORMER_TIMEOUT_SECONDS = 300


async def paraformer_transcribe(wav_bytes: bytes) -> str:
    """Transcribe WAV bytes with Aliyun paraformer; returns text ('' on error)."""
    if not paraformer_enabled() or not wav_bytes:
        return ""
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_paraformer_transcribe_sync, wav_bytes),
            timeout=PARAFORMER_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.error(
            f"Paraformer shadow timed out after {PARAFORMER_TIMEOUT_SECONDS}s "
            f"({len(wav_bytes)} bytes) — likely a stalled OSS upload"
        )
        return ""
    except Exception as e:
        logger.error(f"Paraformer shadow transcription failed: {e}", exc_info=True)
        return ""


async def _timed(coro):
    """Await `coro`, returning (result, elapsed_seconds). For shadows the timer
    starts at task creation (the Stop moment), so elapsed is true end-to-end."""
    t_start = time.monotonic()
    result = await coro
    return result, time.monotonic() - t_start


def write_shadow(text: str, reference_txt_path: str, suffix: str, model_label: str, elapsed: float = None):
    """Write a shadow transcript as `<base>_{suffix}.txt`, next to the primary
    transcript, using the (already renamed) descriptive filename."""
    if not text:
        logger.warning(f"{model_label} shadow produced no text; nothing written")
        return
    base, _ = os.path.splitext(reference_txt_path)
    shadow_path = f"{base}_{suffix}.txt"
    elapsed_note = f" in {elapsed:.1f}s" if elapsed else ""
    model_note = f"\n\n---\n_Transcribed by: {model_label} (shadow){elapsed_note}_\n"
    try:
        with open(shadow_path, 'wb') as f:
            f.write(b'\xef\xbb\xbf')
            f.write(text.encode('utf-8'))
            f.write(model_note.encode('utf-8'))
        logger.info(f"{model_label} shadow transcript saved to {shadow_path} ({len(text)} chars)")
        return shadow_path
    except Exception as e:
        logger.error(f"Failed to write shadow transcript to {shadow_path}: {e}", exc_info=True)
        return None


TRANSCRIPT_FOOTER_SEP = "\n\n---\n"


def rewrite_transcript_body(path: str, new_body: str) -> str:
    """Replace the body of an existing transcript .txt while PRESERVING its
    footer (`_Transcribed by: <model> in N.Ns_` — the time-spent record), and
    append/refresh an edit stamp. Returns the footer kept, for display."""
    raw = open(path, encoding='utf-8-sig').read()
    parts = raw.split(TRANSCRIPT_FOOTER_SEP, 1)
    footer = parts[1] if len(parts) == 2 else ""
    # If the client echoed the footer back inside the body, drop it there.
    new_body = new_body.split(TRANSCRIPT_FOOTER_SEP, 1)[0].rstrip()
    footer_lines = [ln for ln in footer.strip('\n').split('\n')
                    if ln.strip() and not ln.startswith('_Edited in browser')]
    footer_lines.append(f"_Edited in browser {datetime.now().strftime('%Y-%m-%d %H:%M')}_")
    with open(path, 'wb') as f:
        f.write(b'\xef\xbb\xbf')
        f.write(new_body.encode('utf-8'))
        f.write((TRANSCRIPT_FOOTER_SEP + '\n'.join(footer_lines) + '\n').encode('utf-8'))
    return '\n'.join(footer_lines)


def start_shadow_tasks(pcm_audio: bytes):
    """Kick off every enabled shadow transcription the moment audio is ready,
    in parallel with the primary (OpenAI) call. Returns a list of
    (task, suffix, model_label); files are written later, once the
    descriptive filename exists."""
    tasks = []
    if not pcm_audio:
        return tasks
    wav_bytes = None
    if MIMO_API_KEY or paraformer_enabled():
        wav_bytes = pcm_to_wav_bytes(pcm_audio)
    # Order matters: the browser renders one tab per task in this order and
    # selects the FIRST as the default tab — Qwen realtime goes first.
    if qwen_realtime_enabled():
        # Takes raw PCM directly (it resamples to 16 kHz itself).
        tasks.append((asyncio.create_task(_timed(qwen_realtime_transcribe(pcm_audio))),
                      "by_QwenRealtime", QWEN_REALTIME_MODEL))
    if MIMO_API_KEY:
        tasks.append((asyncio.create_task(_timed(mimo_transcribe(wav_bytes))),
                      "by_MiMoASR", MIMO_ASR_MODEL))
    if paraformer_enabled():
        tasks.append((asyncio.create_task(_timed(paraformer_transcribe(wav_bytes))),
                      "by_ParaformerMTL", PARAFORMER_ASR_MODEL))
    return tasks


def start_shadow_tasks_from_file(wav_path: str):
    """Late shadows for modes without a local PCM buffer (e.g. realtime):
    read the saved WAV and run every enabled shadow engine on it."""
    async def _from_file(fn):
        try:
            with open(wav_path, 'rb') as f:
                data = f.read()
        except Exception as e:
            logger.error(f"Shadow file read failed for {wav_path}: {e}", exc_info=True)
            return ""
        return await fn(data)

    tasks = []
    # Same order as start_shadow_tasks: Qwen realtime first (default tab).
    if qwen_realtime_enabled():
        async def _qwen_from_wav():
            # Qwen realtime wants raw PCM; strip the WAV container first.
            try:
                with wave.open(wav_path, 'rb') as wf:
                    pcm = wf.readframes(wf.getnframes())
            except Exception as e:
                logger.error(f"Qwen realtime shadow WAV read failed for {wav_path}: {e}", exc_info=True)
                return ""
            return await qwen_realtime_transcribe(pcm)
        tasks.append((asyncio.create_task(_timed(_qwen_from_wav())),
                      "by_QwenRealtime", QWEN_REALTIME_MODEL))
    if MIMO_API_KEY:
        tasks.append((asyncio.create_task(_timed(_from_file(mimo_transcribe))),
                      "by_MiMoASR", MIMO_ASR_MODEL))
    if paraformer_enabled():
        tasks.append((asyncio.create_task(_timed(_from_file(paraformer_transcribe))),
                      "by_ParaformerMTL", PARAFORMER_ASR_MODEL))
    return tasks


def shadow_engine_labels():
    """Names of the enabled shadow engines, in display order. The browser uses
    this to render one tab per engine (Qwen realtime first = default tab)."""
    labels = []
    if qwen_realtime_enabled():
        labels.append(QWEN_REALTIME_MODEL)
    if MIMO_API_KEY:
        labels.append(MIMO_ASR_MODEL)
    if paraformer_enabled():
        labels.append(PARAFORMER_ASR_MODEL)
    return labels


async def _push_shadow(websocket, model_label: str, text: str, elapsed):
    """Send one finished shadow transcript to the browser (fills its tab)."""
    try:
        if websocket is not None and websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_text(json.dumps({
                "type": "shadow",
                "model": model_label,
                "content": text or "",
                "elapsed": round(elapsed, 1) if elapsed else None,
                "empty": not bool(text),
            }))
    except Exception as e:
        logger.debug(f"Could not push shadow {model_label} to client: {e}")


def attach_shadow_pushers(tasks, websocket):
    """Push each shadow transcript to the browser the MOMENT its own engine
    finishes — deliberately decoupled from finalize_recording, which has to wait
    for the slower gpt-4o call + filename generation. Without this, a MiMo
    result ready at 0.7s sat invisible behind spinners until ~5s. Awaiting a
    task here does not consume it; finalize still awaits it for the file write."""
    async def _push_when_done(task, model_label):
        try:
            text, elapsed = await task
        except Exception as e:
            logger.error(f"{model_label} shadow failed before push: {e}", exc_info=True)
            text, elapsed = "", None
        await _push_shadow(websocket, model_label, text, elapsed)
    for task, _suffix, model_label in tasks:
        asyncio.create_task(_push_when_done(task, model_label))


async def _finish_shadow(task, reference_txt_path: str, suffix: str, model_label: str,
                         registry=None, websocket=None):
    """Await one in-flight shadow task and write its transcript once the
    descriptive filename is known. (Browser push is handled separately by
    attach_shadow_pushers, so it is never gated on this.) Records the written
    path in `registry[model_label]` and tells the browser, so its Save button
    can write edits back to the right file."""
    try:
        text, elapsed = await task
    except Exception as e:
        logger.error(f"{model_label} shadow await failed: {e}", exc_info=True)
        return
    path = write_shadow(text, reference_txt_path, suffix, model_label, elapsed)
    if path:
        if registry is not None:
            registry[model_label] = path
        try:
            if websocket is not None and websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_text(json.dumps({
                    "type": "shadow_saved", "model": model_label, "file": os.path.basename(path)}))
        except Exception as e:
            logger.debug(f"Could not send shadow_saved for {model_label}: {e}")


def finish_shadow_tasks(tasks, reference_txt_path, registry=None, websocket=None):
    """Detach file writers for all pending shadows (or cancel them when there
    is no reference filename to sit next to). Never blocks finalize."""
    for task, suffix, model_label in tasks:
        if reference_txt_path:
            asyncio.create_task(_finish_shadow(task, reference_txt_path, suffix, model_label, registry, websocket))
        else:
            task.cancel()


async def transcribe_wav_fallback(wav_path: str) -> str:
    """Auto-recovery: transcribe an already-saved WAV via gpt-4o-transcribe.

    Fires when the primary (live) transcription yields nothing — a parse miss,
    an empty model reply, or a mid-recording WebSocket drop that only left a
    fail-safe WAV on disk. This turns that WAV into a transcript automatically
    so the user never has to re-upload it by hand. gpt-4o-transcribe (not
    whisper-1) is used because it handles mixed Mandarin/English far better.
    """
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    with open(wav_path, 'rb') as audio_file:
        response = await client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=audio_file,
            response_format="text",
            prompt=TRANSCRIBE_HINT,
        )
    return (response if isinstance(response, str) else getattr(response, "text", "")).strip()


async def transcribe_with_audio15(audio_data: bytes, websocket: WebSocket, audio_processor: AudioProcessor):
    """Transcribe audio using Chat Completions with gpt-audio-1.5"""
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    t_start = time.monotonic()

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        tmp_path = f.name
        with wave.open(f, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(audio_data)

    try:
        with open(tmp_path, 'rb') as f:
            audio_b64 = base64.b64encode(f.read()).decode()

        logger.info("Calling gpt-audio-1.5 via Chat Completions for transcription")
        response = await client.chat.completions.create(
            model="gpt-audio-1.5",
            modalities=["text"],
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": audio_b64, "format": "wav"}
                    },
                    {
                        "type": "text",
                        "text": PROMPTS['paraphrase-gpt-realtime']
                    }
                ]
            }]
        )

        content = (response.choices[0].message.content or "").strip()
        finish_reason = response.choices[0].finish_reason
        audio_seconds = len(audio_data) / 2 / 24000  # 16-bit mono @ 24kHz
        # gpt-audio-1.5 often ignores the "plain text" instruction and wraps the
        # transcript in JSON — and the key varies ("transcription", "message",
        # "text", ...). Only checking "transcription" silently dropped complete
        # transcripts keyed as "message". Pull out whatever string it used.
        transcript_text = content
        try:
            parsed = json.loads(content)
            if isinstance(parsed, str):
                transcript_text = parsed
            elif isinstance(parsed, dict):
                candidates = [parsed[k] for k in ("transcription", "message", "text", "content")
                              if isinstance(parsed.get(k), str) and parsed[k].strip()]
                if not candidates:
                    candidates = [v for v in parsed.values() if isinstance(v, str) and v.strip()]
                if candidates:
                    transcript_text = candidates[0]
        except (json.JSONDecodeError, ValueError):
            transcript_text = content
        transcript_text = AudioProcessor.strip_transcript_header(transcript_text.strip())
        logger.info(
            f"gpt-audio-1.5 transcription complete, length: {len(transcript_text)}, "
            f"finish_reason: {finish_reason}, audio_seconds: {audio_seconds:.1f}"
        )
        if not transcript_text:
            logger.warning(
                f"gpt-audio-1.5 returned EMPTY transcript "
                f"(finish_reason={finish_reason}, audio_seconds={audio_seconds:.1f}, "
                f"raw_content={content!r}, usage={response.usage})"
            )
        if transcript_text:
            audio_processor.current_transcription = [transcript_text]
            audio_processor.transcription_model = "gpt-audio-1.5"
            audio_processor.transcription_elapsed = time.monotonic() - t_start
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_text(json.dumps({
                    "type": "text",
                    "content": transcript_text,
                    "isNewResponse": False
                }))
        return transcript_text

    except Exception as e:
        logger.error(f"Error in gpt-audio-1.5 transcription: {e}", exc_info=True)
        raise
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.websocket("/api/v1/ws")
async def websocket_endpoint(websocket: WebSocket):
    logger.info("New WebSocket connection attempt")
    await websocket.accept()
    logger.info("WebSocket connection accepted")

    # Add initial status update here
    await websocket.send_text(json.dumps({
        "type": "status",
        "status": "idle"  # Set initial status to idle (blue)
    }))

    client = None
    audio_processor = AudioProcessor()
    recording_stopped = asyncio.Event()
    openai_ready = asyncio.Event()
    pending_audio_chunks = []

    # Track transcription mode for current session
    current_mode = "realtime"  # Default mode
    restful_audio_buffer = []  # Buffer for RESTful mode audio
    pending_shadow_tasks = []  # In-flight shadow transcriptions (started at stop)
    saved_files = {}  # model label ('primary' / shadow label) -> transcript path written this session

    # Add synchronization for audio sending operations
    pending_audio_operations = 0
    audio_send_lock = asyncio.Lock()
    all_audio_sent = asyncio.Event()
    all_audio_sent.set()  # Initially set since no audio is pending
    finalize_lock = asyncio.Lock()


    async def initialize_openai():
        nonlocal client
        openai_ready.clear()

        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                client = OpenAIRealtimeAudioTextClient(os.getenv("OPENAI_API_KEY"))
                await client.connect()
                logger.info("Successfully connected to OpenAI client")

                # Register handlers after client is initialized
                client.register_handler("session.updated", lambda data: handle_generic_event("session.updated", data))
                client.register_handler("input_audio_buffer.cleared", lambda data: handle_generic_event("input_audio_buffer.cleared", data))
                client.register_handler("input_audio_buffer.speech_started", lambda data: handle_generic_event("input_audio_buffer.speech_started", data))
                client.register_handler("rate_limits.updated", lambda data: handle_generic_event("rate_limits.updated", data))
                client.register_handler("conversation.item.created", lambda data: handle_generic_event("conversation.item.created", data))
                client.register_handler("error", lambda data: handle_error(data))
                client.register_handler("conversation.item.input_audio_transcription.delta", lambda data: handle_transcription_delta(data))
                client.register_handler("conversation.item.input_audio_transcription.completed", lambda data: handle_transcription_completed(data))
                client.register_handler("conversation.item.input_audio_transcription.failed", lambda data: handle_transcription_failed(data))

                openai_ready.set()
                await websocket.send_text(json.dumps({
                    "type": "status",
                    "status": "connected"
                }))
                return True

            except Exception as e:
                logger.error(f"OpenAI connect attempt {attempt}/{max_attempts} failed: {e}", exc_info=(attempt == max_attempts))
                if client:
                    try:
                        await client.close()
                    except Exception:
                        pass
                    finally:
                        client = None

                if attempt < max_attempts:
                    retry_delay = 2 * attempt  # 2s, 4s
                    logger.info(f"Retrying in {retry_delay}s...")
                    await asyncio.sleep(retry_delay)
                else:
                    openai_ready.clear()
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "content": "Failed to initialize OpenAI connection"
                    }))
                    return False

    async def handle_transcription_delta(data):
        logger.debug(f"Transcription delta: '{data.get('delta', '')}'")

    async def handle_transcription_completed(data):
        raw_transcript = (data.get("transcript") or "").strip()
        logger.info(f"Transcription completed, raw length: {len(raw_transcript)}")
        if raw_transcript:
            try:
                fixed = await asyncio.to_thread(
                    llm_processor.process_text_sync,
                    raw_transcript,
                    PROMPTS['grammar-fix'],
                    "gpt-4o-mini"
                )
            except Exception as e:
                logger.error(f"Grammar fix failed: {e}, using raw transcript")
                fixed = raw_transcript
            audio_processor.current_transcription = [fixed]
            audio_processor.transcription_model = "gpt-realtime (+ gpt-4o-mini grammar-fix)"
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_text(json.dumps({
                    "type": "text",
                    "content": fixed,
                    "isNewResponse": False
                }))
        await finalize_recording(success=bool(raw_transcript), reason="transcription_completed")

    async def handle_error(data):
        error_obj = data.get("error", {})
        error_msg = error_obj.get("message", "Unknown error")
        error_type = error_obj.get("type", "unknown")
        error_code = error_obj.get("code", "")

        full_error = f"OpenAI API Error [{error_type}]: {error_msg}"
        if error_code:
            full_error += f" (code: {error_code})"

        logger.error(full_error)
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "content": full_error
            }))
        except Exception as e:
            logger.error(f"Failed to notify client about error: {e}", exc_info=True)
        logger.info("Handled error message from OpenAI")
        await finalize_recording(success=False, reason="openai_error")

    async def handle_transcription_failed(data):
        """Handle transcription failure events from OpenAI Realtime API"""
        error_obj = data.get("error", {})
        error_msg = error_obj.get("message", "Transcription failed")
        error_type = error_obj.get("type", "unknown")
        error_code = error_obj.get("code", "")

        # Build informative error message
        full_error = f"Transcription Failed [{error_type}]: {error_msg}"
        if error_code:
            full_error += f" (code: {error_code})"

        # Check for common error types and provide helpful context
        if "quota" in error_msg.lower() or "insufficient_quota" in error_code:
            full_error = f"OpenAI Quota Exceeded: {error_msg}. Please check your billing at https://platform.openai.com/account/billing"
        elif "rate" in error_msg.lower() or "rate_limit" in error_type:
            full_error = f"OpenAI Rate Limit: {error_msg}. Please wait and try again."

        logger.error(f"Transcription failed: {full_error}")
        logger.error(f"Full transcription failure data: {json.dumps(data, ensure_ascii=False)}")

        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "content": full_error
                }))
        except Exception as e:
            logger.error(f"Failed to notify client about transcription failure: {e}", exc_info=True)

        await finalize_recording(success=False, reason="transcription_failed")

    async def handle_generic_event(event_type, data):
        logger.info(f"Handled {event_type} with data: {json.dumps(data, ensure_ascii=False)}")

    async def finalize_recording(success=False, reason="", notify_client=True):
        nonlocal client, pending_audio_chunks, pending_audio_operations, restful_audio_buffer, pending_shadow_tasks

        async with finalize_lock:
            logger.info(f"Finalizing recording (success={success}, reason={reason})")
            if audio_processor.has_active_session():
                strategy = "content" if success else "timestamp"
                audio_path = None
                transcription_path = None
                try:
                    audio_path = audio_processor.save_audio_buffer(strategy=strategy)
                except Exception as e:
                    logger.error(f"Failed to save audio buffer during finalize: {e}", exc_info=True)
                try:
                    transcription_path = audio_processor.save_transcription()
                except Exception as e:
                    logger.error(f"Failed to save transcription during finalize: {e}", exc_info=True)

                # Auto-fallback: the live path produced no transcript (parse miss,
                # empty model reply, or a mid-recording socket drop) but a WAV was
                # saved. Recover it via gpt-4o-transcribe so a recording is never
                # silently lost and the user never has to re-upload by hand.
                if not transcription_path and audio_path and os.path.exists(audio_path):
                    try:
                        logger.info(f"No transcript from primary path; auto-fallback on {audio_path}")
                        t_fallback = time.monotonic()
                        recovered = await transcribe_wav_fallback(audio_path)
                        if recovered:
                            audio_processor.current_transcription = [recovered]
                            audio_processor.transcription_model = "gpt-4o-transcribe (auto-fallback)"
                            audio_processor.transcription_elapsed = time.monotonic() - t_fallback
                            audio_processor.current_filename = None  # regenerate a descriptive name
                            transcription_path = audio_processor.save_transcription()
                            # Rename the placeholder WAV (recording-too-short / timestamp) to match,
                            # and drop any leftover fail-safe WAV, so we end with one clean pair.
                            new_base = audio_processor.current_filename
                            if new_base:
                                new_wav = os.path.join(RECORDINGS_DIR, f"{new_base}.wav")
                                if os.path.abspath(new_wav) != os.path.abspath(audio_path):
                                    try:
                                        os.replace(audio_path, new_wav)
                                        audio_processor.saved_paths["content"] = new_wav
                                        audio_path = new_wav
                                    except OSError as rename_err:
                                        logger.error(f"Failed to rename recovered WAV: {rename_err}")
                            leftover = audio_processor.saved_paths.get("timestamp")
                            if leftover and os.path.exists(leftover) and os.path.abspath(leftover) != os.path.abspath(audio_path):
                                try:
                                    os.remove(leftover)
                                    audio_processor.saved_paths.pop("timestamp", None)
                                except OSError as rm_err:
                                    logger.error(f"Failed to remove leftover fail-safe WAV: {rm_err}")
                            logger.info(f"Auto-fallback recovered {len(recovered)} chars -> {transcription_path}")
                            if websocket.client_state == WebSocketState.CONNECTED:
                                await websocket.send_text(json.dumps({
                                    "type": "text", "content": recovered, "isNewResponse": True
                                }))
                        else:
                            logger.warning("Auto-fallback transcription returned empty")
                    except Exception as e:
                        logger.error(f"Auto-fallback transcription failed: {e}", exc_info=True)

                if success and audio_path and transcription_path:
                    audio_processor.cleanup_timestamp_backup()

                # Shadow testing: shadows (MiMo, Paraformer) were launched in
                # parallel with the OpenAI call the moment audio was buffered
                # (start_shadow_tasks), so they're usually already done by now.
                # Write their sibling transcripts using the descriptive name.
                # Non-blocking so this never delays finalize.
                # Remember where this session's primary transcript lives (for the
                # browser's Save button) and tell the client it's on disk.
                if transcription_path:
                    saved_files['primary'] = transcription_path
                    if websocket.client_state == WebSocketState.CONNECTED:
                        try:
                            await websocket.send_text(json.dumps({
                                "type": "saved", "model": "primary",
                                "file": os.path.basename(transcription_path)}))
                        except Exception as e:
                            logger.debug(f"Could not send saved event: {e}")

                reference = transcription_path or audio_path
                if pending_shadow_tasks:
                    # Already pushed to the browser as each finished; just write files.
                    finish_shadow_tasks(pending_shadow_tasks, reference, saved_files, websocket)
                    pending_shadow_tasks = []
                elif reference and audio_path and os.path.exists(audio_path):
                    # Modes without a local PCM buffer (e.g. realtime): late shadows from the WAV.
                    late_tasks = start_shadow_tasks_from_file(audio_path)
                    attach_shadow_pushers(late_tasks, websocket)
                    finish_shadow_tasks(late_tasks, reference, saved_files, websocket)

                audio_processor.end_session()
            else:
                logger.debug("No active session to finalize")

            pending_audio_chunks.clear()
            restful_audio_buffer.clear()
            pending_audio_operations = 0
            all_audio_sent.set()
            recording_stopped.set()

            if client:
                try:
                    await client.close()
                except Exception as e:
                    logger.error(f"Error closing OpenAI client during finalize: {e}", exc_info=True)
                finally:
                    client = None

            openai_ready.clear()

            if notify_client and websocket.client_state == WebSocketState.CONNECTED:
                for payload in (
                    {"type": "status", "status": "idle"},
                    {"type": "cleanup_audio"},
                ):
                    try:
                        await websocket.send_text(json.dumps(payload))
                    except Exception as e:
                        logger.error(f"Failed to send finalize payload {payload.get('type')}: {e}", exc_info=True)
                        break

            logger.info("Finalize recording completed")

    async def receive_messages():
        nonlocal client, current_mode, restful_audio_buffer, pending_audio_operations, pending_shadow_tasks

        try:
            while True:
                if websocket.client_state == WebSocketState.DISCONNECTED:
                    logger.info("WebSocket client disconnected")
                    openai_ready.clear()
                    break

                try:
                    # Add timeout to prevent infinite waiting
                    data = await asyncio.wait_for(websocket.receive(), timeout=30.0)

                    if "bytes" in data:
                        processed_audio = audio_processor.process_audio_chunk(data["bytes"])

                        if current_mode in ("restful", "audio15"):
                            # RESTful / audio15 mode: Buffer audio locally
                            restful_audio_buffer.append(processed_audio)
                            logger.debug(f"{current_mode} mode: Buffered audio chunk, size: {len(processed_audio)} bytes, total chunks: {len(restful_audio_buffer)}")
                        elif not openai_ready.is_set():
                            logger.debug("OpenAI not ready, buffering audio chunk")
                            pending_audio_chunks.append(processed_audio)
                        elif client and not recording_stopped.is_set():
                            # Safety check: Only send audio if recording is still active
                            # Track pending audio operations
                            async with audio_send_lock:
                                pending_audio_operations += 1
                                all_audio_sent.clear()  # Clear the event since we have pending operations

                            try:
                                await asyncio.wait_for(client.send_audio(processed_audio), timeout=2.0)
                                await websocket.send_text(json.dumps({
                                    "type": "status",
                                    "status": "connected"
                                }))
                                logger.debug(f"Sent audio chunk, size: {len(processed_audio)} bytes")
                            except asyncio.TimeoutError:
                                logger.error("Timeout sending audio chunk to OpenAI, finalizing recording for fail-safe save")
                                await finalize_recording(success=False, reason="send_audio_timeout")
                                break
                            except Exception as e:
                                logger.error(f"Error sending audio chunk to OpenAI: {e}", exc_info=True)
                                await finalize_recording(success=False, reason="send_audio_failure")
                                break
                            finally:
                                # Mark operation as complete
                                async with audio_send_lock:
                                    pending_audio_operations -= 1
                                    if pending_audio_operations == 0:
                                        all_audio_sent.set()  # Set event when all operations complete
                        else:
                            logger.warning("Received audio but client is not initialized")

                    elif "text" in data:
                        msg = json.loads(data["text"])

                        if msg.get("type") == "start_recording":
                            saved_files.clear()  # new recording: forget last session's files
                            if audio_processor.has_active_session():
                                logger.warning("Start recording requested while a session is active. Finalizing previous session first.")
                                await finalize_recording(success=False, reason="duplicate_start", notify_client=False)

                            # Get sample rate from browser (default to 48000 if not provided)
                            source_sample_rate = msg.get("sample_rate", 48000)
                            audio_processor.source_sample_rate = source_sample_rate
                            logger.info(f"Browser sample rate: {source_sample_rate}Hz")

                            # Get the transcription mode from the message
                            current_mode = msg.get("mode", "realtime")
                            logger.info(f"Starting recording in {current_mode} mode")

                            audio_processor.start_new_session()
                            recording_stopped.clear()
                            pending_audio_chunks.clear()
                            restful_audio_buffer.clear()

                            # Update status to connecting while initializing
                            await websocket.send_text(json.dumps({
                                "type": "status",
                                "status": "connecting"
                            }))

                            if current_mode == "realtime":
                                # Realtime mode: Initialize OpenAI Realtime API
                                if not await initialize_openai():
                                    logger.warning("OpenAI initialization failed; continuing in local fail-safe mode")
                                    continue

                                # Send any buffered chunks
                                if pending_audio_chunks and client:
                                    logger.info(f"Sending {len(pending_audio_chunks)} buffered chunks")
                                    for chunk in pending_audio_chunks:
                                        # Safety check: Stop sending if recording has been stopped
                                        if recording_stopped.is_set():
                                            logger.info("Recording stopped while sending buffered chunks, stopping transmission")
                                            break
                                        # Track each buffered chunk operation
                                        async with audio_send_lock:
                                            pending_audio_operations += 1
                                            all_audio_sent.clear()

                                        try:
                                            await asyncio.wait_for(client.send_audio(chunk), timeout=2.0)
                                        except asyncio.TimeoutError:
                                            logger.error("Timeout sending buffered chunk to OpenAI, finalizing recording for fail-safe save")
                                            await finalize_recording(success=False, reason="send_buffered_timeout")
                                            break
                                        except Exception as e:
                                            logger.error(f"Error sending buffered audio chunk to OpenAI: {e}", exc_info=True)
                                            await finalize_recording(success=False, reason="send_buffered_failure")
                                            break
                                        finally:
                                            async with audio_send_lock:
                                                pending_audio_operations -= 1
                                                if pending_audio_operations == 0:
                                                    all_audio_sent.set()
                                    pending_audio_chunks.clear()
                            else:
                                # RESTful mode: Just mark as connected, we'll buffer audio locally
                                await websocket.send_text(json.dumps({
                                    "type": "status",
                                    "status": "connected"
                                }))
                                logger.info("RESTful mode: Ready to buffer audio locally")

                        elif msg.get("type") == "save_transcript":
                            # Browser edited a transcript (primary or a shadow tab):
                            # write it back, preserving the time-spent footer.
                            model = msg.get("model") or "primary"
                            content = msg.get("content") or ""
                            path = saved_files.get(model)
                            # Fallback: client may name the file (basename only; must be a .txt in RECORDINGS_DIR)
                            if not path and msg.get("file"):
                                cand = os.path.join(RECORDINGS_DIR, os.path.basename(msg["file"]))
                                if cand.endswith(".txt") and os.path.isfile(cand):
                                    path = cand
                            result = {"type": "save_result", "model": model, "ok": False}
                            if not path or not os.path.isfile(path):
                                result["error"] = "No saved file for this transcript yet (or the page was reloaded)."
                            else:
                                try:
                                    footer = rewrite_transcript_body(path, content)
                                    result.update(ok=True, file=os.path.basename(path), footer=footer)
                                    logger.info(f"Saved browser edit for {model} -> {path} ({len(content)} chars)")
                                except Exception as e:
                                    logger.error(f"Failed to save edit for {model}: {e}", exc_info=True)
                                    result["error"] = str(e)
                            if websocket.client_state == WebSocketState.CONNECTED:
                                await websocket.send_text(json.dumps(result))

                        elif msg.get("type") == "stop_recording":
                            # Always ensure a local fail-safe file exists as soon as recording stops
                            try:
                                audio_processor.save_audio_buffer(strategy="timestamp")
                            except Exception as e:
                                logger.error(f"Failed to save fail-safe recording on stop: {e}", exc_info=True)

                            if current_mode == "restful":
                                # RESTful mode: Send buffered audio to REST API for transcription
                                logger.info(f"RESTful mode: Processing {len(restful_audio_buffer)} buffered audio chunks")

                                if restful_audio_buffer:
                                    # Combine all buffered audio into one
                                    combined_audio = b''.join(restful_audio_buffer)
                                    logger.info(f"RESTful mode: Total audio size: {len(combined_audio)} bytes")

                                    # Launch shadows (MiMo, Paraformer) now, in parallel with the (slower) OpenAI call.
                                    pending_shadow_tasks = start_shadow_tasks(combined_audio)
                                    # Tell the browser which shadow tabs to show (spinning)
                                    # right away; each fills in as its engine finishes.
                                    if pending_shadow_tasks and websocket.client_state == WebSocketState.CONNECTED:
                                        await websocket.send_text(json.dumps({
                                            "type": "shadow_start",
                                            "models": [lbl for (_t, _s, lbl) in pending_shadow_tasks],
                                        }))
                                        # Each tab fills the instant its engine finishes —
                                        # NOT after the (slower) gpt-4o primary completes.
                                        attach_shadow_pushers(pending_shadow_tasks, websocket)

                                    # Send new response indicator
                                    await websocket.send_text(json.dumps({
                                        "type": "text",
                                        "content": "",
                                        "isNewResponse": True
                                    }))

                                    try:
                                        # Non-streaming transcription is already clean UTF-8;
                                        # no grammar-fix pass (it over-edited and dropped content).
                                        # Pass the in-flight shadow tasks so the completeness
                                        # cross-check can flag silent trailing-segment drops.
                                        await transcribe_with_rest_api(combined_audio, websocket, audio_processor)
                                        await finalize_recording(success=True, reason="restful_complete")
                                    except Exception as e:
                                        logger.error(f"RESTful transcription failed: {e}", exc_info=True)
                                        # Parse OpenAI-specific errors for better user feedback
                                        error_msg = str(e)
                                        if hasattr(e, 'status_code'):
                                            if e.status_code == 429:
                                                if 'insufficient_quota' in error_msg.lower() or 'quota' in error_msg.lower():
                                                    error_msg = "OpenAI Quota Exceeded: Please check your billing at https://platform.openai.com/account/billing"
                                                else:
                                                    error_msg = "OpenAI Rate Limit: Too many requests. Please wait and try again."
                                            elif e.status_code == 401:
                                                error_msg = "OpenAI Authentication Failed: Please check your API key."
                                            elif e.status_code == 400:
                                                error_msg = f"OpenAI Bad Request: {error_msg}"
                                        await websocket.send_text(json.dumps({
                                            "type": "error",
                                            "content": error_msg
                                        }))
                                        await finalize_recording(success=False, reason="restful_error")
                                else:
                                    logger.warning("RESTful mode: No audio data buffered")
                                    await finalize_recording(success=False, reason="no_audio_data")

                            elif current_mode == "audio15":
                                # gpt-audio-1.5 via Chat Completions: process buffered audio
                                logger.info(f"audio15 mode: Processing {len(restful_audio_buffer)} buffered audio chunks")

                                if restful_audio_buffer:
                                    combined_audio = b''.join(restful_audio_buffer)
                                    logger.info(f"audio15 mode: Total audio size: {len(combined_audio)} bytes")

                                    # Launch shadows (MiMo, Paraformer) now, in parallel with the (slower) OpenAI call.
                                    pending_shadow_tasks = start_shadow_tasks(combined_audio)
                                    # Tell the browser which shadow tabs to show (spinning)
                                    # right away; each fills in as its engine finishes.
                                    if pending_shadow_tasks and websocket.client_state == WebSocketState.CONNECTED:
                                        await websocket.send_text(json.dumps({
                                            "type": "shadow_start",
                                            "models": [lbl for (_t, _s, lbl) in pending_shadow_tasks],
                                        }))
                                        # Each tab fills the instant its engine finishes —
                                        # NOT after the (slower) gpt-4o primary completes.
                                        attach_shadow_pushers(pending_shadow_tasks, websocket)

                                    await websocket.send_text(json.dumps({
                                        "type": "text",
                                        "content": "",
                                        "isNewResponse": True
                                    }))

                                    try:
                                        await transcribe_with_audio15(combined_audio, websocket, audio_processor)
                                        await finalize_recording(success=True, reason="audio15_complete")
                                    except Exception as e:
                                        logger.error(f"gpt-audio-1.5 transcription failed: {e}", exc_info=True)
                                        error_msg = str(e)
                                        if hasattr(e, 'status_code'):
                                            if e.status_code == 429:
                                                error_msg = "OpenAI Rate Limit: Too many requests. Please wait and try again."
                                            elif e.status_code == 401:
                                                error_msg = "OpenAI Authentication Failed: Please check your API key."
                                            elif e.status_code == 400:
                                                error_msg = f"OpenAI Bad Request: {error_msg}"
                                        await websocket.send_text(json.dumps({
                                            "type": "error",
                                            "content": error_msg
                                        }))
                                        await finalize_recording(success=False, reason="audio15_error")
                                else:
                                    logger.warning("audio15 mode: No audio data buffered")
                                    await finalize_recording(success=False, reason="no_audio_data")

                            elif client:
                                # Realtime mode: Use existing OpenAI Realtime API flow
                                # CRITICAL FIX: Wait for all pending audio operations to complete
                                # before committing to prevent data loss
                                logger.info("Stop recording received, waiting for all audio to be sent...")

                                # Wait for any pending audio chunks to be sent (with timeout for safety)
                                try:
                                    await asyncio.wait_for(all_audio_sent.wait(), timeout=5.0)
                                    logger.info("All pending audio operations completed")
                                except asyncio.TimeoutError:
                                    logger.warning("Timeout waiting for audio operations to complete, proceeding anyway")
                                    # Reset the pending counter to prevent deadlock
                                    async with audio_send_lock:
                                        pending_audio_operations = 0
                                        all_audio_sent.set()

                                # Add a small buffer to ensure network operations complete
                                await asyncio.sleep(0.1)

                                logger.info("All audio sent, committing audio buffer...")
                                try:
                                    await asyncio.wait_for(client.commit_audio(), timeout=5.0)
                                    await websocket.send_text(json.dumps({
                                        "type": "text", "content": "", "isNewResponse": True
                                    }))
                                    try:
                                        await asyncio.wait_for(recording_stopped.wait(), timeout=30.0)
                                    except asyncio.TimeoutError:
                                        logger.warning("Timeout waiting for transcription completion; finalizing with fail-safe save")
                                        await finalize_recording(success=False, reason="transcription_timeout")
                                        continue
                                    # Don't close the client here, let the disconnect timer handle it
                                    # Update client status to connected (waiting for response)
                                    await websocket.send_text(json.dumps({
                                        "type": "status",
                                        "status": "connected"
                                    }))
                                except Exception as e:
                                    logger.error(f"Error while finalizing stop recording: {e}", exc_info=True)
                                    await finalize_recording(success=False, reason="stop_recording_failure")
                            else:
                                logger.info("Stop recording received but OpenAI client is not ready; saving fail-safe recording.")
                                await finalize_recording(success=False, reason="stop_without_client")

                except asyncio.TimeoutError:
                    logger.debug("No message received for 30 seconds")
                    continue
                except Exception as e:
                    logger.error(f"Error in receive_messages loop: {str(e)}", exc_info=True)
                    break

        finally:
            # Ensure any in-progress recording is finalized when the loop exits
            await finalize_recording(success=False, reason="receive_loop_exit")
            logger.info("Receive messages loop ended")

    try:
        await receive_messages()
    finally:
        if client:
            await client.close()
            logger.info("OpenAI client connection closed")

@app.post(
    "/api/v1/readability",
    response_model=ReadabilityResponse,
    summary="Enhance Text Readability",
    description="Improve the readability of the provided text using GPT-4."
)
async def enhance_readability(request: ReadabilityRequest):
    prompt = PROMPTS.get('readability-enhance')
    if not prompt:
        raise HTTPException(status_code=500, detail="Readability prompt not found.")

    try:
        async def text_generator():
            # Use gpt-4o specifically for readability
            async for part in llm_processor.process_text(request.text, prompt, model="gpt-4o"):
                yield part

        return StreamingResponse(text_generator(), media_type="text/plain")

    except Exception as e:
        logger.error(f"Error enhancing readability: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing readability enhancement.")

@app.post(
    "/api/v1/ask_ai",
    response_model=AskAIResponse,
    summary="Ask AI a Question",
    description="Ask AI to provide insights using O1-mini model."
)
def ask_ai(request: AskAIRequest):
    prompt = PROMPTS.get('ask-ai')
    if not prompt:
        raise HTTPException(status_code=500, detail="Ask AI prompt not found.")

    try:
        # Use o1-mini specifically for ask_ai
        answer = llm_processor.process_text_sync(request.text, prompt, model="o1-mini")
        return AskAIResponse(answer=answer)
    except Exception as e:
        logger.error(f"Error processing AI question: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing AI question.")

@app.post(
    "/api/v1/correctness",
    response_model=CorrectnessResponse,
    summary="Check Factual Correctness",
    description="Analyze the text for factual accuracy using GPT-4o."
)
async def check_correctness(request: CorrectnessRequest):
    prompt = PROMPTS.get('correctness-check')
    if not prompt:
        raise HTTPException(status_code=500, detail="Correctness prompt not found.")

    try:
        async def text_generator():
            # Specifically use gpt-4o for correctness checking
            async for part in llm_processor.process_text(request.text, prompt, model="gpt-4o"):
                yield part

        return StreamingResponse(text_generator(), media_type="text/plain")

    except Exception as e:
        logger.error(f"Error checking correctness: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing correctness check.")

@app.post(
    "/api/v1/upload_wav",
    summary="Upload WAV file for transcription",
    description="Upload a WAV file to be processed using OpenAI Realtime API with the same prompt as live recording."
)
async def upload_wav(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.wav'):
        raise HTTPException(status_code=400, detail="Only WAV files are supported.")
    
    try:
        logger.info(f"Processing uploaded WAV file: {file.filename}")
        
        # Read the uploaded file
        file_content = await file.read()
        
        # Create a temporary file to store the uploaded WAV
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name
        
        try:
            # Load and process the WAV file to get the right format
            with wave.open(tmp_file_path, 'rb') as wav_file:
                # Get audio parameters
                frames = wav_file.readframes(wav_file.getnframes())
                sample_rate = wav_file.getframerate()
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                
                logger.info(f"WAV file info: {sample_rate}Hz, {channels} channels, {sample_width} bytes/sample")
                
                # Convert to the format expected by the realtime API (PCM16, mono, 24kHz)
                audio_data = np.frombuffer(frames, dtype=np.int16)
                
                # Convert to mono if stereo
                if channels == 2:
                    audio_data = audio_data.reshape(-1, 2).mean(axis=1).astype(np.int16)
                    
                # Resample to 24kHz if needed
                if sample_rate != 24000:
                    # Convert to float for resampling
                    float_data = audio_data.astype(np.float32) / 32768.0
                    resampled_data = scipy.signal.resample_poly(
                        float_data, 24000, sample_rate
                    )
                    audio_data = (resampled_data * 32768.0).clip(-32768, 32767).astype(np.int16)
                
                # Convert back to bytes
                processed_audio = audio_data.tobytes()
                
            # Initialize OpenAI Realtime client (transcription session)
            client = OpenAIRealtimeAudioTextClient(OPENAI_API_KEY)
            await client.connect()

            whisper_chunks: list[str] = []
            transcript_complete = asyncio.Event()

            async def wav_handle_delta(data):
                delta = data.get("delta", "")
                if delta:
                    whisper_chunks.append(delta)

            async def wav_handle_completed(data):
                full = data.get("transcript", "")
                if full:
                    whisper_chunks.clear()
                    whisper_chunks.append(full)
                transcript_complete.set()

            client.register_handler("conversation.item.input_audio_transcription.delta", wav_handle_delta)
            client.register_handler("conversation.item.input_audio_transcription.completed", wav_handle_completed)

            # Send audio in chunks
            chunk_size = 4096
            for i in range(0, len(processed_audio), chunk_size):
                await client.send_audio(processed_audio[i:i + chunk_size])
                await asyncio.sleep(0.01)

            await client.commit_audio()
            await asyncio.wait_for(transcript_complete.wait(), timeout=30.0)
            await client.close()

            raw_transcript = whisper_chunks[0] if whisper_chunks else ""
            logger.info(f"WAV Upload - whisper transcript length: {len(raw_transcript)}")

            full_response = await asyncio.to_thread(
                llm_processor.process_text_sync,
                raw_transcript,
                PROMPTS['grammar-fix'],
                "gpt-4o-mini"
            )
            logger.info(f"Successfully processed WAV file with Realtime API: {file.filename}")
            logger.info(f"Response length: {len(full_response)} characters")

            # Persist the transcript to a .txt so the Realtime upload matches the
            # Transcribe upload (which saves a file). Uses the timestamp embedded
            # in the uploaded filename; save_transcription generates a fresh
            # descriptive name from the content.
            transcript_text = full_response.strip()
            if transcript_text:
                session_id = extract_time_tag_from_filename(file.filename)
                naming_processor = AudioProcessor()
                naming_processor.current_session_id = session_id
                naming_processor.current_transcription = [transcript_text]
                naming_processor.current_filename = None
                naming_processor._header_removed = True
                naming_processor.transcription_model = "gpt-realtime (+ gpt-4o-mini grammar-fix, manual upload)"
                try:
                    saved_txt = naming_processor.save_transcription()
                    logger.info(f"Saved Realtime upload transcription to {saved_txt}")
                except Exception as save_error:
                    logger.error(f"Failed to save Realtime upload transcription for {session_id}: {save_error}", exc_info=True)

            async def text_generator():
                yield full_response

            return StreamingResponse(text_generator(), media_type="text/plain")
            
        finally:
            # Clean up the temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
    
    except Exception as e:
        logger.error(f"Error processing WAV file: {e}", exc_info=True)
        # Parse OpenAI-specific errors for better user feedback
        error_msg = str(e)
        status_code = 500
        if hasattr(e, 'status_code'):
            status_code = e.status_code
            if e.status_code == 429:
                if 'insufficient_quota' in error_msg.lower() or 'quota' in error_msg.lower():
                    error_msg = "OpenAI Quota Exceeded: Please check your billing at https://platform.openai.com/account/billing"
                else:
                    error_msg = "OpenAI Rate Limit: Too many requests. Please wait and try again."
            elif e.status_code == 401:
                error_msg = "OpenAI Authentication Failed: Please check your API key."
        raise HTTPException(status_code=status_code, detail=error_msg)

@app.post(
    "/api/v1/upload_wav_transcribe",
    summary="Upload WAV file for gpt-4o-transcribe transcription",
    description="Upload a WAV file to be transcribed via the REST transcription API using gpt-4o-transcribe."
)
async def upload_wav_transcribe(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.wav'):
        raise HTTPException(status_code=400, detail="Only WAV files are supported.")

    try:
        logger.info(f"Processing uploaded WAV file with gpt-4o-transcribe: {file.filename}")

        # Read the uploaded file
        file_content = await file.read()

        # Initialize OpenAI client
        client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        
        # Create a temporary file for the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name
        
        try:
            # Transcribe via gpt-4o-transcribe (matches the 🟢 dropdown option).
            t_start = time.monotonic()
            with open(tmp_file_path, 'rb') as audio_file:
                transcript = await client.audio.transcriptions.create(
                    model="gpt-4o-transcribe",
                    file=audio_file,
                    response_format="text",
                    prompt=TRANSCRIBE_HINT,
                )
            upload_elapsed = time.monotonic() - t_start

            logger.info(f"Successfully transcribed WAV file with gpt-4o-transcribe: {file.filename}")
            logger.info(f"Transcription length: {len(transcript)} characters")

            # Persist transcription using the timestamp embedded in the filename (if present)
            transcript_text = transcript.strip()
            session_id = extract_time_tag_from_filename(file.filename)
            naming_processor = AudioProcessor()
            naming_processor.current_session_id = session_id
            naming_processor.current_transcription = [transcript_text]
            naming_processor.current_filename = None
            naming_processor._header_removed = True
            naming_processor.transcription_model = "gpt-4o-transcribe (manual upload)"
            naming_processor.transcription_elapsed = upload_elapsed

            txt_path = None
            try:
                txt_path = naming_processor.save_transcription()
            except Exception as save_error:
                logger.error(f"Failed to save gpt-4o-transcribe transcription for {session_id}: {save_error}", exc_info=True)

            if txt_path:
                logger.info(f"Saved gpt-4o-transcribe transcription to {txt_path}")
            else:
                logger.warning("gpt-4o-transcribe transcription could not be saved; returning text response only.")
            
            # Return the transcription as a streaming response
            async def text_generator():
                yield transcript
            
            return StreamingResponse(text_generator(), media_type="text/plain")
            
        finally:
            # Clean up the temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
    
    except Exception as e:
        logger.error(f"Error processing WAV file with gpt-4o-transcribe: {e}", exc_info=True)
        # Parse OpenAI-specific errors for better user feedback
        error_msg = str(e)
        status_code = 500
        if hasattr(e, 'status_code'):
            status_code = e.status_code
            if e.status_code == 429:
                if 'insufficient_quota' in error_msg.lower() or 'quota' in error_msg.lower():
                    error_msg = "OpenAI Quota Exceeded: Please check your billing at https://platform.openai.com/account/billing"
                else:
                    error_msg = "OpenAI Rate Limit: Too many requests. Please wait and try again."
            elif e.status_code == 401:
                error_msg = "OpenAI Authentication Failed: Please check your API key."
        raise HTTPException(status_code=status_code, detail=error_msg)

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=3005)
