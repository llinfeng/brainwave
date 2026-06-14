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
from prompts import PROMPTS
from openai_realtime_client import OpenAIRealtimeAudioTextClient
from starlette.websockets import WebSocketState
import wave
import datetime
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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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

# Initialize with a default model
llm_processor = get_llm_processor("gpt-4o")  # Default processor

app.mount("/static", StaticFiles(directory="static"), name="static")

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

        with open(txt_path, 'wb') as f:  # Open in binary mode
            # Write UTF-8 BOM
            f.write(b'\xef\xbb\xbf')
            # Write content encoded as UTF-8
            f.write(full_text.encode('utf-8'))
        logger.info(f"Saved transcription to {txt_path} with UTF-8-BOM encoding")
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

async def transcribe_with_rest_api(audio_data: bytes, websocket: WebSocket, audio_processor: AudioProcessor):
    """Transcribe audio using REST API with SSE streaming using gpt-4o-mini-transcribe"""
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    # Create temp file for audio (WAV format at 24kHz mono)
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        tmp_path = f.name
        with wave.open(f, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(24000)
            wf.writeframes(audio_data)

    try:
        logger.info(f"Calling REST API transcription with file: {tmp_path}")

        # Call transcription API with streaming
        with open(tmp_path, 'rb') as audio_file:
            stream = await client.audio.transcriptions.create(
                model="gpt-4o-mini-transcribe",
                file=audio_file,
                response_format="text",
                stream=True,
                language="zh",
                prompt="以下是普通话录音，可能混有英语技术术语，包含项目规划、商业讨论等内容。",
            )

            full_text = ""
            async for event in stream:
                if hasattr(event, 'type'):
                    if event.type == "transcript.text.delta":
                        delta = event.delta if hasattr(event, 'delta') else ""
                        if delta:
                            full_text += delta
                            if websocket.client_state == WebSocketState.CONNECTED:
                                await websocket.send_text(json.dumps({
                                    "type": "text",
                                    "content": delta,
                                    "isNewResponse": False
                                }))
                    elif event.type == "transcript.text.done":
                        logger.info(f"REST API transcription complete, length: {len(full_text)}")
                elif hasattr(event, 'text'):
                    # Fallback for non-streaming response format
                    full_text = event.text
                    if websocket.client_state == WebSocketState.CONNECTED:
                        await websocket.send_text(json.dumps({
                            "type": "text",
                            "content": full_text,
                            "isNewResponse": False
                        }))

            if full_text:
                audio_processor.current_transcription = [full_text]
            return full_text

    except Exception as e:
        logger.error(f"Error in REST API transcription: {e}", exc_info=True)
        raise
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


async def transcribe_with_audio15(audio_data: bytes, websocket: WebSocket, audio_processor: AudioProcessor):
    """Transcribe audio using Chat Completions with gpt-audio-1.5"""
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)

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
        # Model sometimes returns JSON despite the prompt instructing plain text
        try:
            parsed = json.loads(content)
            transcript_text = (parsed.get("transcription") or "").strip() if isinstance(parsed, dict) else content
        except (json.JSONDecodeError, ValueError):
            transcript_text = content
        logger.info(f"gpt-audio-1.5 transcription complete, length: {len(transcript_text)}")
        if transcript_text:
            audio_processor.current_transcription = [transcript_text]
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
        nonlocal client, pending_audio_chunks, pending_audio_operations, restful_audio_buffer

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

                if success and audio_path and transcription_path:
                    audio_processor.cleanup_timestamp_backup()

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
        nonlocal client, current_mode, restful_audio_buffer, pending_audio_operations

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

                                    # Send new response indicator
                                    await websocket.send_text(json.dumps({
                                        "type": "text",
                                        "content": "",
                                        "isNewResponse": True
                                    }))

                                    try:
                                        raw_text = await transcribe_with_rest_api(combined_audio, websocket, audio_processor)
                                        if raw_text:
                                            try:
                                                fixed = await asyncio.to_thread(
                                                    llm_processor.process_text_sync,
                                                    raw_text,
                                                    PROMPTS['grammar-fix'],
                                                    "gpt-4o-mini"
                                                )
                                                audio_processor.current_transcription = [fixed]
                                                if websocket.client_state == WebSocketState.CONNECTED:
                                                    await websocket.send_text(json.dumps({
                                                        "type": "text",
                                                        "content": fixed,
                                                        "isNewResponse": True
                                                    }))
                                            except Exception as gf_err:
                                                logger.error(f"Grammar fix failed for restful mode: {gf_err}")
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
    "/api/v1/upload_wav_whisper",
    summary="Upload WAV file for Whisper transcription",
    description="Upload a WAV file to be transcribed using OpenAI Whisper API for literal transcription."
)
async def upload_wav_whisper(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.wav'):
        raise HTTPException(status_code=400, detail="Only WAV files are supported.")
    
    try:
        logger.info(f"Processing uploaded WAV file with Whisper: {file.filename}")
        
        # Read the uploaded file
        file_content = await file.read()
        
        # Initialize OpenAI client for Whisper
        client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        
        # Create a temporary file for the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name
        
        try:
            # Use OpenAI Whisper to transcribe the audio
            with open(tmp_file_path, 'rb') as audio_file:
                transcript = await client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )
            
            logger.info(f"Successfully transcribed WAV file with Whisper: {file.filename}")
            logger.info(f"Transcription length: {len(transcript)} characters")

            # Persist transcription using the timestamp embedded in the filename (if present)
            transcript_text = transcript.strip()
            session_id = extract_time_tag_from_filename(file.filename)
            naming_processor = AudioProcessor()
            naming_processor.current_session_id = session_id
            naming_processor.current_transcription = [transcript_text]
            naming_processor.current_filename = None
            naming_processor._header_removed = True

            txt_path = None
            try:
                txt_path = naming_processor.save_transcription()
            except Exception as save_error:
                logger.error(f"Failed to save Whisper transcription for {session_id}: {save_error}", exc_info=True)

            if txt_path:
                logger.info(f"Saved Whisper transcription to {txt_path}")
            else:
                logger.warning("Whisper transcription could not be saved; returning text response only.")
            
            # Return the transcription as a streaming response
            async def text_generator():
                yield transcript
            
            return StreamingResponse(text_generator(), media_type="text/plain")
            
        finally:
            # Clean up the temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
    
    except Exception as e:
        logger.error(f"Error processing WAV file with Whisper: {e}", exc_info=True)
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
