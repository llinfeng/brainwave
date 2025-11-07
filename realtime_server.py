import asyncio
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
        """Remove the fixed header line from a full transcript string."""
        header_line = "下面是语音识别转录结果："
        if text.startswith(header_line):
            remainder = text[len(header_line):]
            return remainder.lstrip("\r\n")
        return text

    def _strip_transcript_header(self, text: str) -> str:
        """Remove the fixed header line from the streaming transcript output."""
        if self._header_removed:
            return text

        combined = f"{self._header_buffer}{text}"
        header = self._expected_header
        header_line = self._transcript_header_line

        if header.startswith(combined):
            # Still collecting header characters
            self._header_buffer = combined
            if len(self._header_buffer) == len(header):
                self._header_buffer = ""
                self._header_removed = True
            return ""

        if combined.startswith(header):
            # Header fully matched; remove it and mark as removed
            self._header_removed = True
            self._header_buffer = ""
            return combined[len(header):]

        if combined.startswith(header_line):
            # Header line present but whitespace differs; strip line and leading newlines
            remainder = combined[len(header_line):].lstrip("\r\n")
            self._header_removed = True
            self._header_buffer = ""
            return remainder

        # Header missing or altered: release buffered text and stop filtering
        self._header_removed = True
        buffered = self._header_buffer
        self._header_buffer = ""
        return f"{buffered}{text}"

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

    # Add synchronization for audio sending operations
    pending_audio_operations = 0
    audio_send_lock = asyncio.Lock()
    all_audio_sent = asyncio.Event()
    all_audio_sent.set()  # Initially set since no audio is pending
    finalize_lock = asyncio.Lock()


    async def initialize_openai():
        nonlocal client
        try:
            # Clear the ready flag while initializing
            openai_ready.clear()

            client = OpenAIRealtimeAudioTextClient(os.getenv("OPENAI_API_KEY"))
            await client.connect()
            logger.info("Successfully connected to OpenAI client")

            # Register handlers after client is initialized
            client.register_handler("session.updated", lambda data: handle_generic_event("session.updated", data))
            client.register_handler("input_audio_buffer.cleared", lambda data: handle_generic_event("input_audio_buffer.cleared", data))
            client.register_handler("input_audio_buffer.speech_started", lambda data: handle_generic_event("input_audio_buffer.speech_started", data))
            client.register_handler("rate_limits.updated", lambda data: handle_generic_event("rate_limits.updated", data))
            client.register_handler("response.output_item.added", lambda data: handle_generic_event("response.output_item.added", data))
            client.register_handler("conversation.item.created", lambda data: handle_generic_event("conversation.item.created", data))
            client.register_handler("response.content_part.added", lambda data: handle_generic_event("response.content_part.added", data))
            client.register_handler("response.text.done", lambda data: handle_generic_event("response.text.done", data))
            client.register_handler("response.content_part.done", lambda data: handle_generic_event("response.content_part.done", data))
            client.register_handler("response.output_item.done", lambda data: handle_generic_event("response.output_item.done", data))
            client.register_handler("response.done", lambda data: handle_response_done(data))
            client.register_handler("error", lambda data: handle_error(data))
            client.register_handler("response.text.delta", lambda data: handle_text_delta(data))
            client.register_handler("response.created", lambda data: handle_response_created(data))

            openai_ready.set()  # Set ready flag after successful initialization
            await websocket.send_text(json.dumps({
                "type": "status",
                "status": "connected"
            }))
            return True
        except Exception as e:
            logger.error(f"Failed to connect to OpenAI: {e}", exc_info=True)
            openai_ready.clear()  # Ensure flag is cleared on failure
            if client:
                try:
                    await client.close()
                except Exception as close_error:
                    logger.error(f"Error closing partially initialized OpenAI client: {close_error}", exc_info=True)
                finally:
                    client = None
            await websocket.send_text(json.dumps({
                "type": "error",
                "content": "Failed to initialize OpenAI connection"
            }))
            return False

    # Move the handler definitions here (before initialize_openai)
    async def handle_text_delta(data):
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                text_delta = data.get("delta", "")
                logger.info(f"Received text delta: '{text_delta}'")
                cleaned_text = audio_processor.add_transcription_text(text_delta)
                logger.info(f"Current transcription length: {len(''.join(audio_processor.current_transcription))}")
                if cleaned_text:
                    await websocket.send_text(json.dumps({
                        "type": "text",
                        "content": cleaned_text,
                        "isNewResponse": False
                    }))
                    logger.info("Handled response.text.delta")
                else:
                    logger.debug("Filtered transcript header chunk from response stream")
        except Exception as e:
            logger.error(f"Error in handle_text_delta: {str(e)}", exc_info=True)

    async def handle_response_created(data):
        await websocket.send_text(json.dumps({
            "type": "text",
            "content": "",
            "isNewResponse": True
        }))
        logger.info("Handled response.created")

    async def handle_error(data):
        error_msg = data.get("error", {}).get("message", "Unknown error")
        logger.error(f"OpenAI error: {error_msg}")
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "content": error_msg
            }))
        except Exception as e:
            logger.error(f"Failed to notify client about error: {e}", exc_info=True)
        logger.info("Handled error message from OpenAI")
        await finalize_recording(success=False, reason="openai_error")

    async def handle_response_done(data):
        logger.info("Handled response.done")
        await finalize_recording(success=True, reason="response_done")

    async def handle_generic_event(event_type, data):
        logger.info(f"Handled {event_type} with data: {json.dumps(data, ensure_ascii=False)}")

    async def finalize_recording(success=False, reason=""):
        nonlocal client, pending_audio_chunks, pending_audio_operations

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

            if websocket.client_state == WebSocketState.CONNECTED:
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

    # Create a queue to handle incoming audio chunks
    audio_queue = asyncio.Queue()

    async def receive_messages():
        nonlocal client

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
                        if not openai_ready.is_set():
                            logger.debug("OpenAI not ready, buffering audio chunk")
                            pending_audio_chunks.append(processed_audio)
                        elif client and not recording_stopped.is_set():
                            # Safety check: Only send audio if recording is still active
                            # Track pending audio operations
                            async with audio_send_lock:
                                nonlocal pending_audio_operations
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
                                await finalize_recording(success=False, reason="duplicate_start")

                            audio_processor.start_new_session()
                            recording_stopped.clear()
                            pending_audio_chunks.clear()

                            # Update status to connecting while initializing OpenAI
                            await websocket.send_text(json.dumps({
                                "type": "status",
                                "status": "connecting"
                            }))

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

                        elif msg.get("type") == "stop_recording":
                            # Always ensure a local fail-safe file exists as soon as recording stops
                            try:
                                audio_processor.save_audio_buffer(strategy="timestamp")
                            except Exception as e:
                                logger.error(f"Failed to save fail-safe recording on stop: {e}", exc_info=True)

                            if client:
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
                                    # Timeout protection around network calls so we can fall back if OpenAI is unreachable
                                    await asyncio.wait_for(client.commit_audio(), timeout=5.0)

                                    # Debug: log the prompt being sent
                                    prompt_text = PROMPTS['paraphrase-gpt-realtime']
                                    logger.info(f"Sending prompt to realtime API: {prompt_text[:200]}...")
                                    await asyncio.wait_for(client.start_response(prompt_text), timeout=5.0)
                                    try:
                                        await asyncio.wait_for(recording_stopped.wait(), timeout=15.0)
                                    except asyncio.TimeoutError:
                                        logger.warning("Timeout waiting for OpenAI response completion; finalizing with fail-safe save")
                                        await finalize_recording(success=False, reason="response_timeout")
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

    async def send_audio_messages():
        while True:
            try:
                processed_audio = await audio_queue.get()
                if processed_audio is None:
                    break

                # Add validation
                if len(processed_audio) == 0:
                    logger.warning("Empty audio chunk received, skipping")
                    continue

                # Append the processed audio to the buffer
                audio_processor.audio_buffer.append(processed_audio)

                await client.send_audio(processed_audio)
                logger.info(f"Audio chunk sent to OpenAI client, size: {len(processed_audio)} bytes")

            except Exception as e:
                logger.error(f"Error in send_audio_messages: {str(e)}", exc_info=True)
                break

        # After processing all audio, set the event
        recording_stopped.set()

    # Start concurrent tasks for receiving and sending
    receive_task = asyncio.create_task(receive_messages())
    send_task = asyncio.create_task(send_audio_messages())

    try:
        # Wait for both tasks to complete
        await asyncio.gather(receive_task, send_task)
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
                
            # Initialize OpenAI Realtime client
            client = OpenAIRealtimeAudioTextClient(OPENAI_API_KEY)
            await client.connect()
            
            # Collect response text
            response_text = []
            response_complete = asyncio.Event()
            
            async def handle_text_delta(data):
                text_delta = data.get("delta", "")
                if text_delta:
                    response_text.append(text_delta)
            
            async def handle_response_done(data):
                response_complete.set()
            
            # Register handlers
            client.register_handler("response.text.delta", handle_text_delta)
            client.register_handler("response.done", handle_response_done)
            
            # Send the audio data in chunks (like realtime recording)
            chunk_size = 4096  # Same as realtime recording
            for i in range(0, len(processed_audio), chunk_size):
                chunk = processed_audio[i:i + chunk_size]
                await client.send_audio(chunk)
                # Small delay to simulate realtime
                await asyncio.sleep(0.01)
            
            # Commit the audio and start response with the same prompt
            await client.commit_audio()
            
            # Debug: log the prompt being sent
            prompt_text = PROMPTS['paraphrase-gpt-realtime']
            logger.info(f"WAV Upload - Sending prompt to realtime API: {prompt_text[:200]}...")
            await client.start_response(prompt_text)
            
            # Wait for response completion
            await response_complete.wait()
            
            # Clean up
            await client.close()
            
            # Return the collected response
            full_response = ''.join(response_text)
            full_response = AudioProcessor.strip_transcript_header(full_response)
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
        raise HTTPException(status_code=500, detail=f"Error processing WAV file: {str(e)}")

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
        raise HTTPException(status_code=500, detail=f"Error processing WAV file with Whisper: {str(e)}")

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=3005)
