import websockets
import websockets_proxy
import json
import base64
import logging
import os
from typing import Callable, Dict
import asyncio
from urllib.parse import urlparse

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class OpenAIRealtimeAudioTextClient:
    def __init__(self, api_key: str, model: str = "gpt-realtime-1.5"):
        self.api_key = api_key
        self.model = model
        self.ws = None
        self.session_id = None
        self.base_url = "wss://api.openai.com/v1/realtime"
        self.receive_task = None
        self.handlers: Dict[str, Callable[[dict], asyncio.Future]] = {}

    async def connect(self):
        """Connect to OpenAI's realtime API and configure a transcription session."""
        proxy_url = os.environ.get('ALL_PROXY') or os.environ.get('all_proxy')
        logger.info(f"Proxy detection - ALL_PROXY: {os.environ.get('ALL_PROXY')}, all_proxy: {os.environ.get('all_proxy')}, using: {proxy_url}")

        headers = {"Authorization": f"Bearer {self.api_key}"}
        url = f"{self.base_url}?model={self.model}"

        if proxy_url:
            parsed = urlparse(proxy_url)
            if parsed.scheme == 'socks5':
                from python_socks.async_.asyncio import Proxy
                proxy = Proxy.from_url(proxy_url)
                self.ws = await websockets_proxy.proxy_connect(url, proxy=proxy, extra_headers=headers, open_timeout=15)
            else:
                logger.warning(f"Unsupported proxy scheme: {parsed.scheme}, falling back to direct connection")
                self.ws = await websockets.connect(url, extra_headers=headers, open_timeout=15)
        else:
            self.ws = await websockets.connect(url, extra_headers=headers, open_timeout=15)

        # Wait for session creation
        response = await self.ws.recv()
        response_data = json.loads(response)
        if response_data["type"] == "session.created":
            self.session_id = response_data["session"]["id"]
            logger.info(f"Session created with ID: {self.session_id}")

            await self.ws.send(json.dumps({
                "type": "session.update",
                "session": {
                    "type": "realtime",
                    "output_modalities": ["text"],
                    "audio": {
                        "input": {
                            "format": {"type": "audio/pcm", "rate": 24000},
                            "transcription": {"model": "gpt-4o-mini-transcribe"},
                            "turn_detection": None,
                        }
                    },
                }
            }))

        self.register_handler("default", self.default_handler)
        self.receive_task = asyncio.create_task(self.receive_messages())

    async def receive_messages(self):
        try:
            async for message in self.ws:
                data = json.loads(message)
                message_type = data.get("type", "default")
                handler = self.handlers.get(message_type, self.handlers.get("default"))
                if handler:
                    await handler(data)
        except websockets.exceptions.ConnectionClosed as e:
            logger.error(f"OpenAI WebSocket connection closed: {e}")
        except Exception as e:
            logger.error(f"Error in receive_messages: {e}", exc_info=True)

    def register_handler(self, message_type: str, handler: Callable[[dict], asyncio.Future]):
        self.handlers[message_type] = handler

    async def default_handler(self, data: dict):
        logger.warning(f"Unhandled message type received from OpenAI: {data.get('type', 'unknown')}")

    async def send_audio(self, audio_data: bytes):
        if self.ws and self.ws.open:
            await self.ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(audio_data).decode('utf-8')
            }))
        else:
            logger.error("WebSocket is not open. Cannot send audio.")

    async def commit_audio(self):
        if self.ws and self.ws.open:
            await self.ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
            logger.info("Sent input_audio_buffer.commit message to OpenAI")
        else:
            logger.error("WebSocket is not open. Cannot commit audio.")

    async def clear_audio_buffer(self):
        if self.ws and self.ws.open:
            await self.ws.send(json.dumps({"type": "input_audio_buffer.clear"}))
            logger.info("Sent input_audio_buffer.clear message to OpenAI")
        else:
            logger.error("WebSocket is not open. Cannot clear audio buffer.")

    async def close(self):
        if self.ws:
            await self.ws.close()
            logger.info("Closed OpenAI WebSocket connection")
        if self.receive_task:
            self.receive_task.cancel()
            try:
                await self.receive_task
            except asyncio.CancelledError:
                pass
