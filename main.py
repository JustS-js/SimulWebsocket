import asyncio
import json
import logging
from typing import Dict
from uuid import UUID
import opuslib

import numpy as np
from websockets import ServerConnection
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosedOK
import threading
import librosa

from session import STTSession

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebSocketSTTServer:
    def __init__(
            self,
            host: str = "0.0.0.0",
            port: int = 8765,
            model_path: str = "ggml-model",
            samplerate: int = 48000,
            target_samplerate: int = 16000,
            channels: int = 1
    ):
        self.host = host
        self.port = port
        self.model_path = model_path
        self.samplerate = samplerate
        self.target_samplerate = target_samplerate
        self.channels = channels

        self.sessions: Dict[UUID, STTSession] = {}
        self.decoder = opuslib.Decoder(self.samplerate, self.channels)

        self._lock = threading.Lock()
        self._shutdown_event = asyncio.Event()

    def resample_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Ресамплинг аудио с 48kHz до 16kHz"""
        if self.samplerate == self.target_samplerate:
            return audio_data

        return librosa.resample(
            y=audio_data,
            orig_sr=self.samplerate,
            target_sr=self.target_samplerate
            #res_type='kaiser_best'  # или 'soxr_hq' для высокого качества
        )

    async def handle_mic(self, packet: dict, websocket: ServerConnection):
        user_id = packet.get("userId", None)
        try:
            if user_id not in self.sessions.keys():
                logger.warning(f"Mic packet for user without session: {user_id}")
                return

            with self._lock:
                session = self.sessions[user_id]

            pcm_raw = packet.get("pcm", None)
            if pcm_raw is None:
                logger.warning(f"no pcm data in Mic packet for user: {user_id}")
                return

            pcm_data = self.decoder.decode(packet["pcm"], self.samplerate)

            audio_array = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0

            if self.samplerate != self.target_samplerate:
                audio_array = self.resample_audio(audio_array)

            session.audio_queue.put(audio_array)

        except Exception as e:
            logger.error(f"Error decoding audio for {user_id}: {e}")

    async def handle_connect(self, packet: dict, websocket: ServerConnection):
        user_id: UUID = packet.get("userId", None)
        if user_id is None:
            logger.warning(f"User Id was not specified in connection packet")
            return
        session = STTSession(
            user_id=user_id,
            language="ru",  # или другой язык
            model_path=self.model_path,
            translate=False,  # или True если нужен перевод
            samplerate=self.target_samplerate,
            block_duration=0.5,
            chunk_duration=2,
            channels=self.channels,
            callback=lambda result: self._on_transcription(user_id, result)
        )

        with self._lock:
            self.sessions[user_id] = session

        session.start()

    async def handle_disconnect(self, packet: dict, websocket: ServerConnection):
        user_id: UUID = packet.get("userId", None)
        if user_id is None:
            logger.warning(f"User Id was not specified in disconnection packet")
            return

        with self._lock:
            if user_id not in self.sessions.keys():
                logger.warning(f"User Id specified in disconnection packet does not have a dedicated session")
                return
            session = self.sessions.pop(user_id)

        session.stop()

        if session._thread.is_alive():
            session._thread.join(timeout=5)

        logger.info(f"Session cleaned up for user {user_id}")

    async def handle_websocket(self, websocket: ServerConnection):
        """Обработка WebSocket соединения"""
        try:
            async for message in websocket:
                packet: dict = json.loads(message)
                packet_type = packet.get("type", None)
                if packet_type == "mic":
                    await self.handle_mic(packet, websocket)
                elif packet_type == "connect":
                    await self.handle_connect(packet, websocket)
                elif packet_type == "disconnect":
                    await self.handle_disconnect(packet, websocket)
        except ConnectionClosedOK:
            logger.info(f"Connection closed normally")
        except Exception as e:
            logger.error(f"Error in WebSocket handler: {e}")


    def _on_transcription(self, user_id: UUID, websocket: ServerConnection, result):
        """Callback для отправки транскрипции клиенту"""
        try:
            message = json.dumps({
                "type": "transcription",
                "userId": str(user_id),
                "result": result
            })

            asyncio.run_coroutine_threadsafe(
                websocket.send(message),
                asyncio.get_event_loop()
            )
        except Exception as e:
            logger.error(f"Error sending transcription for user {user_id}: {e}")

    async def start(self):
        """Запуск WebSocket сервера"""
        logger.info(f"Starting WebSocket STT server on {self.host}:{self.port}")

        async with serve(
                self.handle_websocket,
                self.host,
                self.port,
                ping_interval=20,
                ping_timeout=20,
                max_size=2 ** 20  # 1MB max message size
        ) as s:
            logger.info(f"Server started. Waiting for connections...")
            await self._shutdown_event.wait()
            await s.wait_closed()

    def shutdown(self):
        with self._lock:
            for user_id, session in list(self.sessions.items()):
                try:
                    session.stop()
                    if session._thread.is_alive():
                        session._thread.join(timeout=2)
                    logger.info(f"Session stopped for user {user_id}")
                except Exception as e:
                    logger.error(f"Error stopping session for {user_id}: {e}")

        self._shutdown_event.set()


if __name__ == "__main__":
    server = WebSocketSTTServer(
        host="0.0.0.0",
        port=8765,
        model_path="ggml-base",
        samplerate=48000,
        target_samplerate=16000,
        channels=1
    )

    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        logger.info("Gracefully shutting down...")
        server.shutdown()
        logger.info("Shutdown successful")