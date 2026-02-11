import asyncio
import json
import logging
from typing import Dict, Deque
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
            model_path: str = "base",
            samplerate: int = 48000,
            target_samplerate: int = 16000,
            channels: int = 1,
            device="cpu"
    ):
        self.host = host
        self.port = port
        self.model_path = model_path
        self.samplerate = samplerate
        self.target_samplerate = target_samplerate
        self.channels = channels
        self.buffer_lock = asyncio.Lock()
        self.transcription_buffer = Deque()
        self.device = device

        self.sessions: Dict[UUID, STTSession] = {}
        self.decoder = opuslib.Decoder(self.samplerate, self.channels)

        self._lock = threading.Lock()
        self._shutdown_event = asyncio.Event()

        self.websocket = None

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

    async def handle_mic(self, user_id: UUID, pcm_raw: bytes, websocket: ServerConnection):
        try:
            if user_id not in self.sessions.keys():
                logger.warning(f"Mic packet for user without session: {user_id}")
                return

            with self._lock:
                session = self.sessions[user_id]

            if pcm_raw is None:
                logger.warning(f"no pcm data in Mic packet for user: {user_id}")
                return

            pcm_data = self.decoder.decode(pcm_raw, self.samplerate)

            audio_array = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0

            if self.samplerate != self.target_samplerate:
                audio_array = self.resample_audio(audio_array)

            session.audio_queue.put(audio_array)

        except Exception as e:
            logger.error(f"Error decoding audio for {user_id}: {e}")

    async def handle_connect(self, user_id: UUID, websocket: ServerConnection):
        if user_id is None:
            logger.warning(f"User Id was not specified in connection packet")
            return
        session = STTSession(
            user_id=user_id,
            language="ru",  # или другой язык
            model_path=self.model_path,
            device=self.device,
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

    async def handle_disconnect(self, user_id: UUID, websocket: ServerConnection):
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

    async def check_secondary_buffer(self, websocket: ServerConnection):
        """Проверка и обработка данных из второстепенного буфера"""
        async with self.buffer_lock:
            if not self.transcription_buffer:
                return

            # Обрабатываем все накопленные данные из буфера
            while self.transcription_buffer:
                packet = self.transcription_buffer.popleft()
                await websocket.send(packet)

    async def handle_websocket(self, websocket: ServerConnection):
        """Обработка WebSocket соединения"""
        self.websocket = websocket
        try:
            websocket_task = asyncio.create_task(self._websocket_receiver(websocket))

            while not self._shutdown_event.is_set():
                try:
                    done, pending = await asyncio.wait(
                        [websocket_task],
                        timeout=0.1,  # Проверяем буфер каждые 100мс
                        return_when=asyncio.FIRST_COMPLETED
                    )

                    # logger.info("before checking buffer")
                    await self.check_secondary_buffer(websocket)

                    if websocket_task in done:
                        # Если задача завершена, перезапускаем её
                        websocket_task = asyncio.create_task(self._websocket_receiver(websocket))
                except asyncio.CancelledError:
                    break
        except Exception as e:
            logger.error(f"Error in WebSocket handler: {e}")

    async def _websocket_receiver(self, websocket: ServerConnection):
        """Отдельная задача для приема сообщений из WebSocket"""
        try:
            data_bytes: bytes = await websocket.recv()
            user_id_str = data_bytes[:36].decode("utf-8")
            user_id = UUID(user_id_str)
            if len(data_bytes) == 36:
                await self.handle_mic(user_id, bytes([]), websocket)
                return
            additional_data = data_bytes[36:]
            if len(additional_data) > 1:
                await self.handle_mic(user_id, additional_data, websocket)
            elif additional_data == b'\x00':
                await self.handle_disconnect(user_id, websocket)
            else:
                await self.handle_connect(user_id, websocket)
        except ConnectionClosedOK:
            self._shutdown_event.set()
            logger.info(f"Connection closed normally")
        except Exception as e:
            logger.error(f"Error processing WebSocket message: {e}")


    def _on_transcription(self, user_id: UUID, result):
        """Callback для отправки транскрипции клиенту"""
        try:
            message = json.dumps({
                "type": "transcription",
                "userId": str(user_id),
                "result": result
            })

            logger.info("Sending " + message)
            self.transcription_buffer.append(message)

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
