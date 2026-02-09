import queue
from uuid import UUID
from queue import Queue
import numpy as np
import threading
import logging

from simulstreaming_whisper import SimulWhisperASR, SimulWhisperOnline

logger = logging.getLogger(__name__)


class STTSession:
    def __init__(
            self,
            user_id: UUID,
            language: str = "en",
            model_path: str = "ggml-base",
            translate: bool = True,
            samplerate: int = 16000,
            block_duration: float = 0.5,
            chunk_duration: float = 2,
            channels: int = 1,
            callback = None
    ):
        self.user_id = user_id
        self.language = language
        self.model_path = model_path
        self.should_translate = translate
        self.audio_queue = Queue()
        self.audio_buffer = []
        self.frames_per_block = int(samplerate * block_duration)
        self.frames_per_chunk = int(samplerate * chunk_duration)
        self.channels = channels

        self.asr = SimulWhisperASR(self.language, self.model_path)
        self.model = SimulWhisperOnline(self.asr)

        self._callback = callback
        self._is_running = False

        self._thread = threading.Thread(
            target=self.transcription_task,
            name=f"stt-session-task-{self.user_id}",
            daemon=True
        )
        self._stop_event = threading.Event()

    def start(self):
        if self._is_running:
            return
        self._is_running = True
        self._thread.start()
        logger.debug(f"Started transcription thread for user {self.user_id}")
    
    def stop(self):
        if not self._is_running:
            return
        self._stop_event.set()
        self.audio_queue.put(np.array([]))

    def transcription_task(self):
        while not self._stop_event.is_set():
            try:
                try:
                    block = self.audio_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                self.audio_buffer.append(block)

                total_frames = sum(len(b) for b in self.audio_buffer)
                if total_frames >= self.frames_per_chunk:
                    audio_data = np.concatenate(self.audio_buffer)[:self.frames_per_chunk]
                    remaining_frames = total_frames - self.frames_per_chunk
                    if remaining_frames > 0:
                        remaining_data = np.concatenate(self.audio_buffer)[self.frames_per_chunk:]
                        self.audio_buffer = [remaining_data]
                    else:
                        self.audio_buffer.clear()

                    audio_data = audio_data.flatten().astype(np.float32)
                    self.model.insert_audio_chunk(audio_data)

                    result = self.model.process_iter()
                    if self._callback and result:
                        self._callback(result)

            except Exception as e:
                logger.error(f"Error in transcription task for user {self.user_id}: {e}")

        try:
            result = self.model.finish()
            if self._callback and result:
                self._callback(result)

        except Exception as e:
            logger.error(f"Error finishing transcription for user {self.user_id}: {e}")

        self._is_running = False
        logger.debug(f"Transcription thread stopped for user {self.user_id}")
