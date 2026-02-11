import asyncio
import logging

from server import WebSocketSTTServer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    server = WebSocketSTTServer(
        host="0.0.0.0",
        port=8765,
        model_path="small",
        samplerate=48000,
        target_samplerate=16000,
        channels=1,
        device="cuda"
    )

    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        logger.info("Gracefully shutting down...")
        server.shutdown()
        logger.info("Shutdown successful")
