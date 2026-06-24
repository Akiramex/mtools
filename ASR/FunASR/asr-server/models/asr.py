import logging

from models.base import ModelRunner

logger = logging.getLogger(__name__)


class AsrModel:
    def __init__(self, runner: ModelRunner):
        self.runner = runner

    async def transcribe(self, audio_path: str, batch_size_s: int = 300) -> dict:
        results = await self.runner.generate(
            input=audio_path, batch_size_s=batch_size_s
        )
        if not results:
            return {}
        logger.info("ASR transcribed: %d chars", len(results[0].get("text", "")))
        return results[0] if isinstance(results, list) else results
