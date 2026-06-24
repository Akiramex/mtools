import asyncio
import functools
import logging

logger = logging.getLogger(__name__)


class ModelRunner:
    """Wraps a model with semaphore-guarded thread-pool execution."""

    def __init__(self, model, executor, semaphore):
        self.model = model
        self.executor = executor
        self.sem = semaphore

    async def generate(self, *args, **kwargs):
        loop = asyncio.get_running_loop()
        async with self.sem:
            return await loop.run_in_executor(
                self.executor, functools.partial(self.model.generate, *args, **kwargs)
            )


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
