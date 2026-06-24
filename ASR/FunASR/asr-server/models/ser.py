import logging

from models.base import ModelRunner

logger = logging.getLogger(__name__)


class SerModel:
    def __init__(self, runner: ModelRunner):
        self.runner = runner

    async def detect_emotion(self, audio_path: str) -> dict:
        def _detect(path):
            result = self.runner.model.generate(
                input=path,
                granularity="utterance",
                extract_embedding=False,
            )
            item = result[0] if isinstance(result, list) else result
            labels = item.get("labels", [])
            scores = item.get("scores", [])
            score_map = {str(label): float(score) for label, score in zip(labels, scores)}
            max_label = str(labels[max(range(len(scores)), key=lambda i: scores[i])]) if scores else "unknown"
            return {"emotion": max_label, "scores": score_map}

        result = await self.runner.run(_detect, audio_path)
        logger.info("SER detected: %s", result["emotion"])
        return result
