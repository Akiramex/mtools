import json
import logging
import time

import numpy as np
from config import SpeakerDbConfig
from models.base import ModelRunner
from scipy.spatial.distance import cosine

logger = logging.getLogger(__name__)


class SvModel:
    def __init__(self, runner: ModelRunner, config: SpeakerDbConfig, db_path: str):
        self.runner = runner
        self.config = config
        self.db_path = db_path
        self._db_cache: dict = {}
        self._db_cache_ts: float = 0.0

    def _load_db_sync(self) -> dict:
        try:
            if not __import__("os").path.exists(self.db_path):
                return {}
            with open(self.db_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error("Failed to load speaker_db: %s", e)
            return {}

    def get_speaker_db(self) -> dict:
        now = time.time()
        if now - self._db_cache_ts > self.config.reload_interval_sec:
            self._db_cache = self._load_db_sync()
            self._db_cache_ts = now
        return self._db_cache

    def save_speaker_db(self, db: dict) -> None:
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(db, f, ensure_ascii=False)
        self._db_cache = db
        self._db_cache_ts = time.time()

    def _extract_embedding_sync(self, audio_path: str) -> np.ndarray:
        result = self.runner.model.generate(input=audio_path, embedding=True)
        return result[0]["spk_embedding"][0].cpu().numpy()

    async def extract_embedding(self, audio_path: str) -> np.ndarray:
        return await self.runner.run(self._extract_embedding_sync, audio_path)

    async def register_speaker(self, audio_path: str, name: str) -> None:
        embedding = await self.extract_embedding(audio_path)
        db = self.get_speaker_db()
        db[name] = embedding.tolist()
        self.save_speaker_db(db)
        logger.info("Registered speaker: %s", name)

    async def identify_speaker(self, audio_path: str) -> tuple[str, float]:
        def _identify(path):
            embedding = self._extract_embedding_sync(path)
            db = self.get_speaker_db()
            best_name = "unknown"
            best_score = 0.0
            for name, ref_emb in db.items():
                if not ref_emb:
                    continue
                arr = np.array(ref_emb, dtype=np.float32)
                sim = 1.0 - cosine(embedding, arr)
                if sim > best_score and sim > self.config.similarity_threshold:
                    best_score = sim
                    best_name = name
            return best_name, float(best_score)

        return await self.runner.run(_identify, audio_path)

    def delete_speaker(self, name: str) -> bool:
        db = self.get_speaker_db()
        if name in db:
            del db[name]
            self.save_speaker_db(db)
            logger.info("Deleted speaker: %s", name)
            return True
        return False

    def list_speakers(self) -> list[str]:
        return list(self.get_speaker_db().keys())
