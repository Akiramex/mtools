import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field

from config import TaskQueueConfig

logger = logging.getLogger(__name__)


@dataclass
class TaskInfo:
    task_id: str
    status: str = "pending"
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    error: str | None = None
    result: dict | None = None
    audio_path: str = ""


class TaskManager:
    def __init__(self, config: TaskQueueConfig):
        self.config = config
        self.queue: asyncio.Queue[TaskInfo] = asyncio.Queue()
        self.tasks: dict[str, TaskInfo] = {}
        self._worker_tasks: list[asyncio.Task] = []
        self._cleanup_task: asyncio.Task | None = None

    def start(self, process_fn):
        self._process_fn = process_fn
        for i in range(self.config.max_workers):
            t = asyncio.create_task(self._worker(), name=f"task-worker-{i}")
            self._worker_tasks.append(t)
        self._cleanup_task = asyncio.create_task(self._cleanup(), name="task-cleanup")
        logger.info("TaskManager started with %d workers.", self.config.max_workers)

    async def stop(self):
        for t in self._worker_tasks:
            t.cancel()
        if self._cleanup_task:
            self._cleanup_task.cancel()
        logger.info("TaskManager stopped.")

    async def submit(self, audio_path: str) -> str:
        task_id = uuid.uuid4().hex[:16]
        task = TaskInfo(task_id=task_id, audio_path=audio_path)
        self.tasks[task_id] = task
        await self.queue.put(task)
        logger.info("Task submitted: %s", task_id)
        return task_id

    def get_task(self, task_id: str) -> TaskInfo | None:
        return self.tasks.get(task_id)

    async def _worker(self):
        while True:
            try:
                task = await self.queue.get()
                task.status = "processing"
                logger.info("Processing task: %s", task.task_id)
                try:
                    await self._process_fn(task)
                    task.status = "completed"
                    task.finished_at = time.time()
                    logger.info("Task completed: %s", task.task_id)
                except Exception as e:
                    task.status = "failed"
                    task.error = str(e)
                    task.finished_at = time.time()
                    logger.error("Task failed: %s — %s", task.task_id, e, exc_info=True)
                finally:
                    self.queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Worker error: %s", e, exc_info=True)

    async def _cleanup(self):
        while True:
            try:
                await asyncio.sleep(self.config.cleanup_interval_sec)
                now = time.time()
                expired = [
                    tid
                    for tid, t in self.tasks.items()
                    if t.finished_at and (now - t.finished_at) > self.config.result_ttl_sec
                ]
                for tid in expired:
                    del self.tasks[tid]
                if expired:
                    logger.info("Cleaned up %d expired tasks", len(expired))
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Cleanup error: %s", e, exc_info=True)
