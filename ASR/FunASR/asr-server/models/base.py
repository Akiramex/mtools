import asyncio
import functools
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

logger = logging.getLogger(__name__)

_proc_model = None


def create_model_from_config(**kwargs):
    """Top-level factory for creating FunASR models in worker processes."""
    from funasr import AutoModel

    return AutoModel(**kwargs)


def _init_proc_model(factory_fn):
    global _proc_model
    _proc_model = factory_fn()


def _proc_generate(*args, **kwargs):
    return _proc_model.generate(*args, **kwargs)


class ModelRunner:
    """Wraps a model with semaphore-guarded execution (thread or process pool)."""

    def __init__(self, model=None, semaphore=None, executor=None,
                 model_factory=None, max_workers=None):
        self.model = model
        self.sem = semaphore
        self._use_process = model_factory is not None

        if self._use_process:
            workers = max_workers or (semaphore._value if semaphore else 2)
            self.executor = ProcessPoolExecutor(
                max_workers=workers,
                initializer=_init_proc_model,
                initargs=(model_factory,),
                mp_context=multiprocessing.get_context("spawn"),
            )
        else:
            self.executor = executor

    async def run(self, fn, *args, **kwargs):
        loop = asyncio.get_running_loop()
        async with self.sem:
            return await loop.run_in_executor(
                self.executor, functools.partial(fn, *args, **kwargs)
            )

    async def generate(self, *args, **kwargs):
        loop = asyncio.get_running_loop()
        async with self.sem:
            if self._use_process:
                return await loop.run_in_executor(
                    self.executor, functools.partial(_proc_generate, *args, **kwargs)
                )
            else:
                return await loop.run_in_executor(
                    self.executor, functools.partial(self.model.generate, *args, **kwargs)
                )

    def shutdown(self):
        if self._use_process and self.executor:
            self.executor.shutdown(wait=False, cancel_futures=True)
