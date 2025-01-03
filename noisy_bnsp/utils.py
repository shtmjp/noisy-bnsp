import datetime
from logging import (
    CRITICAL,
    DEBUG,
    INFO,
    FileHandler,
    Formatter,
    Logger,
    StreamHandler,
    getLogger,
)
from pathlib import Path
from typing import Callable


def get_my_logger(name: str, is_stream: bool = False):
    logger = getLogger(name)
    logger.setLevel(DEBUG)

    dt = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    Path("./log/").mkdir(exist_ok=True)
    handler = FileHandler(f"./log/{name}_{dt}.log")
    handler.setFormatter(
        Formatter(
            "%(asctime)s %(name)s:%(lineno)s %(funcName)s [%(levelname)s]: %(message)s"
        )
    )
    handler.setLevel(DEBUG)
    logger.addHandler(handler)

    if is_stream:
        ch = StreamHandler()
        ch.setLevel(INFO)
        logger.addHandler(ch)

    logger.propagate = False
    return logger


def get_mocked_logger(name: str):
    logger = getLogger(name)
    logger.setLevel(CRITICAL)
    return logger


class OptimizeHistory:
    """Callback for scipy.optimize.minimize"""

    def __init__(self, objective: Callable, logger: Logger):
        self.optimize_count = 0
        self.objective = objective
        self.logger = logger
        self.xs: list[float] = []
        self.objective_values: list[float] = []

    def callback(self, x) -> None:
        self.optimize_count += 1
        val = self.objective(x)
        self.xs.append(x)
        self.objective_values.append(val)
        self.logger.debug(f"count: {self.optimize_count}, x: {x}, value: {val}")


# https://blog.ysk.im/x/joblib-with-progress-bar
# https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution/58936697#58936697
# @contextlib.contextmanager
# def tqdm_joblib(total: Optional[int] = None, **kwargs):
#     pbar = tqdm(total=total, miniters=1, smoothing=0, **kwargs)

#     class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
#         def __call__(self, *args, **kwargs):
#             pbar.update(n=self.batch_size)
#             return super().__call__(*args, **kwargs)

#     old_batch_callback = joblib.parallel.BatchCompletionCallBack
#     joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback

#     try:
#         yield pbar
#     finally:
#         joblib.parallel.BatchCompletionCallBack = old_batch_callback
#         pbar.close()
