from collections.abc import Callable
from logging import CRITICAL, Logger, getLogger

import numpy as np


def get_mocked_logger(name: str) -> Logger:
    """Get a logger that does not output anything."""
    logger = getLogger(name)
    logger.setLevel(CRITICAL)
    return logger


class OptimizeHistory:
    """Callback for scipy.optimize.minimize."""

    def __init__(
        self, objective: Callable[[np.ndarray], float], logger: Logger
    ) -> None:
        """Initialize the callback.

        Args:
            objective: Objective function to be optimized.
            logger: Logger to log the optimization history.

        """
        self.optimize_count = 0
        self.objective = objective
        self.logger = logger
        self.xs: list[np.ndarray] = []
        self.objective_values: list[float] = []

    def callback(self, x: np.ndarray) -> None:
        """Call the callback function for scipy.optimize.minimize."""
        self.optimize_count += 1
        val = self.objective(x)
        self.xs.append(x)
        self.objective_values.append(val)
        self.logger.debug("count: %d, x: %s, value: %f", self.optimize_count, x, val)
