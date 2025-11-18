import time
from typing import List


class BenchmarkTimer:
    """Context manager for timing operations with support for multiple runs"""

    def __init__(self, name: str, num_runs: int = 1):
        self.name = name
        self.num_runs = num_runs
        self.start_time = None
        self.end_time = None
        self.times = []

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.end_time = time.time()
        self.times.append(self.end_time - self.start_time)

    @property
    def elapsed(self) -> float:
        """Return the most recent elapsed time"""
        if self.times:
            return self.times[-1]
        return 0.0

    @property
    def average_elapsed(self) -> float:
        """Return the average elapsed time across all runs"""
        if self.times:
            return sum(self.times) / len(self.times)
        return 0.0

    @property
    def all_times(self) -> List[float]:
        """Return all recorded times"""
        return self.times
