"""
Resource monitoring utilities for tracking CPU and memory usage during benchmarks.
Supports both native processes and Docker containers.
"""

import psutil
import threading
import time
import subprocess
import re
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class ResourceUsage:
    """Resource usage statistics."""
    cpu_percent_avg: float = 0.0
    cpu_percent_max: float = 0.0
    memory_mb_avg: float = 0.0
    memory_mb_max: float = 0.0
    memory_percent_avg: float = 0.0
    memory_percent_max: float = 0.0
    samples: int = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "cpu_percent_avg": round(self.cpu_percent_avg, 2),
            "cpu_percent_max": round(self.cpu_percent_max, 2),
            "memory_mb_avg": round(self.memory_mb_avg, 2),
            "memory_mb_max": round(self.memory_mb_max, 2),
            "memory_percent_avg": round(self.memory_percent_avg, 2),
            "memory_percent_max": round(self.memory_percent_max, 2),
            "samples": self.samples
        }


class ResourceMonitor:
    """
    Monitor CPU and memory usage in a background thread.

    Usage:
        monitor = ResourceMonitor(interval=0.1)
        monitor.start()
        # ... do work ...
        stats = monitor.stop()
    """

    def __init__(self, interval: float = 0.1):
        """
        Initialize the resource monitor.

        Args:
            interval: Sampling interval in seconds (default: 0.1s = 100ms)
        """
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._process = psutil.Process()

        # Accumulated stats
        self._cpu_samples = []
        self._memory_mb_samples = []
        self._memory_percent_samples = []

    def _monitor_loop(self):
        """Background monitoring loop."""
        # Get initial CPU percent (first call returns 0.0)
        self._process.cpu_percent(interval=None)

        while not self._stop_event.is_set():
            try:
                # CPU percent (per-process)
                cpu = self._process.cpu_percent(interval=None)
                self._cpu_samples.append(cpu)

                # Memory usage
                mem_info = self._process.memory_info()
                memory_mb = mem_info.rss / (1024 * 1024)  # Convert bytes to MB
                self._memory_mb_samples.append(memory_mb)

                # Memory percent
                mem_percent = self._process.memory_percent()
                self._memory_percent_samples.append(mem_percent)

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                # Process might have ended or we lost access
                break

            time.sleep(self.interval)

    def start(self):
        """Start monitoring in a background thread."""
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("Monitor is already running")

        # Reset state
        self._stop_event.clear()
        self._cpu_samples = []
        self._memory_mb_samples = []
        self._memory_percent_samples = []

        # Start monitoring thread
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self) -> ResourceUsage:
        """
        Stop monitoring and return statistics.

        Returns:
            ResourceUsage object with average and max values
        """
        if self._thread is None or not self._thread.is_alive():
            # Not running, return empty stats
            return ResourceUsage()

        # Signal stop and wait for thread
        self._stop_event.set()
        self._thread.join(timeout=5.0)

        # Calculate statistics
        if not self._cpu_samples:
            return ResourceUsage()

        stats = ResourceUsage(
            cpu_percent_avg=sum(self._cpu_samples) / len(self._cpu_samples),
            cpu_percent_max=max(self._cpu_samples),
            memory_mb_avg=sum(self._memory_mb_samples) / len(self._memory_mb_samples),
            memory_mb_max=max(self._memory_mb_samples),
            memory_percent_avg=sum(self._memory_percent_samples) / len(self._memory_percent_samples),
            memory_percent_max=max(self._memory_percent_samples),
            samples=len(self._cpu_samples)
        )

        return stats

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False


class BenchmarkTimerWithResources:
    """
    Context manager that times execution and monitors resources.

    Usage:
        with BenchmarkTimerWithResources() as timer:
            # ... do work ...

        print(f"Time: {timer.elapsed_ms}ms")
        print(f"CPU: {timer.resources.cpu_percent_avg}%")
        print(f"Memory: {timer.resources.memory_mb_avg}MB")
    """

    def __init__(self, interval: float = 0.1, num_runs: int = 1):
        """
        Initialize timer with resource monitoring.

        Args:
            interval: Resource sampling interval in seconds
            num_runs: Number of runs to average (for multiple iterations)
        """
        self.interval = interval
        self.num_runs = num_runs
        self.elapsed_ms: float = 0.0
        self.resources = ResourceUsage()
        self._monitor = ResourceMonitor(interval=interval)
        self._start_time: Optional[float] = None

    def __enter__(self):
        """Start timing and resource monitoring."""
        self._monitor.start()
        self._start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and resource monitoring."""
        # Calculate elapsed time
        if self._start_time is not None:
            elapsed_seconds = time.time() - self._start_time
            self.elapsed_ms = elapsed_seconds * 1000.0

        # Get resource stats
        self.resources = self._monitor.stop()

        return False

    def get_results_dict(self) -> Dict:
        """Get combined timing and resource results."""
        return {
            "time_ms": round(self.elapsed_ms, 2),
            **self.resources.to_dict()
        }


def monitor_external_process(pid: int, interval: float = 0.1) -> ResourceMonitor:
    """
    Monitor an external process (e.g., database server).

    Args:
        pid: Process ID to monitor
        interval: Sampling interval in seconds

    Returns:
        ResourceMonitor configured for the external process
    """
    monitor = ResourceMonitor(interval=interval)
    monitor._process = psutil.Process(pid)
    return monitor


class DockerStatsMonitor:
    """
    Monitor Docker container resource usage using docker stats command.
    This is more reliable than trying to monitor container processes directly.
    """

    def __init__(self, container_name: str, interval: float = 0.5):
        """
        Initialize Docker stats monitor.

        Args:
            container_name: Name or ID of the Docker container
            interval: Sampling interval in seconds (min 0.5s for docker stats)
        """
        self.container_name = container_name
        self.interval = max(interval, 0.5)  # Docker stats is slow, min 0.5s
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Accumulated stats
        self._cpu_samples = []
        self._memory_mb_samples = []

    def _parse_docker_stats(self, stats_output: str) -> Optional[Tuple[float, float]]:
        """
        Parse docker stats output.

        Expected format: "45.23%,156.7MiB / 2GiB"
        Returns: (cpu_percent, memory_mb)
        """
        try:
            parts = stats_output.strip().split(',')
            if len(parts) != 2:
                return None

            # Parse CPU percentage
            cpu_str = parts[0].strip().rstrip('%')
            cpu_percent = float(cpu_str)

            # Parse memory usage (before the '/')
            mem_str = parts[1].strip().split('/')[0].strip()

            # Convert to MB
            if 'GiB' in mem_str or 'GB' in mem_str:
                memory_mb = float(re.findall(r'[\d.]+', mem_str)[0]) * 1024
            elif 'MiB' in mem_str or 'MB' in mem_str:
                memory_mb = float(re.findall(r'[\d.]+', mem_str)[0])
            elif 'KiB' in mem_str or 'KB' in mem_str:
                memory_mb = float(re.findall(r'[\d.]+', mem_str)[0]) / 1024
            else:
                # Assume bytes
                memory_mb = float(re.findall(r'[\d.]+', mem_str)[0]) / (1024 * 1024)

            return cpu_percent, memory_mb

        except (ValueError, IndexError, AttributeError):
            return None

    def _monitor_loop(self):
        """Background monitoring loop using docker stats."""
        while not self._stop_event.is_set():
            try:
                # Run docker stats command (non-streaming, single snapshot)
                result = subprocess.run(
                    [
                        "docker", "stats", self.container_name,
                        "--no-stream", "--no-trunc",
                        "--format", "{{.CPUPerc}},{{.MemUsage}}"
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5.0,
                    check=True
                )

                stats = self._parse_docker_stats(result.stdout)
                if stats:
                    cpu_percent, memory_mb = stats
                    self._cpu_samples.append(cpu_percent)
                    self._memory_mb_samples.append(memory_mb)

            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
                # Docker command failed, skip this sample
                pass

            time.sleep(self.interval)

    def start(self):
        """Start monitoring Docker container."""
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("Monitor is already running")

        # Reset state
        self._stop_event.clear()
        self._cpu_samples = []
        self._memory_mb_samples = []

        # Start monitoring thread
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self) -> ResourceUsage:
        """
        Stop monitoring and return statistics.

        Returns:
            ResourceUsage object with average and max values
        """
        if self._thread is None or not self._thread.is_alive():
            return ResourceUsage()

        # Signal stop and wait
        self._stop_event.set()
        self._thread.join(timeout=10.0)

        # Calculate statistics
        if not self._cpu_samples:
            return ResourceUsage()

        stats = ResourceUsage(
            cpu_percent_avg=sum(self._cpu_samples) / len(self._cpu_samples),
            cpu_percent_max=max(self._cpu_samples),
            memory_mb_avg=sum(self._memory_mb_samples) / len(self._memory_mb_samples),
            memory_mb_max=max(self._memory_mb_samples),
            memory_percent_avg=0.0,  # Not available from docker stats
            memory_percent_max=0.0,   # Not available from docker stats
            samples=len(self._cpu_samples)
        )

        return stats

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False


def get_docker_container_process_pid(container_name: str, process_name: str = None) -> Optional[int]:
    """
    Try to get the PID of a process running inside a Docker container.

    This works by finding the container's main process and optionally searching
    for a specific child process by name.

    Args:
        container_name: Name or ID of the Docker container
        process_name: Optional process name to search for (e.g., 'postgres', 'clickhouse')

    Returns:
        PID of the process, or None if not found
    """
    try:
        # Get the container's main PID
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Pid}}", container_name],
            capture_output=True,
            text=True,
            check=True,
            timeout=5.0
        )
        container_pid = int(result.stdout.strip())

        if container_pid == 0:
            return None

        # If no specific process name, return the main PID
        if not process_name:
            return container_pid

        # Search for child processes
        try:
            container_proc = psutil.Process(container_pid)
            for child in container_proc.children(recursive=True):
                if process_name.lower() in child.name().lower():
                    return child.pid
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

        # Fall back to main PID if child not found
        return container_pid

    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError, FileNotFoundError):
        return None


def monitor_docker_container(container_name: str, interval: float = 0.5) -> DockerStatsMonitor:
    """
    Create a monitor for a Docker container using docker stats.

    This is the recommended way to monitor Docker containers as it's more
    reliable than trying to access container processes directly.

    Args:
        container_name: Name or ID of the Docker container
        interval: Sampling interval in seconds (min 0.5s)

    Returns:
        DockerStatsMonitor instance ready to start

    Example:
        monitor = monitor_docker_container("my-postgres")
        monitor.start()
        # ... database work ...
        stats = monitor.stop()
    """
    return DockerStatsMonitor(container_name, interval)
