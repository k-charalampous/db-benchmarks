from typing import Dict, Optional, Tuple, TypedDict

from resource_monitor import ResourceMonitor, monitor_docker_container


class QueryTypes(TypedDict):
    simple_where: str
    complex_where: str
    pagination_early: str
    pagination_deep: str
    nested_array_filter: str
    simple_nested_agg: str
    deep_nested_agg: str
    array_aggregation: str
    complex_where_agg: str
    seller_rating_agg: str


class Strategy:
    queries: QueryTypes = {}
    docker_container: Optional[str] = (
        None  # Optional Docker container name for server-side monitoring
    )
    _strategy_monitor: Optional[any] = None  # Shared monitor for entire strategy run

    def execute_query(self, query_type: str) -> int:
        """Execute the given query and return the number of rows fetched"""
        raise NotImplementedError("Subclasses must implement this method")

    def start_monitoring(self):
        """Start monitoring for the entire strategy benchmark run."""
        if self._strategy_monitor is not None:
            return  # Already monitoring

        if self.docker_container:
            self._strategy_monitor = monitor_docker_container(
                self.docker_container, interval=0.5
            )
        else:
            self._strategy_monitor = ResourceMonitor(interval=0.1)

        self._strategy_monitor.start()

    def stop_monitoring(self) -> Dict:
        """Stop monitoring and return aggregate resource usage."""
        if self._strategy_monitor is None:
            return {
                "cpu_percent_avg": 0.0,
                "cpu_percent_max": 0.0,
                "memory_mb_avg": 0.0,
                "memory_mb_max": 0.0,
                "memory_percent_avg": 0.0,
                "memory_percent_max": 0.0,
                "samples": 0,
            }

        resources = self._strategy_monitor.stop()
        self._strategy_monitor = None
        return resources.to_dict()

    def run_query(self, query_type: str, num_runs=5) -> Tuple[float, int, Dict]:
        """
        Run query and return time + row count + empty resource dict.
        """
        times = []
        result_count = 0

        for _ in range(num_runs):
            exec_time, count = self.execute_query(query_type)
            times.append(exec_time)
            result_count = (
                count  # Take the last result count (should be same for all runs)
            )

        if len(times) == 0:
            print(query_type)
            print(self.queries[query_type])

        avg_time = sum(times) / len(times)

        # Return empty resource dict - resources are now tracked at strategy level
        empty_resources = {
            "cpu_percent_avg": 0.0,
            "cpu_percent_max": 0.0,
            "memory_mb_avg": 0.0,
            "memory_mb_max": 0.0,
            "memory_percent_avg": 0.0,
            "memory_percent_max": 0.0,
            "samples": 0,
        }
        return avg_time, result_count, empty_resources
