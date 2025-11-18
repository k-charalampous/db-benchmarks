import argparse
from benchmarks import Benchmark

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run database query benchmarks"
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=100000,
        help="Size of the dataset to benchmark (default: 100000)",
    )

    args = parser.parse_args()

    PG_CONN = "postgresql://benchmark_user:benchmark_pass@localhost:5432/benchmark_db"
    PG_CONN_18 = (
        "postgresql://benchmark_user:benchmark_pass@localhost:5433/benchmark_db"
    )
    MONGO_CONN = "mongodb://localhost:27017/"
    MONGO_DB = "benchmark_db"

    # Docker container names for server-side resource monitoring
    PG_DOCKER_CONTAINER = "benchmark_postgres17"
    PG_18_DOCKER_CONTAINER = "benchmark_postgres18"
    CLICKHOUSE_DOCKER_CONTAINER = "benchmark_clickhouse"
    MONGO_DOCKER_CONTAINER = "benchmark_mongodb"

    benchmark = Benchmark(
        PG_CONN,
        PG_CONN_18,
        MONGO_CONN,
        MONGO_DB,
        pg_docker_container=PG_DOCKER_CONTAINER,
        pg_18_docker_container=PG_18_DOCKER_CONTAINER,
        clickhouse_docker_container=CLICKHOUSE_DOCKER_CONTAINER,
        mongo_docker_container=MONGO_DOCKER_CONTAINER,
    )

    print(f"\nRunning benchmark with dataset size: {args.dataset_size:,}")
    benchmark.run_full_benchmark(dataset_size=args.dataset_size)

    print("\n✓ Non-aggregated query benchmark complete!")
