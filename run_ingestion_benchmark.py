#!/usr/bin/env python3
"""
Ingestion Benchmark Runner

This script runs the ingestion benchmark, which measures how fast data can be loaded
into existing database structures from DataFrames.

Features:
- Works with existing tables from the main benchmark
- Loads test data and measures ingestion time
- For ClickHouse strategies, also measures materialized view refresh time
- Cleans up test data after benchmarking, leaving existing data intact

Usage:
    python run_ingestion_benchmark.py [OPTIONS]

This script runs BOTH bulk and high-frequency ingestion benchmarks and generates
comparison visualizations.

Examples:
    # Run both benchmarks with default settings
    python run_ingestion_benchmark.py --dataset-size 1000000 --test-size 10000

    # Customize high-frequency parameters
    python run_ingestion_benchmark.py --dataset-size 1000000 --test-size 10000 --batch-size 100 --batches-per-second 1 --duration 60
"""

import argparse
from ingestion_benchmark import IngestionBenchmark


def main():
    parser = argparse.ArgumentParser(
        description="Run both bulk and high-frequency ingestion benchmarks"
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=1000000,
        help="Size of the EXISTING dataset (for table name resolution). Default: 1000000",
    )

    # Bulk mode arguments
    parser.add_argument(
        "--test-size",
        type=int,
        default=10000,
        help="Size of test data to INSERT during bulk benchmark. Default: 10000",
    )

    # High-frequency mode arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of records per batch in high-frequency mode. Default: 100",
    )
    parser.add_argument(
        "--batches-per-second",
        type=int,
        default=1,
        help="Target number of batches per second in high-frequency mode. Default: 1",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Duration of high-frequency test in seconds. Default: 60",
    )

    args = parser.parse_args()

    # Database connection strings
    PG_CONN = "postgresql://benchmark_user:benchmark_pass@localhost:5432/benchmark_db"
    PG_CONN_18 = "postgresql://benchmark_user:benchmark_pass@localhost:5433/benchmark_db"
    MONGO_CONN = "mongodb://localhost:27017/"
    MONGO_DB = "benchmark_db"

    # Docker container names for server-side resource monitoring
    PG_DOCKER_CONTAINER = "benchmark_postgres17"
    PG_18_DOCKER_CONTAINER = "benchmark_postgres18"
    CLICKHOUSE_DOCKER_CONTAINER = "benchmark_clickhouse"
    MONGO_DOCKER_CONTAINER = "benchmark_mongodb"

    # Create and run benchmark
    benchmark = IngestionBenchmark(
        PG_CONN,
        PG_CONN_18,
        MONGO_CONN,
        MONGO_DB,
        pg_docker_container=PG_DOCKER_CONTAINER,
        pg_18_docker_container=PG_18_DOCKER_CONTAINER,
        clickhouse_docker_container=CLICKHOUSE_DOCKER_CONTAINER,
        mongo_docker_container=MONGO_DOCKER_CONTAINER,
    )

    records_per_sec = args.batch_size * args.batches_per_second
    total_records = records_per_sec * args.duration

    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                      COMPREHENSIVE INGESTION BENCHMARK                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

This benchmark will run TWO tests:

1. BULK INGESTION
   • Test size: {args.test_size:,} records
   • Single large insert operation
   • Measures maximum throughput capability

2. HIGH-FREQUENCY INGESTION
   • Batch size: {args.batch_size} records
   • Target frequency: {args.batches_per_second} batch/sec ({records_per_sec} records/sec)
   • Duration: {args.duration} seconds
   • Expected total: {total_records:,} records
   • Measures sustained performance and latency

Configuration:
  • Existing dataset size: {args.dataset_size:,} records

Prerequisites:
  ✓ Main benchmark must have been run first with dataset-size={args.dataset_size:,}
  ✓ All database instances must be running
  ✓ Tables must exist in the databases

""")

    try:
        # Run bulk ingestion benchmark
        print("=" * 80)
        print("PHASE 1: BULK INGESTION BENCHMARK")
        print("=" * 80)
        benchmark.run_full_ingestion_benchmark(
            dataset_size=args.dataset_size,
            test_data_size=args.test_size
        )

        print("\n" + "=" * 80)
        print("PHASE 2: HIGH-FREQUENCY INGESTION BENCHMARK")
        print("=" * 80)
        benchmark.run_high_frequency_benchmark(
            dataset_size=args.dataset_size,
            batch_size=args.batch_size,
            batches_per_second=args.batches_per_second,
            duration_seconds=args.duration
        )

        # Generate comparison visualization
        print("\n" + "=" * 80)
        print("GENERATING COMPARISON VISUALIZATIONS")
        print("=" * 80)
        benchmark.generate_comparison_visualization(args.test_size, args.batch_size, args.batches_per_second, args.duration)

    except KeyboardInterrupt:
        print("\n\n⚠ Benchmark interrupted by user")
    except Exception as e:
        print(f"\n\n✗ Benchmark failed with error: {e}")
        raise

    print("\n✓ Complete ingestion benchmark finished!")


if __name__ == "__main__":
    main()
