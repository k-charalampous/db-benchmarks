"""
Ingestion Benchmark - Measures data loading performance from DataFrames

This benchmark:
1. Works with existing tables from the main benchmark
2. Loads test data from DataFrames to measure ingestion performance
3. For ClickHouse strategies with materialized views, measures MV refresh time
4. Cleans up only the newly added test data, leaving existing data intact
"""

import json
import os
import time
from typing import Dict, List

import clickhouse_connect
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import psycopg
import pymongo
from tabulate import tabulate

from data_generation import DataFileManager
from resource_monitor import ResourceMonitor, monitor_docker_container


class IngestionBenchmark:
    """Benchmark data ingestion performance using existing database structures"""

    def __init__(
        self,
        pg_conn: str,
        pg_conn_18: str,
        mongo_conn: str,
        mongo_db: str,
        pg_docker_container: str = None,
        pg_18_docker_container: str = None,
        clickhouse_docker_container: str = None,
        mongo_docker_container: str = None,
    ):
        """
        Initialize the ingestion benchmark.

        Args:
            pg_conn: PostgreSQL 17 connection string
            pg_conn_18: PostgreSQL 18 connection string
            mongo_conn: MongoDB connection string
            mongo_db: MongoDB database name
            pg_docker_container: Optional Docker container name for PostgreSQL 17
            pg_18_docker_container: Optional Docker container name for PostgreSQL 18
            clickhouse_docker_container: Optional Docker container name for ClickHouse
            mongo_docker_container: Optional Docker container name for MongoDB
        """
        self.pg_conn = pg_conn
        self.pg_conn_18 = pg_conn_18
        self.mongo_conn = mongo_conn
        self.mongo_db = mongo_db
        self.results: List[Dict] = []
        self.high_freq_results: List[
            Dict
        ] = []  # Track high-frequency benchmark results

        # Docker container names for server-side monitoring
        self.pg_docker_container = pg_docker_container
        self.pg_18_docker_container = pg_18_docker_container
        self.clickhouse_docker_container = clickhouse_docker_container
        self.mongo_docker_container = mongo_docker_container

    def _check_table_exists(self, conn, table_name: str) -> bool:
        """Check if a table exists"""
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = %s
                )
            """,
                (table_name,),
            )
            return cur.fetchone()[0]

    def _pg_bulk_insert(self, conn, table_name: str, df: pd.DataFrame):
        """Helper method for PostgreSQL bulk insert using COPY (psycopg3)"""
        from io import StringIO

        buffer = StringIO()
        df.to_csv(buffer, index=False, header=False)
        buffer.seek(0)

        # psycopg3 uses copy() context manager
        with conn.cursor() as cur:
            with cur.copy(
                f"COPY {table_name} ({','.join(df.columns)}) FROM STDIN WITH (FORMAT CSV)"
            ) as copy:
                copy.write(buffer.read())
        conn.commit()

    def run_full_ingestion_benchmark(
        self, dataset_size: int = 10000, test_data_size: int = 1000
    ):
        """
        Run complete ingestion benchmark using existing tables

        Args:
            dataset_size: Size of the existing dataset (for table name resolution)
            test_data_size: Size of test data to insert for benchmarking
        """
        print(f"\n{'=' * 80}")
        print("INGESTION BENCHMARK")
        print(f"Existing dataset: {dataset_size:,} records")
        print(f"Test ingestion size: {test_data_size:,} records")
        print(f"{'=' * 80}\n")

        # Clean up old test data files to ensure fresh generation
        print("Cleaning up old test data files...")
        file_manager = DataFileManager()
        import glob
        import os

        cleanup_patterns = [
            '*_ingestion_test_offset*.jsonl',
            '*_ingestion_test_offset*.csv',
            '*_high_freq_test_offset*.jsonl',
            '*_high_freq_test_offset*.csv'
        ]

        files_removed = 0
        for pattern in cleanup_patterns:
            for file_path in glob.glob(str(file_manager.data_dir / pattern)):
                try:
                    os.remove(file_path)
                    files_removed += 1
                except Exception as e:
                    print(f"  ⚠ Could not remove {file_path}: {e}")

        if files_removed > 0:
            print(f"  ✓ Removed {files_removed} old test data file(s)")
        else:
            print("  ✓ No old test data files to remove")

        # Generate test data for ingestion
        print("Generating test data for ingestion benchmark...")

        # Use timestamp-based starting IDs to ensure absolute uniqueness across runs
        # This guarantees no conflicts even if previous cleanup failed
        import time

        timestamp_id = int(time.time() * 1000)  # Milliseconds since epoch
        starting_id = 10_000_000_000 + timestamp_id
        print(f"Using starting order ID: {starting_id:,} to ensure uniqueness")

        file_paths = file_manager.generate_and_save(
            test_data_size, prefix="ingestion_test", starting_order_id=starting_id
        )

        if not file_paths:
            file_paths = {
                "nested_jsonl": f"benchmark_data/{test_data_size}_ingestion_test_nested.jsonl",
                "csv_flat": f"benchmark_data/{test_data_size}_ingestion_test_flat.csv",
            }

        print("✓ Test data generated\n")

        # Load test data into memory for benchmarking
        print("Loading test data into memory...")
        df_flat = pd.read_csv(file_paths["csv_flat"])

        with open(file_paths["nested_jsonl"], "r") as f:
            nested_data = [json.loads(line.strip()) for line in f]

        print(
            f"✓ Test data loaded: {len(df_flat)} flat rows, {len(nested_data)} nested documents\n"
        )

        # Run ingestion benchmarks for each strategy
        print("Starting ingestion benchmarks...\n")

        # self.benchmark_native_pg(nested_data, dataset_size, test_data_size)
        # self.benchmark_hydra(df_flat, dataset_size, test_data_size)
        # self.benchmark_paradedb(df_flat, dataset_size, test_data_size)
        # self.benchmark_flat_postgresql(
        #     df_flat, dataset_size, test_data_size,
        #     conn=self.pg_conn, strategy_name="PostgreSQL17 Flat"
        # )
        self.benchmark_flat_postgresql(
            df_flat,
            dataset_size,
            test_data_size,
            conn=self.pg_conn_18,
            strategy_name="PostgreSQL18 Flat",
        )
        self.benchmark_flat_postgresql_join(
            df_flat,
            dataset_size,
            test_data_size,
            conn=self.pg_conn_18,
            strategy_name="PostgreSQL18 Flat Join",
            table_prefix=f"norm_{dataset_size}",
        )
        self.benchmark_pg_duckdb(df_flat, dataset_size, test_data_size)
        self.benchmark_clickhouse_flat(df_flat, dataset_size, test_data_size)
        self.benchmark_clickhouse_flat_mv(df_flat, dataset_size, test_data_size)
        self.benchmark_clickhouse_flat_join(df_flat, dataset_size, test_data_size)
        self.benchmark_clickhouse_flat_join_mv(df_flat, dataset_size, test_data_size)
        # self.benchmark_clickhouse_nested(nested_data, dataset_size, test_data_size)
        # self.benchmark_clickhouse_nested_mv(nested_data, dataset_size, test_data_size)
        self.benchmark_mongodb(nested_data, dataset_size, test_data_size)

        # Print results
        self.print_results(test_data_size)

    def run_high_frequency_benchmark(
        self,
        dataset_size: int = 10000,
        batch_size: int = 100,
        batches_per_second: int = 1,
        duration_seconds: int = 60,
    ):
        """
        Run high-frequency insert benchmark simulating streaming workloads

        Args:
            dataset_size: Size of existing dataset (for table name resolution)
            batch_size: Number of records per insert batch (default: 100)
            batches_per_second: Target insert frequency (default: 1 batch/sec)
            duration_seconds: How long to run the benchmark (default: 60 seconds)
        """
        print(f"\n{'=' * 80}")
        print("HIGH-FREQUENCY INGESTION BENCHMARK")
        print(f"Existing dataset: {dataset_size:,} records")
        print(f"Batch size: {batch_size} records")
        print(
            f"Target frequency: {batches_per_second} batch/sec ({batch_size * batches_per_second} records/sec)"
        )
        print(f"Duration: {duration_seconds} seconds")
        print(
            f"Expected total: {batch_size * batches_per_second * duration_seconds:,} records"
        )
        print(f"{'=' * 80}\n")

        # Generate test data pool
        print("Generating test data pool...")
        file_manager = DataFileManager()

        # Use timestamp-based starting IDs to ensure absolute uniqueness across runs
        import time

        timestamp_id = int(time.time() * 1000)  # Milliseconds since epoch
        starting_id = 20_000_000_000 + timestamp_id
        print(f"Using starting order ID: {starting_id:,} to ensure uniqueness")

        # Generate enough data for the entire test
        total_records_needed = batch_size * batches_per_second * duration_seconds
        file_paths = file_manager.generate_and_save(
            total_records_needed, prefix="high_freq_test", starting_order_id=starting_id
        )

        if not file_paths:
            file_paths = {
                "nested_jsonl": f"benchmark_data/{total_records_needed}_high_freq_test_nested.jsonl",
                "csv_flat": f"benchmark_data/{total_records_needed}_high_freq_test_flat.csv",
            }

        print("✓ Test data generated\n")

        # Load test data into memory
        print("Loading test data into memory...")
        df_flat = pd.read_csv(file_paths["csv_flat"])

        with open(file_paths["nested_jsonl"], "r") as f:
            nested_data = [json.loads(line.strip()) for line in f]

        print(
            f"✓ Test data loaded: {len(df_flat)} flat rows, {len(nested_data)} nested documents\n"
        )

        # Run high-frequency benchmarks
        print("Starting high-frequency ingestion benchmarks...\n")

        # PostgreSQL strategies
        # self._high_freq_benchmark_native_pg(
        #     nested_data, dataset_size, batch_size, batches_per_second, duration_seconds
        # )
        self._high_freq_benchmark_flat_postgresql(
            df_flat, dataset_size, batch_size, batches_per_second, duration_seconds
        )

        # ClickHouse strategies
        self._high_freq_benchmark_clickhouse_flat(
            df_flat, dataset_size, batch_size, batches_per_second, duration_seconds
        )
        self._high_freq_benchmark_clickhouse_flat_mv(
            df_flat, dataset_size, batch_size, batches_per_second, duration_seconds
        )
        self._high_freq_benchmark_clickhouse_flat_join(
            df_flat, dataset_size, batch_size, batches_per_second, duration_seconds
        )
        self._high_freq_benchmark_clickhouse_flat_join_mv(
            df_flat, dataset_size, batch_size, batches_per_second, duration_seconds
        )
        # self._high_freq_benchmark_clickhouse_nested(
        #     nested_data, dataset_size, batch_size, batches_per_second, duration_seconds
        # )
        # self._high_freq_benchmark_clickhouse_nested_mv(
        #     nested_data, dataset_size, batch_size, batches_per_second, duration_seconds
        # )

        # MongoDB
        self._high_freq_benchmark_mongodb(
            nested_data, dataset_size, batch_size, batches_per_second, duration_seconds
        )

        # Print results
        self.print_high_freq_results(batch_size, batches_per_second, duration_seconds)

    def benchmark_native_pg(
        self, nested_data: List[Dict], dataset_size: int, test_data_size: int
    ):
        """Benchmark ingestion for JSONB PostgreSQL"""
        print("--- Benchmarking: PostgreSQL JSONB ---")

        table_name = f"nested_native_pg_{dataset_size}"

        try:
            conn = psycopg.connect(self.pg_conn)

            # Get initial row count
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {table_name}")
                initial_count = cur.fetchone()[0]

            # Insert test data and measure time with resource monitoring
            monitor = ResourceMonitor(interval=0.1)
            monitor.start()
            start_time = time.perf_counter()

            with conn.cursor() as cur:
                for record in nested_data:
                    cur.execute(
                        f"INSERT INTO {table_name} (data) VALUES (%s)",
                        (json.dumps(record),),
                    )
            conn.commit()

            ingestion_time = time.perf_counter() - start_time
            resources = monitor.stop()

            # Verify insertion
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {table_name}")
                final_count = cur.fetchone()[0]

            rows_inserted = final_count - initial_count

            print(
                f"  ✓ Ingested {rows_inserted:,} rows in {ingestion_time:.2f}s "
                f"({rows_inserted / ingestion_time:,.0f} records/sec)"
            )
            print(
                f"    CPU: {resources.cpu_percent_avg:.1f}% avg, {resources.cpu_percent_max:.1f}% max | "
                f"Memory: {resources.memory_mb_avg:.1f} MB avg, {resources.memory_mb_max:.1f} MB max"
            )

            self.results.append(
                {
                    "strategy": "PostgreSQL JSONB",
                    "ingestion_time_seconds": ingestion_time,
                    "throughput_records_per_sec": rows_inserted / ingestion_time,
                    "mv_refresh_time_seconds": None,
                    "total_time_seconds": ingestion_time,
                    "rows_inserted": rows_inserted,
                    "cpu_percent_avg": resources.cpu_percent_avg,
                    "cpu_percent_max": resources.cpu_percent_max,
                    "memory_mb_avg": resources.memory_mb_avg,
                    "memory_mb_max": resources.memory_mb_max,
                }
            )

            # Cleanup: delete the test data
            with conn.cursor() as cur:
                # Delete the last N inserted rows (ordered by insertion)
                cur.execute(
                    f"""
                    DELETE FROM {table_name}
                    WHERE ctid IN (
                        SELECT ctid FROM {table_name}
                        ORDER BY ctid DESC
                        LIMIT %s
                    )
                """,
                    (rows_inserted,),
                )
                conn.commit()

            print(f"  ✓ Cleaned up {rows_inserted:,} test rows\n")

            conn.close()

        except Exception as e:
            print(f"  ✗ Error: {e}\n")

    def benchmark_hydra(self, df: pd.DataFrame, dataset_size: int, test_data_size: int):
        """Benchmark ingestion for PostgreSQL with Hydra"""
        print("--- Benchmarking: PostgreSQL Hydra Flat ---")

        table_name = f"flat_pg_{dataset_size}"
        conn_string = "host=localhost port=5434 dbname=benchmark_db user=benchmark_user password=benchmark_pass"

        try:
            conn = psycopg.connect(conn_string)

            # Check if table exists
            if not self._check_table_exists(conn, table_name):
                print(f"  ⚠ Table {table_name} does not exist, skipping\n")
                conn.close()
                return

            # Get initial row count
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {table_name}")
                initial_count = cur.fetchone()[0]

            # Insert test data with resource monitoring
            monitor = ResourceMonitor(interval=0.1)
            monitor.start()
            start_time = time.perf_counter()
            self._pg_bulk_insert(conn, table_name, df)
            ingestion_time = time.perf_counter() - start_time
            resources = monitor.stop()

            # Verify insertion
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {table_name}")
                final_count = cur.fetchone()[0]

            rows_inserted = final_count - initial_count

            print(
                f"  ✓ Ingested {rows_inserted:,} rows in {ingestion_time:.2f}s "
                f"({rows_inserted / ingestion_time:,.0f} records/sec)"
            )
            print(
                f"    CPU: {resources.cpu_percent_avg:.1f}% avg, {resources.cpu_percent_max:.1f}% max | "
                f"Memory: {resources.memory_mb_avg:.1f} MB avg, {resources.memory_mb_max:.1f} MB max"
            )

            self.results.append(
                {
                    "strategy": "PostgreSQL Hydra Flat",
                    "ingestion_time_seconds": ingestion_time,
                    "throughput_records_per_sec": rows_inserted / ingestion_time,
                    "mv_refresh_time_seconds": None,
                    "total_time_seconds": ingestion_time,
                    "rows_inserted": rows_inserted,
                    "cpu_percent_avg": resources.cpu_percent_avg,
                    "cpu_percent_max": resources.cpu_percent_max,
                    "memory_mb_avg": resources.memory_mb_avg,
                    "memory_mb_max": resources.memory_mb_max,
                }
            )

            # Cleanup
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    DELETE FROM {table_name}
                    WHERE ctid IN (
                        SELECT ctid FROM {table_name}
                        ORDER BY ctid DESC
                        LIMIT %s
                    )
                """,
                    (rows_inserted,),
                )
                conn.commit()

            print(f"  ✓ Cleaned up {rows_inserted:,} test rows\n")

            conn.close()

        except Exception as e:
            print(f"  ✗ Error: {e}\n")

    def benchmark_paradedb(
        self, df: pd.DataFrame, dataset_size: int, test_data_size: int
    ):
        """Benchmark ingestion for PostgreSQL with ParadeDB"""
        print("--- Benchmarking: PostgreSQL ParadeDB Flat ---")

        table_name = f"flat_pg_{dataset_size}"
        conn_string = "host=localhost port=5435 dbname=benchmark_db user=benchmark_user password=benchmark_pass"

        try:
            conn = psycopg.connect(conn_string)

            # Check if table exists
            if not self._check_table_exists(conn, table_name):
                print(f"  ⚠ Table {table_name} does not exist, skipping\n")
                conn.close()
                return

            # Get initial row count
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {table_name}")
                initial_count = cur.fetchone()[0]

            # Insert test data with resource monitoring
            monitor = ResourceMonitor(interval=0.1)
            monitor.start()
            start_time = time.perf_counter()
            self._pg_bulk_insert(conn, table_name, df)
            ingestion_time = time.perf_counter() - start_time
            resources = monitor.stop()

            # Verify insertion
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {table_name}")
                final_count = cur.fetchone()[0]

            rows_inserted = final_count - initial_count

            print(
                f"  ✓ Ingested {rows_inserted:,} rows in {ingestion_time:.2f}s "
                f"({rows_inserted / ingestion_time:,.0f} records/sec)"
            )
            print(
                f"    CPU: {resources.cpu_percent_avg:.1f}% avg, {resources.cpu_percent_max:.1f}% max | "
                f"Memory: {resources.memory_mb_avg:.1f} MB avg, {resources.memory_mb_max:.1f} MB max"
            )

            self.results.append(
                {
                    "strategy": "PostgreSQL ParadeDB Flat",
                    "ingestion_time_seconds": ingestion_time,
                    "throughput_records_per_sec": rows_inserted / ingestion_time,
                    "mv_refresh_time_seconds": None,
                    "total_time_seconds": ingestion_time,
                    "rows_inserted": rows_inserted,
                    "cpu_percent_avg": resources.cpu_percent_avg,
                    "cpu_percent_max": resources.cpu_percent_max,
                    "memory_mb_avg": resources.memory_mb_avg,
                    "memory_mb_max": resources.memory_mb_max,
                }
            )

            # Cleanup
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    DELETE FROM {table_name}
                    WHERE ctid IN (
                        SELECT ctid FROM {table_name}
                        ORDER BY ctid DESC
                        LIMIT %s
                    )
                """,
                    (rows_inserted,),
                )
                conn.commit()

            print(f"  ✓ Cleaned up {rows_inserted:,} test rows\n")

            conn.close()

        except Exception as e:
            print(f"  ✗ Error: {e}\n")

    def benchmark_flat_postgresql(
        self,
        df: pd.DataFrame,
        dataset_size: int,
        test_data_size: int,
        conn: str,
        strategy_name: str,
    ):
        """Benchmark ingestion for flat PostgreSQL"""
        print(f"--- Benchmarking: {strategy_name} ---")

        table_name = f"flat_pg_{dataset_size}"

        try:
            connection = psycopg.connect(conn)

            # Check if table exists
            if not self._check_table_exists(connection, table_name):
                print(f"  ⚠ Table {table_name} does not exist, skipping\n")
                connection.close()
                return

            # Clean up any leftover test data from failed previous runs
            # Test data has order IDs starting with "ORD-10" (10 billion+)
            with connection.cursor() as cur:
                cur.execute(f"DELETE FROM {table_name} WHERE order_id LIKE 'ORD-10%'")
                deleted_count = cur.rowcount
                connection.commit()
                if deleted_count > 0:
                    print(f"  ✓ Cleaned up {deleted_count:,} leftover test rows from previous runs")

            # Get initial row count
            with connection.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {table_name}")
                initial_count = cur.fetchone()[0]

            # Insert test data with resource monitoring
            monitor = ResourceMonitor(interval=0.1)
            monitor.start()
            start_time = time.perf_counter()
            self._pg_bulk_insert(connection, table_name, df)
            ingestion_time = time.perf_counter() - start_time
            resources = monitor.stop()

            # Verify insertion
            with connection.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {table_name}")
                final_count = cur.fetchone()[0]

            rows_inserted = final_count - initial_count

            print(
                f"  ✓ Ingested {rows_inserted:,} rows in {ingestion_time:.2f}s "
                f"({rows_inserted / ingestion_time:,.0f} records/sec)"
            )
            print(
                f"    CPU: {resources.cpu_percent_avg:.1f}% avg, {resources.cpu_percent_max:.1f}% max | "
                f"Memory: {resources.memory_mb_avg:.1f} MB avg, {resources.memory_mb_max:.1f} MB max"
            )

            self.results.append(
                {
                    "strategy": strategy_name,
                    "ingestion_time_seconds": ingestion_time,
                    "throughput_records_per_sec": rows_inserted / ingestion_time,
                    "mv_refresh_time_seconds": None,
                    "total_time_seconds": ingestion_time,
                    "rows_inserted": rows_inserted,
                    "cpu_percent_avg": resources.cpu_percent_avg,
                    "cpu_percent_max": resources.cpu_percent_max,
                    "memory_mb_avg": resources.memory_mb_avg,
                    "memory_mb_max": resources.memory_mb_max,
                }
            )

            # Cleanup
            with connection.cursor() as cur:
                cur.execute(
                    f"""
                    DELETE FROM {table_name}
                    WHERE ctid IN (
                        SELECT ctid FROM {table_name}
                        ORDER BY ctid DESC
                        LIMIT %s
                    )
                """,
                    (rows_inserted,),
                )
                connection.commit()

            print(f"  ✓ Cleaned up {rows_inserted:,} test rows\n")

            connection.close()

        except Exception as e:
            print(f"  ✗ Error: {e}\n")

    def benchmark_flat_postgresql_join(
        self,
        df: pd.DataFrame,
        dataset_size: int,
        test_data_size: int,
        conn: str,
        strategy_name: str,
        table_prefix: str,
    ):
        """Benchmark ingestion for normalized PostgreSQL with joins"""
        print(f"--- Benchmarking: {strategy_name} ---")

        try:
            connection = psycopg.connect(conn)

            # Check if the order_items table exists (main table for flat data)
            items_table = f"{table_prefix}_order_items"
            if not self._check_table_exists(connection, items_table):
                print(f"  ⚠ Table {items_table} does not exist, skipping\n")
                connection.close()
                return

            # Clean up any leftover test data from failed previous runs
            # Test data has order IDs starting with "ORD-10" (10 billion+)
            with connection.cursor() as cur:
                cur.execute(f"DELETE FROM {items_table} WHERE order_id LIKE 'ORD-10%'")
                deleted_count = cur.rowcount
                connection.commit()
                if deleted_count > 0:
                    print(f"  ✓ Cleaned up {deleted_count:,} leftover test rows from previous runs")

            # Get initial count
            with connection.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {items_table}")
                initial_count = cur.fetchone()[0]

            # Filter DataFrame to only include columns that exist in order_items table
            # order_items schema: order_id, product_id, quantity, discount_applied, discount_percentage
            order_items_columns = [
                "order_id",
                "product_id",
                "quantity",
                "discount_applied",
                "discount_percentage",
            ]
            df_filtered = df[order_items_columns].copy()

            # Insert test data with resource monitoring (simplified - real normalized ingestion would be more complex)
            monitor = ResourceMonitor(interval=0.1)
            monitor.start()
            start_time = time.perf_counter()
            self._pg_bulk_insert(connection, items_table, df_filtered)
            ingestion_time = time.perf_counter() - start_time
            resources = monitor.stop()

            # Get final count
            with connection.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {items_table}")
                final_count = cur.fetchone()[0]

            rows_inserted = final_count - initial_count

            print(
                f"  ✓ Ingested {rows_inserted:,} rows in {ingestion_time:.2f}s "
                f"({rows_inserted / ingestion_time:,.0f} records/sec)"
            )
            print(
                f"    CPU: {resources.cpu_percent_avg:.1f}% avg, {resources.cpu_percent_max:.1f}% max | "
                f"Memory: {resources.memory_mb_avg:.1f} MB avg, {resources.memory_mb_max:.1f} MB max"
            )

            self.results.append(
                {
                    "strategy": strategy_name,
                    "ingestion_time_seconds": ingestion_time,
                    "throughput_records_per_sec": rows_inserted / ingestion_time,
                    "mv_refresh_time_seconds": None,
                    "total_time_seconds": ingestion_time,
                    "rows_inserted": rows_inserted,
                    "cpu_percent_avg": resources.cpu_percent_avg,
                    "cpu_percent_max": resources.cpu_percent_max,
                    "memory_mb_avg": resources.memory_mb_avg,
                    "memory_mb_max": resources.memory_mb_max,
                }
            )

            # Cleanup - delete test rows
            with connection.cursor() as cur:
                cur.execute(
                    f"""
                    DELETE FROM {items_table}
                    WHERE ctid IN (
                        SELECT ctid FROM {items_table}
                        ORDER BY ctid DESC
                        LIMIT %s
                    )
                """,
                    (rows_inserted,),
                )
                connection.commit()

            print(f"  ✓ Cleaned up {rows_inserted:,} test rows\n")

            connection.close()

        except Exception as e:
            print(f"  ✗ Error: {e}\n")

    def benchmark_pg_duckdb(
        self, df: pd.DataFrame, dataset_size: int, test_data_size: int
    ):
        """Benchmark ingestion for PostgreSQL with pg_duckdb"""
        print("--- Benchmarking: PostgreSQL pg_duckdb Flat ---")

        # pg_duckdb uses the same flat table
        table_name = f"flat_pg_{dataset_size}"

        try:
            connection = psycopg.connect(self.pg_conn)

            # Check if table exists
            if not self._check_table_exists(connection, table_name):
                print(f"  ⚠ Table {table_name} does not exist, skipping\n")
                connection.close()
                return

            # Clean up any leftover test data from failed previous runs
            # Test data has order IDs starting with "ORD-10" (10 billion+)
            with connection.cursor() as cur:
                cur.execute(f"DELETE FROM {table_name} WHERE order_id LIKE 'ORD-10%'")
                deleted_count = cur.rowcount
                connection.commit()
                if deleted_count > 0:
                    print(f"  ✓ Cleaned up {deleted_count:,} leftover test rows from previous runs")

            # Get initial row count
            with connection.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {table_name}")
                initial_count = cur.fetchone()[0]

            # Insert test data with resource monitoring
            monitor = ResourceMonitor(interval=0.1)
            monitor.start()
            start_time = time.perf_counter()
            self._pg_bulk_insert(connection, table_name, df)
            ingestion_time = time.perf_counter() - start_time
            resources = monitor.stop()

            # Verify insertion
            with connection.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {table_name}")
                final_count = cur.fetchone()[0]

            rows_inserted = final_count - initial_count

            print(
                f"  ✓ Ingested {rows_inserted:,} rows in {ingestion_time:.2f}s "
                f"({rows_inserted / ingestion_time:,.0f} records/sec)"
            )
            print(
                f"    CPU: {resources.cpu_percent_avg:.1f}% avg, {resources.cpu_percent_max:.1f}% max | "
                f"Memory: {resources.memory_mb_avg:.1f} MB avg, {resources.memory_mb_max:.1f} MB max"
            )

            self.results.append(
                {
                    "strategy": "PostgreSQL pg_duckdb Flat",
                    "ingestion_time_seconds": ingestion_time,
                    "throughput_records_per_sec": rows_inserted / ingestion_time,
                    "mv_refresh_time_seconds": None,
                    "total_time_seconds": ingestion_time,
                    "rows_inserted": rows_inserted,
                    "cpu_percent_avg": resources.cpu_percent_avg,
                    "cpu_percent_max": resources.cpu_percent_max,
                    "memory_mb_avg": resources.memory_mb_avg,
                    "memory_mb_max": resources.memory_mb_max,
                }
            )

            # Cleanup
            with connection.cursor() as cur:
                cur.execute(
                    f"""
                    DELETE FROM {table_name}
                    WHERE ctid IN (
                        SELECT ctid FROM {table_name}
                        ORDER BY ctid DESC
                        LIMIT %s
                    )
                """,
                    (rows_inserted,),
                )
                connection.commit()

            print(f"  ✓ Cleaned up {rows_inserted:,} test rows\n")

            connection.close()

        except Exception as e:
            print(f"  ✗ Error: {e}\n")

    def benchmark_clickhouse_flat(
        self, df: pd.DataFrame, dataset_size: int, test_data_size: int
    ):
        """Benchmark ingestion for ClickHouse flat"""
        print("--- Benchmarking: ClickHouse Flat ---")

        table_name = f"flat_clickhouse_{dataset_size}"

        try:
            client = clickhouse_connect.get_client(
                host="localhost",
                port=8123,
                username="benchmark_user",
                password="benchmark_pass",
                database="benchmark_db",
            )

            # Clean up any leftover test data from failed previous runs
            # Test data has order IDs starting with "ORD-10" (10 billion+)
            client.command(f"ALTER TABLE {table_name} DELETE WHERE order_id LIKE 'ORD-10%'")
            # Wait for mutation to complete
            import time as time_module
            time_module.sleep(1)
            print("  ✓ Cleaned up leftover test rows from previous runs")

            # Get initial row count
            result = client.query(f"SELECT count() FROM {table_name}")
            initial_count = result.result_rows[0][0]

            # Prepare data - handle NaN values
            df_clean = df.copy()
            string_columns = df_clean.select_dtypes(include=["object"]).columns
            for col in string_columns:
                df_clean[col] = df_clean[col].fillna("")

            # Insert test data with resource monitoring
            monitor = ResourceMonitor(interval=0.1)
            monitor.start()
            start_time = time.perf_counter()
            client.insert_df(table_name, df_clean)
            ingestion_time = time.perf_counter() - start_time
            resources = monitor.stop()

            # Verify insertion
            result = client.query(f"SELECT count() FROM {table_name}")
            final_count = result.result_rows[0][0]

            rows_inserted = final_count - initial_count

            print(
                f"  ✓ Ingested {rows_inserted:,} rows in {ingestion_time:.2f}s "
                f"({rows_inserted / ingestion_time:,.0f} records/sec)"
            )
            print(
                f"    CPU: {resources.cpu_percent_avg:.1f}% avg, {resources.cpu_percent_max:.1f}% max | "
                f"Memory: {resources.memory_mb_avg:.1f} MB avg, {resources.memory_mb_max:.1f} MB max"
            )

            self.results.append(
                {
                    "strategy": "ClickHouse Flat",
                    "ingestion_time_seconds": ingestion_time,
                    "throughput_records_per_sec": rows_inserted / ingestion_time,
                    "mv_refresh_time_seconds": None,
                    "total_time_seconds": ingestion_time,
                    "rows_inserted": rows_inserted,
                    "cpu_percent_avg": resources.cpu_percent_avg,
                    "cpu_percent_max": resources.cpu_percent_max,
                    "memory_mb_avg": resources.memory_mb_avg,
                    "memory_mb_max": resources.memory_mb_max,
                }
            )

            # Cleanup - delete last N rows (ClickHouse doesn't have easy row-by-row delete)
            # We can use ALTER TABLE DELETE with a condition
            # Get max order_id before cleanup
            result = client.query(f"SELECT max(order_id) FROM {table_name}")
            max_order_id = result.result_rows[0][0]

            # Delete rows we just inserted (they'll have the highest order_ids)
            # This is an approximation
            client.command(f"""
                ALTER TABLE {table_name}
                DELETE WHERE order_id IN (
                    SELECT order_id FROM {table_name}
                    ORDER BY order_id DESC
                    LIMIT {rows_inserted}
                )
            """)

            print(f"  ✓ Cleaned up {rows_inserted:,} test rows\n")

            client.close()

        except Exception as e:
            print(f"  ✗ Error: {e}\n")

    def benchmark_clickhouse_flat_mv(
        self, df: pd.DataFrame, dataset_size: int, test_data_size: int
    ):
        """Benchmark ingestion for ClickHouse flat with materialized views - measures MV refresh"""
        print("--- Benchmarking: ClickHouse Flat + Materialized Views ---")

        table_name = f"flat_clickhouse_mv_{dataset_size}"

        try:
            client = clickhouse_connect.get_client(
                host="localhost",
                port=8123,
                username="benchmark_user",
                password="benchmark_pass",
                database="benchmark_db",
            )

            # Get initial row count
            result = client.query(f"SELECT count() FROM {table_name}")
            initial_count = result.result_rows[0][0]

            # Prepare data
            df_clean = df.copy()
            string_columns = df_clean.select_dtypes(include=["object"]).columns
            for col in string_columns:
                df_clean[col] = df_clean[col].fillna("")

            # Insert test data and measure both ingestion and MV update with resource monitoring
            ingestion_monitor = ResourceMonitor(interval=0.1)
            ingestion_monitor.start()
            start_time = time.perf_counter()
            client.insert_df(table_name, df_clean)
            ingestion_time = time.perf_counter() - start_time
            ingestion_resources = ingestion_monitor.stop()

            # Wait for MVs to update and measure the time
            # ClickHouse MVs update automatically, but we need to ensure they're done
            # Use Docker monitoring if available (monitors server), otherwise client monitoring
            if self.clickhouse_docker_container:
                mv_refresh_monitor = monitor_docker_container(
                    self.clickhouse_docker_container, interval=0.5
                )
            else:
                mv_refresh_monitor = ResourceMonitor(interval=0.1)

            mv_refresh_monitor.start()
            mv_refresh_start = time.perf_counter()

            # Force optimization to ensure MVs are updated
            # We need to optimize both the source table and MVs to ensure all parts are merged
            for mv_suffix in [
                "simple_agg",
                "deep_agg",
                "array_agg",
                "complex_agg",
                "seller_agg",
            ]:
                mv_name = f"{table_name}_mv_{mv_suffix}"
                # OPTIMIZE FINAL forces all parts to be merged and MVs to be fully updated
                client.query(f"OPTIMIZE TABLE {mv_name} FINAL")
                # Verify with a count query to ensure data is visible
                client.query(f"SELECT count() FROM {mv_name}")

            mv_refresh_time = time.perf_counter() - mv_refresh_start
            mv_refresh_resources = mv_refresh_monitor.stop()
            total_time = ingestion_time + mv_refresh_time

            # Verify insertion
            result = client.query(f"SELECT count() FROM {table_name}")
            final_count = result.result_rows[0][0]

            rows_inserted = final_count - initial_count

            print(f"  ✓ Ingested {rows_inserted:,} rows in {ingestion_time:.2f}s")
            print(
                f"    CPU: {ingestion_resources.cpu_percent_avg:.1f}% avg, {ingestion_resources.cpu_percent_max:.1f}% max | "
                f"Memory: {ingestion_resources.memory_mb_avg:.1f} MB avg, {ingestion_resources.memory_mb_max:.1f} MB max"
            )
            print(f"  ✓ MV refresh time: {mv_refresh_time:.2f}s")
            print(
                f"    CPU: {mv_refresh_resources.cpu_percent_avg:.1f}% avg, {mv_refresh_resources.cpu_percent_max:.1f}% max | "
                f"Memory: {mv_refresh_resources.memory_mb_avg:.1f} MB avg, {mv_refresh_resources.memory_mb_max:.1f} MB max"
            )
            print(
                f"  ✓ Total time: {total_time:.2f}s ({rows_inserted / total_time:,.0f} records/sec)"
            )

            self.results.append(
                {
                    "strategy": "ClickHouse Flat+MV",
                    "ingestion_time_seconds": ingestion_time,
                    "throughput_records_per_sec": rows_inserted / ingestion_time,
                    "mv_refresh_time_seconds": mv_refresh_time,
                    "total_time_seconds": total_time,
                    "rows_inserted": rows_inserted,
                    "ingestion_cpu_percent_avg": ingestion_resources.cpu_percent_avg,
                    "ingestion_cpu_percent_max": ingestion_resources.cpu_percent_max,
                    "ingestion_memory_mb_avg": ingestion_resources.memory_mb_avg,
                    "ingestion_memory_mb_max": ingestion_resources.memory_mb_max,
                    "mv_refresh_cpu_percent_avg": mv_refresh_resources.cpu_percent_avg,
                    "mv_refresh_cpu_percent_max": mv_refresh_resources.cpu_percent_max,
                    "mv_refresh_memory_mb_avg": mv_refresh_resources.memory_mb_avg,
                    "mv_refresh_memory_mb_max": mv_refresh_resources.memory_mb_max,
                }
            )

            # Cleanup
            client.command(f"""
                ALTER TABLE {table_name}
                DELETE WHERE order_id IN (
                    SELECT order_id FROM {table_name}
                    ORDER BY order_id DESC
                    LIMIT {rows_inserted}
                )
            """)

            print(f"  ✓ Cleaned up {rows_inserted:,} test rows\n")

            client.close()

        except Exception as e:
            print(f"  ✗ Error: {e}\n")

    def benchmark_clickhouse_flat_join(
        self, df: pd.DataFrame, dataset_size: int, test_data_size: int
    ):
        """Benchmark ingestion for ClickHouse normalized"""
        print("--- Benchmarking: ClickHouse Flat Join ---")

        table_prefix = f"ch_norm_{dataset_size}"
        table_name = f"{table_prefix}_order_items"

        try:
            client = clickhouse_connect.get_client(
                host="localhost",
                port=8123,
                username="benchmark_user",
                password="benchmark_pass",
                database="benchmark_db",
            )

            # Check if table exists
            result = client.query(f"""
                SELECT count() FROM system.tables
                WHERE database = 'benchmark_db' AND name = '{table_name}'
            """)
            if result.result_rows[0][0] == 0:
                print(f"  ⚠ Table {table_name} does not exist, skipping\n")
                client.close()
                return

            # Clean up any leftover test data from failed previous runs
            # Test data has order IDs starting with "ORD-10" (10 billion+)
            client.command(f"ALTER TABLE {table_name} DELETE WHERE order_id LIKE 'ORD-10%'")
            # Wait for mutation to complete
            import time as time_module
            time_module.sleep(1)
            print("  ✓ Cleaned up leftover test rows from previous runs")

            # Get table columns to match DataFrame
            result = client.query(f"DESCRIBE TABLE {table_name}")
            table_columns = [row[0] for row in result.result_rows]

            # Filter DataFrame to only columns that exist in the table
            matching_cols = [col for col in df.columns if col in table_columns]
            if not matching_cols:
                print("  ⚠ No matching columns between DataFrame and table, skipping\n")
                client.close()
                return

            df_subset = df[matching_cols].copy()

            # Get initial row count
            result = client.query(f"SELECT count() FROM {table_name}")
            initial_count = result.result_rows[0][0]

            # Prepare data
            string_columns = df_subset.select_dtypes(include=["object"]).columns
            for col in string_columns:
                df_subset[col] = df_subset[col].fillna("")

            # Insert test data with resource monitoring
            monitor = ResourceMonitor(interval=0.1)
            monitor.start()
            start_time = time.perf_counter()
            client.insert_df(table_name, df_subset)
            ingestion_time = time.perf_counter() - start_time
            resources = monitor.stop()

            # Verify insertion
            result = client.query(f"SELECT count() FROM {table_name}")
            final_count = result.result_rows[0][0]

            rows_inserted = final_count - initial_count

            print(
                f"  ✓ Ingested {rows_inserted:,} rows in {ingestion_time:.2f}s "
                f"({rows_inserted / ingestion_time:,.0f} records/sec)"
            )
            print(
                f"    CPU: {resources.cpu_percent_avg:.1f}% avg, {resources.cpu_percent_max:.1f}% max | "
                f"Memory: {resources.memory_mb_avg:.1f} MB avg, {resources.memory_mb_max:.1f} MB max"
            )

            self.results.append(
                {
                    "strategy": "ClickHouse Flat Join",
                    "ingestion_time_seconds": ingestion_time,
                    "throughput_records_per_sec": rows_inserted / ingestion_time,
                    "mv_refresh_time_seconds": None,
                    "total_time_seconds": ingestion_time,
                    "rows_inserted": rows_inserted,
                    "cpu_percent_avg": resources.cpu_percent_avg,
                    "cpu_percent_max": resources.cpu_percent_max,
                    "memory_mb_avg": resources.memory_mb_avg,
                    "memory_mb_max": resources.memory_mb_max,
                }
            )

            # Cleanup
            client.command(f"""
                ALTER TABLE {table_name}
                DELETE WHERE order_id IN (
                    SELECT order_id FROM {table_name}
                    ORDER BY order_id DESC
                    LIMIT {rows_inserted}
                )
            """)

            print(f"  ✓ Cleaned up {rows_inserted:,} test rows\n")

            client.close()

        except Exception as e:
            print(f"  ✗ Error: {e}\n")

    def benchmark_clickhouse_flat_join_mv(
        self, df: pd.DataFrame, dataset_size: int, test_data_size: int
    ):
        """Benchmark ingestion for ClickHouse normalized with MVs"""
        print("--- Benchmarking: ClickHouse Flat Join + Materialized Views ---")

        table_prefix = f"ch_norm_mv_{dataset_size}"
        table_name = f"{table_prefix}_order_items"

        try:
            client = clickhouse_connect.get_client(
                host="localhost",
                port=8123,
                username="benchmark_user",
                password="benchmark_pass",
                database="benchmark_db",
            )

            # Check if table exists
            result = client.query(f"""
                SELECT count() FROM system.tables
                WHERE database = 'benchmark_db' AND name = '{table_name}'
            """)
            if result.result_rows[0][0] == 0:
                print(f"  ⚠ Table {table_name} does not exist, skipping\n")
                client.close()
                return

            # Get table columns to match DataFrame
            result = client.query(f"DESCRIBE TABLE {table_name}")
            table_columns = [row[0] for row in result.result_rows]

            # Filter DataFrame to only columns that exist in the table
            matching_cols = [col for col in df.columns if col in table_columns]
            if not matching_cols:
                print("  ⚠ No matching columns between DataFrame and table, skipping\n")
                client.close()
                return

            df_subset = df[matching_cols].copy()

            # Get initial row count
            result = client.query(f"SELECT count() FROM {table_name}")
            initial_count = result.result_rows[0][0]

            # Prepare data
            string_columns = df_subset.select_dtypes(include=["object"]).columns
            for col in string_columns:
                df_subset[col] = df_subset[col].fillna("")

            # Insert test data with resource monitoring
            ingestion_monitor = ResourceMonitor(interval=0.1)
            ingestion_monitor.start()
            start_time = time.perf_counter()
            client.insert_df(table_name, df_subset)
            ingestion_time = time.perf_counter() - start_time
            ingestion_resources = ingestion_monitor.stop()

            # Measure MV refresh with resource monitoring
            # Use Docker monitoring if available (monitors server), otherwise client monitoring
            if self.clickhouse_docker_container:
                mv_refresh_monitor = monitor_docker_container(
                    self.clickhouse_docker_container, interval=0.5
                )
            else:
                mv_refresh_monitor = ResourceMonitor(interval=0.1)
            mv_refresh_monitor.start()
            mv_refresh_start = time.perf_counter()
            for mv_suffix in [
                "simple_agg",
                "deep_agg",
                "array_agg",
                "complex_agg",
                "seller_agg",
            ]:
                mv_name = f"{table_prefix}_mv_{mv_suffix}"
                try:
                    # OPTIMIZE FINAL forces all parts to be merged and MVs to be fully updated
                    client.query(f"OPTIMIZE TABLE {mv_name} FINAL")
                    # Verify with a count query to ensure data is visible
                    client.query(f"SELECT count() FROM {mv_name}")
                except Exception:
                    pass  # MV might not exist
            mv_refresh_time = time.perf_counter() - mv_refresh_start
            mv_refresh_resources = mv_refresh_monitor.stop()
            total_time = ingestion_time + mv_refresh_time

            # Verify insertion
            result = client.query(f"SELECT count() FROM {table_name}")
            final_count = result.result_rows[0][0]

            rows_inserted = final_count - initial_count

            print(f"  ✓ Ingested {rows_inserted:,} rows in {ingestion_time:.2f}s")
            print(
                f"    CPU: {ingestion_resources.cpu_percent_avg:.1f}% avg, {ingestion_resources.cpu_percent_max:.1f}% max | "
                f"Memory: {ingestion_resources.memory_mb_avg:.1f} MB avg, {ingestion_resources.memory_mb_max:.1f} MB max"
            )
            print(f"  ✓ MV refresh time: {mv_refresh_time:.2f}s")
            print(
                f"    CPU: {mv_refresh_resources.cpu_percent_avg:.1f}% avg, {mv_refresh_resources.cpu_percent_max:.1f}% max | "
                f"Memory: {mv_refresh_resources.memory_mb_avg:.1f} MB avg, {mv_refresh_resources.memory_mb_max:.1f} MB max"
            )
            print(f"  ✓ Total time: {total_time:.2f}s")

            self.results.append(
                {
                    "strategy": "ClickHouse Join+MV",
                    "ingestion_time_seconds": ingestion_time,
                    "throughput_records_per_sec": rows_inserted / ingestion_time,
                    "mv_refresh_time_seconds": mv_refresh_time,
                    "total_time_seconds": total_time,
                    "rows_inserted": rows_inserted,
                    "ingestion_cpu_percent_avg": ingestion_resources.cpu_percent_avg,
                    "ingestion_cpu_percent_max": ingestion_resources.cpu_percent_max,
                    "ingestion_memory_mb_avg": ingestion_resources.memory_mb_avg,
                    "ingestion_memory_mb_max": ingestion_resources.memory_mb_max,
                    "mv_refresh_cpu_percent_avg": mv_refresh_resources.cpu_percent_avg,
                    "mv_refresh_cpu_percent_max": mv_refresh_resources.cpu_percent_max,
                    "mv_refresh_memory_mb_avg": mv_refresh_resources.memory_mb_avg,
                    "mv_refresh_memory_mb_max": mv_refresh_resources.memory_mb_max,
                }
            )

            # Cleanup
            client.command(f"""
                ALTER TABLE {table_name}
                DELETE WHERE order_id IN (
                    SELECT order_id FROM {table_name}
                    ORDER BY order_id DESC
                    LIMIT {rows_inserted}
                )
            """)

            print(f"  ✓ Cleaned up {rows_inserted:,} test rows\n")

            client.close()

        except Exception as e:
            print(f"  ✗ Error: {e}\n")

    def _transform_nested_record(self, record: Dict) -> Dict:
        """Transform JSON record to ClickHouse nested tuple format"""
        return {
            "order_id": record["order_id"],
            "timestamp": record["timestamp"],
            "customer": (
                record["customer"]["id"],
                record["customer"]["name"],
                record["customer"]["email"],
                record["customer"]["tier"],
                record["customer"]["lifetime_value"],
            ),
            "payment": (
                record["payment"]["method"],
                record["payment"]["status"],
                record["payment"]["amount"],
                (
                    record["payment"]["processor"]["name"],
                    record["payment"]["processor"]["fee"],
                ),
            ),
            "shipping": (
                record["shipping"]["status"],
                record["shipping"]["method"],
                record["shipping"]["cost"],
                (
                    record["shipping"]["address"]["city"],
                    record["shipping"]["address"]["state"],
                    record["shipping"]["address"]["country"],
                    (
                        record["shipping"]["address"]["coordinates"]["lat"],
                        record["shipping"]["address"]["coordinates"]["lon"],
                    ),
                ),
            ),
            "items": [
                (
                    item["product_id"],
                    item["name"],
                    (item["category"]["main"], item["category"]["sub"]),
                    item["price"],
                    item["quantity"],
                    (
                        1 if item["discount"]["applied"] else 0,
                        item["discount"]["percentage"],
                    ),
                    (
                        item["seller"]["id"],
                        item["seller"]["name"],
                        (
                            item["seller"]["rating"]["score"],
                            item["seller"]["rating"]["count"],
                        ),
                    ),
                )
                for item in record["items"]
            ],
        }

    def benchmark_clickhouse_nested(
        self, nested_data: List[Dict], dataset_size: int, test_data_size: int
    ):
        """Benchmark ingestion for ClickHouse nested"""
        print("--- Benchmarking: ClickHouse Nested ---")

        table_name = f"nested_clickhouse_{dataset_size}"

        try:
            client = clickhouse_connect.get_client(
                host="localhost",
                port=8123,
                username="benchmark_user",
                password="benchmark_pass",
                database="benchmark_db",
            )

            # Check if table exists
            result = client.query(f"""
                SELECT count() FROM system.tables
                WHERE database = 'benchmark_db' AND name = '{table_name}'
            """)
            if result.result_rows[0][0] == 0:
                print(f"  ⚠ Table {table_name} does not exist, skipping\n")
                client.close()
                return

            # Get initial row count
            result = client.query(f"SELECT count() FROM {table_name}")
            initial_count = result.result_rows[0][0]

            # Transform nested data to ClickHouse tuple format
            print("  Transforming nested data to ClickHouse format...")
            transformed_batch = []
            for record in nested_data:
                transformed = self._transform_nested_record(record)
                transformed_batch.append(
                    [
                        transformed["order_id"],
                        transformed["timestamp"],
                        transformed["customer"],
                        transformed["payment"],
                        transformed["shipping"],
                        transformed["items"],
                    ]
                )

            # Insert test data with resource monitoring
            monitor = ResourceMonitor(interval=0.1)
            monitor.start()
            start_time = time.perf_counter()
            client.insert(
                table_name,
                transformed_batch,
                column_names=[
                    "order_id",
                    "timestamp",
                    "customer",
                    "payment",
                    "shipping",
                    "items",
                ],
            )
            ingestion_time = time.perf_counter() - start_time
            resources = monitor.stop()

            # Verify insertion
            result = client.query(f"SELECT count() FROM {table_name}")
            final_count = result.result_rows[0][0]

            rows_inserted = final_count - initial_count

            print(
                f"  ✓ Ingested {rows_inserted:,} rows in {ingestion_time:.2f}s "
                f"({rows_inserted / ingestion_time:,.0f} records/sec)"
            )
            print(
                f"    CPU: {resources.cpu_percent_avg:.1f}% avg, {resources.cpu_percent_max:.1f}% max | "
                f"Memory: {resources.memory_mb_avg:.1f} MB avg, {resources.memory_mb_max:.1f} MB max"
            )

            self.results.append(
                {
                    "strategy": "ClickHouse Nested",
                    "ingestion_time_seconds": ingestion_time,
                    "throughput_records_per_sec": rows_inserted / ingestion_time,
                    "mv_refresh_time_seconds": None,
                    "total_time_seconds": ingestion_time,
                    "rows_inserted": rows_inserted,
                    "cpu_percent_avg": resources.cpu_percent_avg,
                    "cpu_percent_max": resources.cpu_percent_max,
                    "memory_mb_avg": resources.memory_mb_avg,
                    "memory_mb_max": resources.memory_mb_max,
                }
            )

            # Cleanup
            client.command(f"""
                ALTER TABLE {table_name}
                DELETE WHERE order_id IN (
                    SELECT order_id FROM {table_name}
                    ORDER BY order_id DESC
                    LIMIT {rows_inserted}
                )
            """)

            print(f"  ✓ Cleaned up {rows_inserted:,} test rows\n")

            client.close()

        except Exception as e:
            print(f"  ✗ Error: {e}\n")

    def benchmark_clickhouse_nested_mv(
        self, nested_data: List[Dict], dataset_size: int, test_data_size: int
    ):
        """Benchmark ingestion for ClickHouse nested with MVs"""
        print("--- Benchmarking: ClickHouse Nested + Materialized Views ---")

        table_name = f"nested_clickhouse_mv_{dataset_size}"

        try:
            client = clickhouse_connect.get_client(
                host="localhost",
                port=8123,
                username="benchmark_user",
                password="benchmark_pass",
                database="benchmark_db",
            )

            # Check if table exists
            result = client.query(f"""
                SELECT count() FROM system.tables
                WHERE database = 'benchmark_db' AND name = '{table_name}'
            """)
            if result.result_rows[0][0] == 0:
                print(f"  ⚠ Table {table_name} does not exist, skipping\n")
                client.close()
                return

            # Get initial row count
            result = client.query(f"SELECT count() FROM {table_name}")
            initial_count = result.result_rows[0][0]

            # Transform nested data to ClickHouse tuple format
            print("  Transforming nested data to ClickHouse format...")
            transformed_batch = []
            for record in nested_data:
                transformed = self._transform_nested_record(record)
                transformed_batch.append(
                    [
                        transformed["order_id"],
                        transformed["timestamp"],
                        transformed["customer"],
                        transformed["payment"],
                        transformed["shipping"],
                        transformed["items"],
                    ]
                )

            # Insert test data with resource monitoring
            ingestion_monitor = ResourceMonitor(interval=0.1)
            ingestion_monitor.start()
            start_time = time.perf_counter()
            client.insert(
                table_name,
                transformed_batch,
                column_names=[
                    "order_id",
                    "timestamp",
                    "customer",
                    "payment",
                    "shipping",
                    "items",
                ],
            )
            ingestion_time = time.perf_counter() - start_time
            ingestion_resources = ingestion_monitor.stop()

            # Measure MV refresh with resource monitoring
            # Use Docker monitoring if available (monitors server), otherwise client monitoring
            if self.clickhouse_docker_container:
                mv_refresh_monitor = monitor_docker_container(
                    self.clickhouse_docker_container, interval=0.5
                )
            else:
                mv_refresh_monitor = ResourceMonitor(interval=0.1)
            mv_refresh_monitor.start()
            mv_refresh_start = time.perf_counter()
            for mv_suffix in [
                "simple_agg",
                "deep_agg",
                "array_agg",
                "complex_agg",
                "seller_agg",
            ]:
                mv_name = f"{table_name}_mv_{mv_suffix}"
                try:
                    # OPTIMIZE FINAL forces all parts to be merged and MVs to be fully updated
                    client.query(f"OPTIMIZE TABLE {mv_name} FINAL")
                    # Verify with a count query to ensure data is visible
                    client.query(f"SELECT count() FROM {mv_name}")
                except Exception:
                    pass  # MV might not exist
            mv_refresh_time = time.perf_counter() - mv_refresh_start
            mv_refresh_resources = mv_refresh_monitor.stop()
            total_time = ingestion_time + mv_refresh_time

            # Verify insertion
            result = client.query(f"SELECT count() FROM {table_name}")
            final_count = result.result_rows[0][0]

            rows_inserted = final_count - initial_count

            print(f"  ✓ Ingested {rows_inserted:,} rows in {ingestion_time:.2f}s")
            print(
                f"    CPU: {ingestion_resources.cpu_percent_avg:.1f}% avg, {ingestion_resources.cpu_percent_max:.1f}% max | "
                f"Memory: {ingestion_resources.memory_mb_avg:.1f} MB avg, {ingestion_resources.memory_mb_max:.1f} MB max"
            )
            print(f"  ✓ MV refresh time: {mv_refresh_time:.2f}s")
            print(
                f"    CPU: {mv_refresh_resources.cpu_percent_avg:.1f}% avg, {mv_refresh_resources.cpu_percent_max:.1f}% max | "
                f"Memory: {mv_refresh_resources.memory_mb_avg:.1f} MB avg, {mv_refresh_resources.memory_mb_max:.1f} MB max"
            )
            print(f"  ✓ Total time: {total_time:.2f}s")

            self.results.append(
                {
                    "strategy": "ClickHouse Nested+MV",
                    "ingestion_time_seconds": ingestion_time,
                    "throughput_records_per_sec": rows_inserted / ingestion_time,
                    "mv_refresh_time_seconds": mv_refresh_time,
                    "total_time_seconds": total_time,
                    "rows_inserted": rows_inserted,
                    "ingestion_cpu_percent_avg": ingestion_resources.cpu_percent_avg,
                    "ingestion_cpu_percent_max": ingestion_resources.cpu_percent_max,
                    "ingestion_memory_mb_avg": ingestion_resources.memory_mb_avg,
                    "ingestion_memory_mb_max": ingestion_resources.memory_mb_max,
                    "mv_refresh_cpu_percent_avg": mv_refresh_resources.cpu_percent_avg,
                    "mv_refresh_cpu_percent_max": mv_refresh_resources.cpu_percent_max,
                    "mv_refresh_memory_mb_avg": mv_refresh_resources.memory_mb_avg,
                    "mv_refresh_memory_mb_max": mv_refresh_resources.memory_mb_max,
                }
            )

            # Cleanup
            client.command(f"""
                ALTER TABLE {table_name}
                DELETE WHERE order_id IN (
                    SELECT order_id FROM {table_name}
                    ORDER BY order_id DESC
                    LIMIT {rows_inserted}
                )
            """)

            print(f"  ✓ Cleaned up {rows_inserted:,} test rows\n")

            client.close()

        except Exception as e:
            print(f"  ✗ Error: {e}\n")

    def benchmark_mongodb(
        self, nested_data: List[Dict], dataset_size: int, test_data_size: int
    ):
        """Benchmark ingestion for MongoDB"""
        print("--- Benchmarking: MongoDB ---")

        collection_name = f"nested_benchmark_{dataset_size}"

        try:
            client = pymongo.MongoClient(self.mongo_conn)
            db = client[self.mongo_db]
            collection = db[collection_name]

            # Clean up any leftover test data from failed previous runs
            # Test data has order IDs starting with "ORD-10" (10 billion+)
            delete_result = collection.delete_many({"order_id": {"$regex": "^ORD-10"}})
            if delete_result.deleted_count > 0:
                print(f"  ✓ Cleaned up {delete_result.deleted_count:,} leftover test rows from previous runs")

            # Get initial count
            initial_count = collection.count_documents({})

            # Insert test data with resource monitoring
            monitor = ResourceMonitor(interval=0.1)
            monitor.start()
            start_time = time.perf_counter()
            result = collection.insert_many(nested_data)
            ingestion_time = time.perf_counter() - start_time
            resources = monitor.stop()

            # Verify insertion
            final_count = collection.count_documents({})
            rows_inserted = final_count - initial_count

            print(
                f"  ✓ Ingested {rows_inserted:,} documents in {ingestion_time:.2f}s "
                f"({rows_inserted / ingestion_time:,.0f} records/sec)"
            )
            print(
                f"    CPU: {resources.cpu_percent_avg:.1f}% avg, {resources.cpu_percent_max:.1f}% max | "
                f"Memory: {resources.memory_mb_avg:.1f} MB avg, {resources.memory_mb_max:.1f} MB max"
            )

            self.results.append(
                {
                    "strategy": "MongoDB",
                    "ingestion_time_seconds": ingestion_time,
                    "throughput_records_per_sec": rows_inserted / ingestion_time,
                    "mv_refresh_time_seconds": None,
                    "total_time_seconds": ingestion_time,
                    "rows_inserted": rows_inserted,
                    "cpu_percent_avg": resources.cpu_percent_avg,
                    "cpu_percent_max": resources.cpu_percent_max,
                    "memory_mb_avg": resources.memory_mb_avg,
                    "memory_mb_max": resources.memory_mb_max,
                }
            )

            # Cleanup - delete the inserted documents by their _id
            inserted_ids = result.inserted_ids
            collection.delete_many({"_id": {"$in": inserted_ids}})

            print(f"  ✓ Cleaned up {len(inserted_ids):,} test documents\n")

            client.close()

        except Exception as e:
            print(f"  ✗ Error: {e}\n")

    # High-Frequency Benchmark Methods
    def _high_freq_benchmark_native_pg(
        self,
        nested_data: List[Dict],
        dataset_size: int,
        batch_size: int,
        batches_per_second: int,
        duration_seconds: int,
    ):
        """High-frequency benchmark for PostgreSQL JSONB"""
        print("--- Benchmarking: PostgreSQL JSONB (High-Frequency) ---")

        table_name = f"nested_native_pg_{dataset_size}"

        try:
            import time as time_module

            conn = psycopg.connect(self.pg_conn)

            # Get initial row count
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {table_name}")
                initial_count = cur.fetchone()[0]

            # Prepare batch iterator
            data_pool = nested_data * (
                (batch_size * batches_per_second * duration_seconds) // len(nested_data)
                + 1
            )
            batch_latencies = []
            total_batches = 0
            total_records = 0
            start_time = time.perf_counter()
            target_interval = 1.0 / batches_per_second

            # Run for specified duration
            next_insert_time = start_time
            while time.perf_counter() - start_time < duration_seconds:
                batch_start = time.perf_counter()

                # Get batch data
                batch_offset = total_batches * batch_size
                batch = data_pool[batch_offset : batch_offset + batch_size]

                # Insert batch
                with conn.cursor() as cur:
                    for record in batch:
                        cur.execute(
                            f"INSERT INTO {table_name} (data) VALUES (%s)",
                            (json.dumps(record),),
                        )
                conn.commit()

                batch_latency = time.perf_counter() - batch_start
                batch_latencies.append(batch_latency)
                total_batches += 1
                total_records += len(batch)

                # Sleep to maintain target frequency
                next_insert_time += target_interval
                sleep_time = next_insert_time - time.perf_counter()
                if sleep_time > 0:
                    time_module.sleep(sleep_time)

            total_time = time.perf_counter() - start_time

            # Verify insertion
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {table_name}")
                final_count = cur.fetchone()[0]

            rows_inserted = final_count - initial_count

            # Calculate metrics
            avg_latency = sum(batch_latencies) / len(batch_latencies)
            p50_latency = sorted(batch_latencies)[len(batch_latencies) // 2]
            p95_latency = sorted(batch_latencies)[int(len(batch_latencies) * 0.95)]
            p99_latency = sorted(batch_latencies)[int(len(batch_latencies) * 0.99)]
            actual_throughput = rows_inserted / total_time

            print(
                f"  ✓ Inserted {rows_inserted:,} rows in {total_batches} batches over {total_time:.2f}s"
            )
            print(f"  ✓ Throughput: {actual_throughput:.0f} records/sec")
            print(
                f"  ✓ Latency - Avg: {avg_latency * 1000:.1f}ms, P50: {p50_latency * 1000:.1f}ms, P95: {p95_latency * 1000:.1f}ms, P99: {p99_latency * 1000:.1f}ms"
            )

            self.high_freq_results.append(
                {
                    "strategy": "PostgreSQL JSONB",
                    "total_batches": total_batches,
                    "total_records": rows_inserted,
                    "duration_seconds": total_time,
                    "throughput": actual_throughput,
                    "avg_latency_ms": avg_latency * 1000,
                    "p50_latency_ms": p50_latency * 1000,
                    "p95_latency_ms": p95_latency * 1000,
                    "p99_latency_ms": p99_latency * 1000,
                }
            )

            # Cleanup
            with conn.cursor() as cur:
                cur.execute(
                    f"DELETE FROM {table_name} WHERE ctid IN (SELECT ctid FROM {table_name} ORDER BY ctid DESC LIMIT {rows_inserted})"
                )
            conn.commit()
            print(f"  ✓ Cleaned up {rows_inserted:,} test rows\n")

            conn.close()

        except Exception as e:
            print(f"  ✗ Error: {e}\n")

    def _high_freq_benchmark_flat_postgresql(
        self,
        df_flat: pd.DataFrame,
        dataset_size: int,
        batch_size: int,
        batches_per_second: int,
        duration_seconds: int,
    ):
        """High-frequency benchmark for Flat PostgreSQL"""
        print("--- Benchmarking: Flat PostgreSQL (High-Frequency) ---")

        table_name = f"flat_pg_{dataset_size}"

        try:
            import time as time_module

            conn = psycopg.connect(self.pg_conn)

            if not self._check_table_exists(conn, table_name):
                print(f"  ⚠ Table {table_name} does not exist, skipping\n")
                conn.close()
                return

            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {table_name}")
                initial_count = cur.fetchone()[0]

            # Prepare batch iterator
            total_records_needed = batch_size * batches_per_second * duration_seconds
            df_pool = pd.concat(
                [df_flat] * (total_records_needed // len(df_flat) + 1),
                ignore_index=True,
            )

            batch_latencies = []
            total_batches = 0
            total_records = 0
            start_time = time.perf_counter()
            target_interval = 1.0 / batches_per_second
            next_insert_time = start_time

            while time.perf_counter() - start_time < duration_seconds:
                batch_start = time.perf_counter()

                batch_offset = total_batches * batch_size
                df_batch = df_pool.iloc[batch_offset : batch_offset + batch_size]

                # Use bulk insert for each batch
                self._pg_bulk_insert(conn, table_name, df_batch)

                batch_latency = time.perf_counter() - batch_start
                batch_latencies.append(batch_latency)
                total_batches += 1
                total_records += len(df_batch)

                next_insert_time += target_interval
                sleep_time = next_insert_time - time.perf_counter()
                if sleep_time > 0:
                    time_module.sleep(sleep_time)

            total_time = time.perf_counter() - start_time

            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {table_name}")
                final_count = cur.fetchone()[0]

            rows_inserted = final_count - initial_count

            avg_latency = sum(batch_latencies) / len(batch_latencies)
            p95_latency = sorted(batch_latencies)[int(len(batch_latencies) * 0.95)]
            actual_throughput = rows_inserted / total_time

            print(
                f"  ✓ Inserted {rows_inserted:,} rows in {total_batches} batches over {total_time:.2f}s"
            )
            print(f"  ✓ Throughput: {actual_throughput:.0f} records/sec")
            print(
                f"  ✓ Latency - Avg: {avg_latency * 1000:.1f}ms, P95: {p95_latency * 1000:.1f}ms"
            )

            self.high_freq_results.append(
                {
                    "strategy": "Flat PostgreSQL",
                    "total_batches": total_batches,
                    "total_records": rows_inserted,
                    "duration_seconds": total_time,
                    "throughput": actual_throughput,
                    "avg_latency_ms": avg_latency * 1000,
                    "p50_latency_ms": sorted(batch_latencies)[len(batch_latencies) // 2]
                    * 1000,
                    "p95_latency_ms": p95_latency * 1000,
                    "p99_latency_ms": sorted(batch_latencies)[
                        int(len(batch_latencies) * 0.99)
                    ]
                    * 1000,
                }
            )

            # Cleanup
            with conn.cursor() as cur:
                cur.execute(
                    f"DELETE FROM {table_name} WHERE ctid IN (SELECT ctid FROM {table_name} ORDER BY ctid DESC LIMIT {rows_inserted})"
                )
            conn.commit()
            print(f"  ✓ Cleaned up {rows_inserted:,} test rows\n")

            conn.close()

        except Exception as e:
            print(f"  ✗ Error: {e}\n")

    def _high_freq_benchmark_clickhouse_flat(
        self,
        df_flat: pd.DataFrame,
        dataset_size: int,
        batch_size: int,
        batches_per_second: int,
        duration_seconds: int,
    ):
        """High-frequency benchmark for ClickHouse Flat"""
        print("--- Benchmarking: ClickHouse Flat (High-Frequency) ---")

        table_name = f"flat_clickhouse_{dataset_size}"

        try:
            import time as time_module

            client = clickhouse_connect.get_client(
                host="localhost",
                port=8123,
                username="benchmark_user",
                password="benchmark_pass",
                database="benchmark_db",
            )

            result = client.query(f"SELECT count() FROM {table_name}")
            initial_count = result.result_rows[0][0]

            total_records_needed = batch_size * batches_per_second * duration_seconds
            df_pool = pd.concat(
                [df_flat] * (total_records_needed // len(df_flat) + 1),
                ignore_index=True,
            )

            batch_latencies = []
            total_batches = 0
            total_records = 0
            start_time = time.perf_counter()
            target_interval = 1.0 / batches_per_second
            next_insert_time = start_time

            while time.perf_counter() - start_time < duration_seconds:
                batch_start = time.perf_counter()

                batch_offset = total_batches * batch_size
                df_batch = df_pool.iloc[batch_offset : batch_offset + batch_size]

                df_clean = df_batch.copy()
                for col in df_clean.select_dtypes(include=["object"]).columns:
                    df_clean[col] = df_clean[col].fillna("")

                client.insert_df(table_name, df_clean)

                batch_latency = time.perf_counter() - batch_start
                batch_latencies.append(batch_latency)
                total_batches += 1
                total_records += len(df_batch)

                next_insert_time += target_interval
                sleep_time = next_insert_time - time.perf_counter()
                if sleep_time > 0:
                    time_module.sleep(sleep_time)

            total_time = time.perf_counter() - start_time

            result = client.query(f"SELECT count() FROM {table_name}")
            final_count = result.result_rows[0][0]
            rows_inserted = final_count - initial_count

            avg_latency = sum(batch_latencies) / len(batch_latencies)
            p95_latency = sorted(batch_latencies)[int(len(batch_latencies) * 0.95)]
            actual_throughput = rows_inserted / total_time

            print(
                f"  ✓ Inserted {rows_inserted:,} rows in {total_batches} batches over {total_time:.2f}s"
            )
            print(f"  ✓ Throughput: {actual_throughput:.0f} records/sec")
            print(
                f"  ✓ Latency - Avg: {avg_latency * 1000:.1f}ms, P95: {p95_latency * 1000:.1f}ms"
            )

            self.high_freq_results.append(
                {
                    "strategy": "ClickHouse Flat",
                    "total_batches": total_batches,
                    "total_records": rows_inserted,
                    "duration_seconds": total_time,
                    "throughput": actual_throughput,
                    "avg_latency_ms": avg_latency * 1000,
                    "p50_latency_ms": sorted(batch_latencies)[len(batch_latencies) // 2]
                    * 1000,
                    "p95_latency_ms": p95_latency * 1000,
                    "p99_latency_ms": sorted(batch_latencies)[
                        int(len(batch_latencies) * 0.99)
                    ]
                    * 1000,
                }
            )

            # Cleanup
            client.query(
                f"DELETE FROM {table_name} WHERE order_id IN (SELECT order_id FROM {table_name} ORDER BY order_id DESC LIMIT {rows_inserted})"
            )
            print(f"  ✓ Cleaned up {rows_inserted:,} test rows\n")

            client.close()

        except Exception as e:
            print(f"  ✗ Error: {e}\n")

    def _high_freq_benchmark_clickhouse_flat_mv(
        self,
        df_flat: pd.DataFrame,
        dataset_size: int,
        batch_size: int,
        batches_per_second: int,
        duration_seconds: int,
    ):
        """High-frequency benchmark for ClickHouse Flat with MVs"""
        print("--- Benchmarking: ClickHouse Flat + MV (High-Frequency) ---")

        table_name = f"flat_clickhouse_mv_{dataset_size}"

        try:
            import time as time_module

            client = clickhouse_connect.get_client(
                host="localhost",
                port=8123,
                username="benchmark_user",
                password="benchmark_pass",
                database="benchmark_db",
            )

            result = client.query(f"SELECT count() FROM {table_name}")
            initial_count = result.result_rows[0][0]

            total_records_needed = batch_size * batches_per_second * duration_seconds
            df_pool = pd.concat(
                [df_flat] * (total_records_needed // len(df_flat) + 1),
                ignore_index=True,
            )

            batch_latencies = []
            mv_check_times = []
            total_batches = 0
            total_records = 0
            start_time = time.perf_counter()
            target_interval = 1.0 / batches_per_second
            next_insert_time = start_time

            while time.perf_counter() - start_time < duration_seconds:
                batch_start = time.perf_counter()

                batch_offset = total_batches * batch_size
                df_batch = df_pool.iloc[batch_offset : batch_offset + batch_size]

                df_clean = df_batch.copy()
                for col in df_clean.select_dtypes(include=["object"]).columns:
                    df_clean[col] = df_clean[col].fillna("")

                client.insert_df(table_name, df_clean)

                # Check MV refresh (sample every 5th batch to reduce overhead)
                if total_batches % 5 == 0:
                    mv_start = time.perf_counter()
                    for mv_suffix in [
                        "simple_agg"
                    ]:  # Just check one MV to reduce overhead
                        mv_name = f"{table_name}_mv_{mv_suffix}"
                        try:
                            client.query(f"OPTIMIZE TABLE {mv_name} FINAL")
                            client.query(f"SELECT count() FROM {mv_name}")
                        except Exception:
                            pass
                    mv_check_times.append(time.perf_counter() - mv_start)

                batch_latency = time.perf_counter() - batch_start
                batch_latencies.append(batch_latency)
                total_batches += 1
                total_records += len(df_batch)

                next_insert_time += target_interval
                sleep_time = next_insert_time - time.perf_counter()
                if sleep_time > 0:
                    time_module.sleep(sleep_time)

            total_time = time.perf_counter() - start_time

            result = client.query(f"SELECT count() FROM {table_name}")
            final_count = result.result_rows[0][0]
            rows_inserted = final_count - initial_count

            avg_latency = sum(batch_latencies) / len(batch_latencies)
            p95_latency = sorted(batch_latencies)[int(len(batch_latencies) * 0.95)]
            actual_throughput = rows_inserted / total_time
            avg_mv_lag = (
                sum(mv_check_times) / len(mv_check_times) if mv_check_times else 0
            )

            print(
                f"  ✓ Inserted {rows_inserted:,} rows in {total_batches} batches over {total_time:.2f}s"
            )
            print(f"  ✓ Throughput: {actual_throughput:.0f} records/sec")
            print(
                f"  ✓ Latency - Avg: {avg_latency * 1000:.1f}ms, P95: {p95_latency * 1000:.1f}ms"
            )
            print(f"  ✓ MV Refresh Lag - Avg: {avg_mv_lag * 1000:.1f}ms")

            self.high_freq_results.append(
                {
                    "strategy": "ClickHouse Flat + MV",
                    "total_batches": total_batches,
                    "total_records": rows_inserted,
                    "duration_seconds": total_time,
                    "throughput": actual_throughput,
                    "avg_latency_ms": avg_latency * 1000,
                    "p50_latency_ms": sorted(batch_latencies)[len(batch_latencies) // 2]
                    * 1000,
                    "p95_latency_ms": p95_latency * 1000,
                    "p99_latency_ms": sorted(batch_latencies)[
                        int(len(batch_latencies) * 0.99)
                    ]
                    * 1000,
                    "avg_mv_lag_ms": avg_mv_lag * 1000,
                }
            )

            # Cleanup
            client.query(
                f"DELETE FROM {table_name} WHERE order_id IN (SELECT order_id FROM {table_name} ORDER BY order_id DESC LIMIT {rows_inserted})"
            )
            print(f"  ✓ Cleaned up {rows_inserted:,} test rows\n")

            client.close()

        except Exception as e:
            print(f"  ✗ Error: {e}\n")

    def _high_freq_benchmark_clickhouse_nested(
        self,
        nested_data: List[Dict],
        dataset_size: int,
        batch_size: int,
        batches_per_second: int,
        duration_seconds: int,
    ):
        """High-frequency benchmark for ClickHouse Nested"""
        print("--- Benchmarking: ClickHouse Nested (High-Frequency) ---")

        table_name = f"nested_clickhouse_{dataset_size}"

        try:
            import time as time_module

            client = clickhouse_connect.get_client(
                host="localhost",
                port=8123,
                username="benchmark_user",
                password="benchmark_pass",
                database="benchmark_db",
            )

            result = client.query(f"SELECT count() FROM {table_name}")
            initial_count = result.result_rows[0][0]

            data_pool = nested_data * (
                (batch_size * batches_per_second * duration_seconds) // len(nested_data)
                + 1
            )

            batch_latencies = []
            total_batches = 0
            total_records = 0
            start_time = time.perf_counter()
            target_interval = 1.0 / batches_per_second
            next_insert_time = start_time

            while time.perf_counter() - start_time < duration_seconds:
                batch_start = time.perf_counter()

                batch_offset = total_batches * batch_size
                batch = data_pool[batch_offset : batch_offset + batch_size]

                transformed_batch = []
                for record in batch:
                    transformed = self._transform_nested_record(record)
                    transformed_batch.append(
                        [
                            transformed["order_id"],
                            transformed["timestamp"],
                            transformed["customer"],
                            transformed["payment"],
                            transformed["shipping"],
                            transformed["items"],
                        ]
                    )

                client.insert(
                    table_name,
                    transformed_batch,
                    column_names=[
                        "order_id",
                        "timestamp",
                        "customer",
                        "payment",
                        "shipping",
                        "items",
                    ],
                )

                batch_latency = time.perf_counter() - batch_start
                batch_latencies.append(batch_latency)
                total_batches += 1
                total_records += len(batch)

                next_insert_time += target_interval
                sleep_time = next_insert_time - time.perf_counter()
                if sleep_time > 0:
                    time_module.sleep(sleep_time)

            total_time = time.perf_counter() - start_time

            result = client.query(f"SELECT count() FROM {table_name}")
            final_count = result.result_rows[0][0]
            rows_inserted = final_count - initial_count

            avg_latency = sum(batch_latencies) / len(batch_latencies)
            p95_latency = sorted(batch_latencies)[int(len(batch_latencies) * 0.95)]
            actual_throughput = rows_inserted / total_time

            print(
                f"  ✓ Inserted {rows_inserted:,} rows in {total_batches} batches over {total_time:.2f}s"
            )
            print(f"  ✓ Throughput: {actual_throughput:.0f} records/sec")
            print(
                f"  ✓ Latency - Avg: {avg_latency * 1000:.1f}ms, P95: {p95_latency * 1000:.1f}ms"
            )

            self.high_freq_results.append(
                {
                    "strategy": "ClickHouse Nested",
                    "total_batches": total_batches,
                    "total_records": rows_inserted,
                    "duration_seconds": total_time,
                    "throughput": actual_throughput,
                    "avg_latency_ms": avg_latency * 1000,
                    "p50_latency_ms": sorted(batch_latencies)[len(batch_latencies) // 2]
                    * 1000,
                    "p95_latency_ms": p95_latency * 1000,
                    "p99_latency_ms": sorted(batch_latencies)[
                        int(len(batch_latencies) * 0.99)
                    ]
                    * 1000,
                }
            )

            # Cleanup
            client.query(
                f"DELETE FROM {table_name} WHERE order_id IN (SELECT order_id FROM {table_name} ORDER BY order_id DESC LIMIT {rows_inserted})"
            )
            print(f"  ✓ Cleaned up {rows_inserted:,} test rows\n")

            client.close()

        except Exception as e:
            print(f"  ✗ Error: {e}\n")

    def _high_freq_benchmark_clickhouse_nested_mv(
        self,
        nested_data: List[Dict],
        dataset_size: int,
        batch_size: int,
        batches_per_second: int,
        duration_seconds: int,
    ):
        """High-frequency benchmark for ClickHouse Nested with MVs"""
        print("--- Benchmarking: ClickHouse Nested + MV (High-Frequency) ---")

        table_name = f"nested_clickhouse_mv_{dataset_size}"

        try:
            import time as time_module

            client = clickhouse_connect.get_client(
                host="localhost",
                port=8123,
                username="benchmark_user",
                password="benchmark_pass",
                database="benchmark_db",
            )

            result = client.query(f"SELECT count() FROM {table_name}")
            initial_count = result.result_rows[0][0]

            data_pool = nested_data * (
                (batch_size * batches_per_second * duration_seconds) // len(nested_data)
                + 1
            )

            batch_latencies = []
            mv_refresh_lags = []
            total_batches = 0
            total_records = 0
            start_time = time.perf_counter()
            target_interval = 1.0 / batches_per_second
            next_insert_time = start_time

            while time.perf_counter() - start_time < duration_seconds:
                batch_start = time.perf_counter()

                batch_offset = total_batches * batch_size
                batch = data_pool[batch_offset : batch_offset + batch_size]

                transformed_batch = []
                for record in batch:
                    transformed = self._transform_nested_record(record)
                    transformed_batch.append(
                        [
                            transformed["order_id"],
                            transformed["timestamp"],
                            transformed["customer"],
                            transformed["payment"],
                            transformed["shipping"],
                            transformed["items"],
                        ]
                    )

                client.insert(
                    table_name,
                    transformed_batch,
                    column_names=[
                        "order_id",
                        "timestamp",
                        "customer",
                        "payment",
                        "shipping",
                        "items",
                    ],
                )

                # Check MV refresh (sample every 5th batch to reduce overhead)
                if total_batches % 5 == 0:
                    mv_start = time.perf_counter()
                    for mv_suffix in [
                        "simple_agg"
                    ]:  # Just check one MV to reduce overhead
                        mv_name = f"{table_name}_mv_{mv_suffix}"
                        try:
                            client.query(f"OPTIMIZE TABLE {mv_name} FINAL")
                            client.query(f"SELECT count() FROM {mv_name}")
                        except Exception:
                            pass
                    mv_refresh_lags.append(time.perf_counter() - mv_start)

                batch_latency = time.perf_counter() - batch_start
                batch_latencies.append(batch_latency)
                total_batches += 1
                total_records += len(batch)

                next_insert_time += target_interval
                sleep_time = next_insert_time - time.perf_counter()
                if sleep_time > 0:
                    time_module.sleep(sleep_time)

            total_time = time.perf_counter() - start_time

            result = client.query(f"SELECT count() FROM {table_name}")
            final_count = result.result_rows[0][0]
            rows_inserted = final_count - initial_count

            avg_latency = sum(batch_latencies) / len(batch_latencies)
            p95_latency = sorted(batch_latencies)[int(len(batch_latencies) * 0.95)]
            actual_throughput = rows_inserted / total_time
            avg_mv_lag = (
                sum(mv_refresh_lags) / len(mv_refresh_lags) if mv_refresh_lags else 0
            )

            print(
                f"  ✓ Inserted {rows_inserted:,} rows in {total_batches} batches over {total_time:.2f}s"
            )
            print(f"  ✓ Throughput: {actual_throughput:.0f} records/sec")
            print(
                f"  ✓ Latency - Avg: {avg_latency * 1000:.1f}ms, P95: {p95_latency * 1000:.1f}ms"
            )
            print(f"  ✓ MV Refresh Lag - Avg: {avg_mv_lag * 1000:.1f}ms")

            self.high_freq_results.append(
                {
                    "strategy": "ClickHouse Nested + MV",
                    "total_batches": total_batches,
                    "total_records": rows_inserted,
                    "duration_seconds": total_time,
                    "throughput": actual_throughput,
                    "avg_latency_ms": avg_latency * 1000,
                    "p50_latency_ms": sorted(batch_latencies)[len(batch_latencies) // 2]
                    * 1000,
                    "p95_latency_ms": p95_latency * 1000,
                    "p99_latency_ms": sorted(batch_latencies)[
                        int(len(batch_latencies) * 0.99)
                    ]
                    * 1000,
                    "avg_mv_lag_ms": avg_mv_lag * 1000,
                }
            )

            # Cleanup
            client.query(
                f"DELETE FROM {table_name} WHERE order_id IN (SELECT order_id FROM {table_name} ORDER BY order_id DESC LIMIT {rows_inserted})"
            )
            print(f"  ✓ Cleaned up {rows_inserted:,} test rows\n")

            client.close()

        except Exception as e:
            print(f"  ✗ Error: {e}\n")

    def _high_freq_benchmark_mongodb(
        self,
        nested_data: List[Dict],
        dataset_size: int,
        batch_size: int,
        batches_per_second: int,
        duration_seconds: int,
    ):
        """High-frequency benchmark for MongoDB"""
        print("--- Benchmarking: MongoDB (High-Frequency) ---")

        collection_name = f"nested_{dataset_size}_orders"

        try:
            import time as time_module

            client = pymongo.MongoClient(self.mongo_conn)
            db = client[self.mongo_db]
            collection = db[collection_name]

            initial_count = collection.count_documents({})

            data_pool = nested_data * (
                (batch_size * batches_per_second * duration_seconds) // len(nested_data)
                + 1
            )

            batch_latencies = []
            total_batches = 0
            total_records = 0
            inserted_ids = []
            start_time = time.perf_counter()
            target_interval = 1.0 / batches_per_second
            next_insert_time = start_time

            while time.perf_counter() - start_time < duration_seconds:
                batch_start = time.perf_counter()

                batch_offset = total_batches * batch_size
                batch = data_pool[batch_offset : batch_offset + batch_size]

                result = collection.insert_many(batch)
                inserted_ids.extend(result.inserted_ids)

                batch_latency = time.perf_counter() - batch_start
                batch_latencies.append(batch_latency)
                total_batches += 1
                total_records += len(batch)

                next_insert_time += target_interval
                sleep_time = next_insert_time - time.perf_counter()
                if sleep_time > 0:
                    time_module.sleep(sleep_time)

            total_time = time.perf_counter() - start_time

            final_count = collection.count_documents({})
            rows_inserted = final_count - initial_count

            avg_latency = sum(batch_latencies) / len(batch_latencies)
            p95_latency = sorted(batch_latencies)[int(len(batch_latencies) * 0.95)]
            actual_throughput = rows_inserted / total_time

            print(
                f"  ✓ Inserted {rows_inserted:,} documents in {total_batches} batches over {total_time:.2f}s"
            )
            print(f"  ✓ Throughput: {actual_throughput:.0f} records/sec")
            print(
                f"  ✓ Latency - Avg: {avg_latency * 1000:.1f}ms, P95: {p95_latency * 1000:.1f}ms"
            )

            self.high_freq_results.append(
                {
                    "strategy": "MongoDB",
                    "total_batches": total_batches,
                    "total_records": rows_inserted,
                    "duration_seconds": total_time,
                    "throughput": actual_throughput,
                    "avg_latency_ms": avg_latency * 1000,
                    "p50_latency_ms": sorted(batch_latencies)[len(batch_latencies) // 2]
                    * 1000,
                    "p95_latency_ms": p95_latency * 1000,
                    "p99_latency_ms": sorted(batch_latencies)[
                        int(len(batch_latencies) * 0.99)
                    ]
                    * 1000,
                }
            )

            # Cleanup
            collection.delete_many({"_id": {"$in": inserted_ids}})
            print(f"  ✓ Cleaned up {len(inserted_ids):,} test documents\n")

            client.close()

        except Exception as e:
            print(f"  ✗ Error: {e}\n")

    def _high_freq_benchmark_clickhouse_flat_join(
        self,
        df_flat: pd.DataFrame,
        dataset_size: int,
        batch_size: int,
        batches_per_second: int,
        duration_seconds: int,
    ):
        """High-frequency benchmark for ClickHouse Flat Join (normalized)"""
        print("--- Benchmarking: ClickHouse Flat Join (High-Frequency) ---")

        table_prefix = f"ch_norm_{dataset_size}"

        try:
            import time as time_module

            client = clickhouse_connect.get_client(
                host="localhost",
                port=8123,
                username="benchmark_user",
                password="benchmark_pass",
                database="benchmark_db",
            )

            # Get initial counts
            result = client.query(f"SELECT count() FROM {table_prefix}_orders")
            initial_count = result.result_rows[0][0]

            total_records_needed = batch_size * batches_per_second * duration_seconds
            df_pool = pd.concat(
                [df_flat] * (total_records_needed // len(df_flat) + 1),
                ignore_index=True,
            )

            batch_latencies = []
            total_batches = 0
            total_records = 0
            start_time = time.perf_counter()
            target_interval = 1.0 / batches_per_second
            next_insert_time = start_time

            while time.perf_counter() - start_time < duration_seconds:
                batch_start = time.perf_counter()

                batch_offset = total_batches * batch_size
                df_batch = df_pool.iloc[batch_offset : batch_offset + batch_size]

                # Clean NaN values that would cause issues with ClickHouse insert
                df_batch = df_batch.fillna("")

                # Insert into normalized tables (same as bulk ingestion)
                # Group by order to handle normalized structure
                for order_id in df_batch["order_id"].unique():
                    order_rows = df_batch[df_batch["order_id"] == order_id]
                    first_row = order_rows.iloc[0]

                    # Insert into orders
                    client.insert(
                        f"{table_prefix}_orders",
                        [
                            [
                                first_row["order_id"],
                                first_row["timestamp"],
                                first_row["customer_id"],
                            ]
                        ],
                        column_names=["order_id", "timestamp", "customer_id"],
                    )

                    # Insert into customers (if not exists - simple approach)
                    try:
                        client.insert(
                            f"{table_prefix}_customers",
                            [
                                [
                                    first_row["customer_id"],
                                    first_row["customer_name"],
                                    first_row["customer_email"],
                                    first_row["customer_tier"],
                                    first_row["customer_lifetime_value"],
                                ]
                            ],
                            column_names=[
                                "customer_id",
                                "customer_name",
                                "customer_email",
                                "customer_tier",
                                "customer_lifetime_value",
                            ],
                        )
                    except:
                        pass  # Already exists

                    # Insert into payments
                    client.insert(
                        f"{table_prefix}_payments",
                        [
                            [
                                first_row["order_id"],
                                first_row["payment_method"],
                                first_row["payment_status"],
                                first_row["payment_amount"],
                                first_row["payment_processor"],
                                first_row["payment_fee"],
                            ]
                        ],
                        column_names=[
                            "order_id",
                            "payment_method",
                            "payment_status",
                            "payment_amount",
                            "payment_processor",
                            "payment_fee",
                        ],
                    )

                    # Insert into shipping
                    client.insert(
                        f"{table_prefix}_shipping",
                        [
                            [
                                first_row["order_id"],
                                first_row["shipping_status"],
                                first_row["shipping_method"],
                                first_row["shipping_cost"],
                                first_row["shipping_city"],
                                first_row["shipping_state"],
                                first_row["shipping_country"],
                                first_row["shipping_lat"],
                                first_row["shipping_lon"],
                            ]
                        ],
                        column_names=[
                            "order_id",
                            "shipping_status",
                            "shipping_method",
                            "shipping_cost",
                            "shipping_city",
                            "shipping_state",
                            "shipping_country",
                            "shipping_lat",
                            "shipping_lon",
                        ],
                    )

                    # Insert each product/item
                    for _, item_row in order_rows.iterrows():
                        # Insert into products (if not exists)
                        try:
                            client.insert(
                                f"{table_prefix}_products",
                                [
                                    [
                                        item_row["product_id"],
                                        item_row["product_name"],
                                        item_row["category_main"],
                                        item_row["category_sub"],
                                        item_row["price"],
                                        item_row["seller_id"],
                                    ]
                                ],
                                column_names=[
                                    "product_id",
                                    "product_name",
                                    "category_main",
                                    "category_sub",
                                    "price",
                                    "seller_id",
                                ],
                            )
                        except:
                            pass  # Already exists

                        # Insert into sellers (if not exists)
                        try:
                            client.insert(
                                f"{table_prefix}_sellers",
                                [
                                    [
                                        item_row["seller_id"],
                                        item_row["seller_name"],
                                        item_row["seller_rating_score"],
                                        item_row["seller_rating_count"],
                                    ]
                                ],
                                column_names=[
                                    "seller_id",
                                    "seller_name",
                                    "seller_rating_score",
                                    "seller_rating_count",
                                ],
                            )
                        except:
                            pass  # Already exists

                        # Insert into order_items
                        client.insert(
                            f"{table_prefix}_order_items",
                            [
                                [
                                    item_row["order_id"],
                                    item_row["product_id"],
                                    item_row["quantity"],
                                    item_row["discount_applied"],
                                    item_row["discount_percentage"],
                                ]
                            ],
                            column_names=[
                                "order_id",
                                "product_id",
                                "quantity",
                                "discount_applied",
                                "discount_percentage",
                            ],
                        )

                batch_latency = time.perf_counter() - batch_start
                batch_latencies.append(batch_latency)
                total_batches += 1
                total_records += len(df_batch)

                next_insert_time += target_interval
                sleep_time = next_insert_time - time.perf_counter()
                if sleep_time > 0:
                    time_module.sleep(sleep_time)

            total_time = time.perf_counter() - start_time

            result = client.query(f"SELECT count() FROM {table_prefix}_orders")
            final_count = result.result_rows[0][0]
            rows_inserted = final_count - initial_count

            avg_latency = sum(batch_latencies) / len(batch_latencies)
            p95_latency = sorted(batch_latencies)[int(len(batch_latencies) * 0.95)]
            actual_throughput = rows_inserted / total_time

            print(
                f"  ✓ Inserted {rows_inserted:,} orders in {total_batches} batches over {total_time:.2f}s"
            )
            print(f"  ✓ Throughput: {actual_throughput:.0f} records/sec")
            print(
                f"  ✓ Latency - Avg: {avg_latency * 1000:.1f}ms, P95: {p95_latency * 1000:.1f}ms"
            )

            self.high_freq_results.append(
                {
                    "strategy": "ClickHouse Flat Join",
                    "total_batches": total_batches,
                    "total_records": rows_inserted,
                    "duration_seconds": total_time,
                    "throughput": actual_throughput,
                    "avg_latency_ms": avg_latency * 1000,
                    "p50_latency_ms": sorted(batch_latencies)[len(batch_latencies) // 2]
                    * 1000,
                    "p95_latency_ms": p95_latency * 1000,
                    "p99_latency_ms": sorted(batch_latencies)[
                        int(len(batch_latencies) * 0.99)
                    ]
                    * 1000,
                }
            )

            # Cleanup - delete from all tables
            client.query(
                f"DELETE FROM {table_prefix}_order_items WHERE order_id IN (SELECT order_id FROM {table_prefix}_orders ORDER BY order_id DESC LIMIT {rows_inserted})"
            )
            client.query(
                f"DELETE FROM {table_prefix}_payments WHERE order_id IN (SELECT order_id FROM {table_prefix}_orders ORDER BY order_id DESC LIMIT {rows_inserted})"
            )
            client.query(
                f"DELETE FROM {table_prefix}_shipping WHERE order_id IN (SELECT order_id FROM {table_prefix}_orders ORDER BY order_id DESC LIMIT {rows_inserted})"
            )
            client.query(
                f"DELETE FROM {table_prefix}_orders WHERE order_id IN (SELECT order_id FROM {table_prefix}_orders ORDER BY order_id DESC LIMIT {rows_inserted})"
            )
            print(f"  ✓ Cleaned up {rows_inserted:,} test orders\n")

            client.close()

        except Exception as e:
            import traceback

            print(f"  ✗ Error: {e}")
            print(f"  Traceback: {traceback.format_exc()}\n")

    def _high_freq_benchmark_clickhouse_flat_join_mv(
        self,
        df_flat: pd.DataFrame,
        dataset_size: int,
        batch_size: int,
        batches_per_second: int,
        duration_seconds: int,
    ):
        """High-frequency benchmark for ClickHouse Flat Join with MVs"""
        print("--- Benchmarking: ClickHouse Flat Join + MV (High-Frequency) ---")

        table_prefix = f"ch_norm_mv_{dataset_size}"

        try:
            import time as time_module

            client = clickhouse_connect.get_client(
                host="localhost",
                port=8123,
                username="benchmark_user",
                password="benchmark_pass",
                database="benchmark_db",
            )

            # Get initial counts
            result = client.query(f"SELECT count() FROM {table_prefix}_orders")
            initial_count = result.result_rows[0][0]

            total_records_needed = batch_size * batches_per_second * duration_seconds
            df_pool = pd.concat(
                [df_flat] * (total_records_needed // len(df_flat) + 1),
                ignore_index=True,
            )

            batch_latencies = []
            mv_check_times = []
            total_batches = 0
            total_records = 0
            start_time = time.perf_counter()
            target_interval = 1.0 / batches_per_second
            next_insert_time = start_time

            while time.perf_counter() - start_time < duration_seconds:
                batch_start = time.perf_counter()

                batch_offset = total_batches * batch_size
                df_batch = df_pool.iloc[batch_offset : batch_offset + batch_size]

                # Clean NaN values that would cause issues with ClickHouse insert
                df_batch = df_batch.fillna("")

                # Insert into normalized tables (same as non-MV version)
                for order_id in df_batch["order_id"].unique():
                    order_rows = df_batch[df_batch["order_id"] == order_id]
                    first_row = order_rows.iloc[0]

                    client.insert(
                        f"{table_prefix}_orders",
                        [
                            [
                                first_row["order_id"],
                                first_row["timestamp"],
                                first_row["customer_id"],
                            ]
                        ],
                        column_names=["order_id", "timestamp", "customer_id"],
                    )

                    try:
                        client.insert(
                            f"{table_prefix}_customers",
                            [
                                [
                                    first_row["customer_id"],
                                    first_row["customer_name"],
                                    first_row["customer_email"],
                                    first_row["customer_tier"],
                                    first_row["customer_lifetime_value"],
                                ]
                            ],
                            column_names=[
                                "customer_id",
                                "customer_name",
                                "customer_email",
                                "customer_tier",
                                "customer_lifetime_value",
                            ],
                        )
                    except:
                        pass

                    client.insert(
                        f"{table_prefix}_payments",
                        [
                            [
                                first_row["order_id"],
                                first_row["payment_method"],
                                first_row["payment_status"],
                                first_row["payment_amount"],
                                first_row["payment_processor"],
                                first_row["payment_fee"],
                            ]
                        ],
                        column_names=[
                            "order_id",
                            "payment_method",
                            "payment_status",
                            "payment_amount",
                            "payment_processor",
                            "payment_fee",
                        ],
                    )

                    client.insert(
                        f"{table_prefix}_shipping",
                        [
                            [
                                first_row["order_id"],
                                first_row["shipping_status"],
                                first_row["shipping_method"],
                                first_row["shipping_cost"],
                                first_row["shipping_city"],
                                first_row["shipping_state"],
                                first_row["shipping_country"],
                                first_row["shipping_lat"],
                                first_row["shipping_lon"],
                            ]
                        ],
                        column_names=[
                            "order_id",
                            "shipping_status",
                            "shipping_method",
                            "shipping_cost",
                            "shipping_city",
                            "shipping_state",
                            "shipping_country",
                            "shipping_lat",
                            "shipping_lon",
                        ],
                    )

                    for _, item_row in order_rows.iterrows():
                        try:
                            client.insert(
                                f"{table_prefix}_products",
                                [
                                    [
                                        item_row["product_id"],
                                        item_row["product_name"],
                                        item_row["category_main"],
                                        item_row["category_sub"],
                                        item_row["price"],
                                        item_row["seller_id"],
                                    ]
                                ],
                                column_names=[
                                    "product_id",
                                    "product_name",
                                    "category_main",
                                    "category_sub",
                                    "price",
                                    "seller_id",
                                ],
                            )
                        except:
                            pass

                        try:
                            client.insert(
                                f"{table_prefix}_sellers",
                                [
                                    [
                                        item_row["seller_id"],
                                        item_row["seller_name"],
                                        item_row["seller_rating_score"],
                                        item_row["seller_rating_count"],
                                    ]
                                ],
                                column_names=[
                                    "seller_id",
                                    "seller_name",
                                    "seller_rating_score",
                                    "seller_rating_count",
                                ],
                            )
                        except:
                            pass

                        client.insert(
                            f"{table_prefix}_order_items",
                            [
                                [
                                    item_row["order_id"],
                                    item_row["product_id"],
                                    item_row["quantity"],
                                    item_row["discount_applied"],
                                    item_row["discount_percentage"],
                                ]
                            ],
                            column_names=[
                                "order_id",
                                "product_id",
                                "quantity",
                                "discount_applied",
                                "discount_percentage",
                            ],
                        )

                # Check MV refresh (sample every 5th batch to reduce overhead)
                if total_batches % 5 == 0:
                    mv_start = time.perf_counter()
                    for mv_suffix in ["simple_agg"]:
                        mv_name = f"{table_prefix}_mv_{mv_suffix}"
                        try:
                            client.query(f"OPTIMIZE TABLE {mv_name} FINAL")
                            client.query(f"SELECT count() FROM {mv_name}")
                        except Exception:
                            pass
                    mv_check_times.append(time.perf_counter() - mv_start)

                batch_latency = time.perf_counter() - batch_start
                batch_latencies.append(batch_latency)
                total_batches += 1
                total_records += len(df_batch)

                next_insert_time += target_interval
                sleep_time = next_insert_time - time.perf_counter()
                if sleep_time > 0:
                    time_module.sleep(sleep_time)

            total_time = time.perf_counter() - start_time

            result = client.query(f"SELECT count() FROM {table_prefix}_orders")
            final_count = result.result_rows[0][0]
            rows_inserted = final_count - initial_count

            avg_latency = sum(batch_latencies) / len(batch_latencies)
            p95_latency = sorted(batch_latencies)[int(len(batch_latencies) * 0.95)]
            actual_throughput = rows_inserted / total_time
            avg_mv_lag = (
                sum(mv_check_times) / len(mv_check_times) if mv_check_times else 0
            )

            print(
                f"  ✓ Inserted {rows_inserted:,} orders in {total_batches} batches over {total_time:.2f}s"
            )
            print(f"  ✓ Throughput: {actual_throughput:.0f} records/sec")
            print(
                f"  ✓ Latency - Avg: {avg_latency * 1000:.1f}ms, P95: {p95_latency * 1000:.1f}ms"
            )
            print(f"  ✓ MV Refresh Lag - Avg: {avg_mv_lag * 1000:.1f}ms")

            self.high_freq_results.append(
                {
                    "strategy": "ClickHouse Join+MV",
                    "total_batches": total_batches,
                    "total_records": rows_inserted,
                    "duration_seconds": total_time,
                    "throughput": actual_throughput,
                    "avg_latency_ms": avg_latency * 1000,
                    "p50_latency_ms": sorted(batch_latencies)[len(batch_latencies) // 2]
                    * 1000,
                    "p95_latency_ms": p95_latency * 1000,
                    "p99_latency_ms": sorted(batch_latencies)[
                        int(len(batch_latencies) * 0.99)
                    ]
                    * 1000,
                    "avg_mv_lag_ms": avg_mv_lag * 1000,
                }
            )

            # Cleanup
            client.query(
                f"DELETE FROM {table_prefix}_order_items WHERE order_id IN (SELECT order_id FROM {table_prefix}_orders ORDER BY order_id DESC LIMIT {rows_inserted})"
            )
            client.query(
                f"DELETE FROM {table_prefix}_payments WHERE order_id IN (SELECT order_id FROM {table_prefix}_orders ORDER BY order_id DESC LIMIT {rows_inserted})"
            )
            client.query(
                f"DELETE FROM {table_prefix}_shipping WHERE order_id IN (SELECT order_id FROM {table_prefix}_orders ORDER BY order_id DESC LIMIT {rows_inserted})"
            )
            client.query(
                f"DELETE FROM {table_prefix}_orders WHERE order_id IN (SELECT order_id FROM {table_prefix}_orders ORDER BY order_id DESC LIMIT {rows_inserted})"
            )
            print(f"  ✓ Cleaned up {rows_inserted:,} test orders\n")

            client.close()

        except Exception as e:
            import traceback

            print(f"  ✗ Error: {e}")
            print(f"  Traceback: {traceback.format_exc()}\n")

    def print_high_freq_results(
        self, batch_size: int, batches_per_second: int, duration_seconds: int
    ):
        """Print formatted high-frequency benchmark results"""
        if not self.high_freq_results:
            print("\nNo results to display")
            return

        print(f"\n{'=' * 80}")
        print("HIGH-FREQUENCY BENCHMARK RESULTS")
        print(
            f"Batch size: {batch_size} | Target: {batches_per_second} batch/sec | Duration: {duration_seconds}s"
        )
        print(f"{'=' * 80}\n")

        # Sort by throughput (highest first)
        sorted_results = sorted(
            self.high_freq_results, key=lambda x: x["throughput"], reverse=True
        )

        # Throughput comparison
        print("THROUGHPUT RANKING")
        print(f"{'=' * 80}")
        for i, result in enumerate(sorted_results, 1):
            throughput = result["throughput"]
            batches = result["total_batches"]
            records = result["total_records"]
            print(
                f"{i}. {result['strategy']:35s}: {throughput:>8,.0f} rec/sec ({batches} batches, {records:,} records)"
            )

        # Latency comparison
        print(f"\n{'=' * 80}")
        print("LATENCY METRICS (per batch)")
        print(f"{'=' * 80}")
        print(
            f"{'Strategy':<35} {'Avg (ms)':>10} {'P50 (ms)':>10} {'P95 (ms)':>10} {'P99 (ms)':>10}"
        )
        print(f"{'-' * 80}")
        for result in sorted_results:
            print(
                f"{result['strategy']:<35} "
                f"{result['avg_latency_ms']:>10.1f} "
                f"{result['p50_latency_ms']:>10.1f} "
                f"{result['p95_latency_ms']:>10.1f} "
                f"{result['p99_latency_ms']:>10.1f}"
            )

        # MV lag if available
        mv_results = [r for r in sorted_results if "avg_mv_lag_ms" in r]
        if mv_results:
            print(f"\n{'=' * 80}")
            print("MATERIALIZED VIEW LAG")
            print(f"{'=' * 80}")
            for result in mv_results:
                print(
                    f"{result['strategy']:35s}: {result['avg_mv_lag_ms']:.1f}ms average"
                )

        print()

    def print_results(self, test_data_size: int):
        """Print formatted ingestion benchmark results"""
        if not self.results:
            print("\nNo results to display")
            return

        print(f"\n{'=' * 80}")
        print(f"INGESTION BENCHMARK RESULTS - {test_data_size:,} test records")
        print(f"{'=' * 80}\n")

        # Sort by total time (fastest first)
        sorted_results = sorted(self.results, key=lambda x: x["total_time_seconds"])

        # ANSI color codes
        GREEN = "\033[92m"
        RED = "\033[91m"
        YELLOW = "\033[93m"
        RESET = "\033[0m"

        # Prepare table data
        table_data = []
        min_ingestion = min(r["ingestion_time_seconds"] for r in sorted_results)
        max_ingestion = max(r["ingestion_time_seconds"] for r in sorted_results)
        min_total = min(r["total_time_seconds"] for r in sorted_results)
        max_total = max(r["total_time_seconds"] for r in sorted_results)

        for result in sorted_results:
            strategy = result["strategy"]
            ingestion_time = result["ingestion_time_seconds"]
            throughput = result["throughput_records_per_sec"]
            mv_time = result["mv_refresh_time_seconds"]
            total_time = result["total_time_seconds"]

            # Color code the fastest and slowest
            ingestion_str = f"{ingestion_time:.2f}s"
            total_str = f"{total_time:.2f}s"

            if ingestion_time == min_ingestion and min_ingestion != max_ingestion:
                ingestion_str = f"{GREEN}{ingestion_str}{RESET}"
            elif ingestion_time == max_ingestion and min_ingestion != max_ingestion:
                ingestion_str = f"{RED}{ingestion_str}{RESET}"

            if total_time == min_total and min_total != max_total:
                total_str = f"{GREEN}{total_str}{RESET}"
            elif total_time == max_total and min_total != max_total:
                total_str = f"{RED}{total_str}{RESET}"

            mv_str = f"{YELLOW}{mv_time:.2f}s{RESET}" if mv_time else "-"
            throughput_str = f"{throughput:,.0f}"

            table_data.append(
                [
                    strategy,
                    ingestion_str,
                    throughput_str,
                    mv_str,
                    total_str,
                ]
            )

        headers = [
            "Strategy",
            "Ingestion Time",
            "Records/Sec",
            "MV Refresh",
            "Total Time",
        ]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

        # Print speedup comparison
        print(f"\n{'=' * 80}")
        print("SPEEDUP COMPARISON (vs Slowest Strategy)")
        print(f"{'=' * 80}\n")

        baseline = max_total
        for result in sorted_results:
            if result["total_time_seconds"] < baseline:
                speedup = baseline / result["total_time_seconds"]
                savings = ((baseline - result["total_time_seconds"]) / baseline) * 100
                symbol = "⚡" if speedup > 1.5 else ""
                print(
                    f"  {result['strategy']:35s}: {speedup:.2f}x faster ({savings:.1f}% time saved) {symbol}"
                )

        print(f"\n{'=' * 80}")
        print("MATERIALIZED VIEW REFRESH OVERHEAD")
        print(f"{'=' * 80}\n")

        mv_strategies = [
            r for r in sorted_results if r["mv_refresh_time_seconds"] is not None
        ]
        if mv_strategies:
            for result in mv_strategies:
                mv_time = result["mv_refresh_time_seconds"]
                ingestion_time = result["ingestion_time_seconds"]
                overhead_pct = (mv_time / ingestion_time) * 100
                print(
                    f"  {result['strategy']:35s}: {mv_time:.2f}s ({overhead_pct:.1f}% of ingestion time)"
                )
        else:
            print("  No materialized view strategies benchmarked")

        print()

        # Generate visualizations
        self._generate_visualizations(sorted_results, test_data_size)

    def _generate_visualizations(self, results: List[Dict], test_data_size: int):
        """Generate ingestion time and throughput visualizations"""
        if not results:
            return

        # Extract data for plotting
        strategies = [r["strategy"] for r in results]
        ingestion_times = [r["ingestion_time_seconds"] for r in results]
        mv_refresh_times = [r["mv_refresh_time_seconds"] or 0 for r in results]
        throughputs = [r["throughput_records_per_sec"] for r in results]

        # Extract CPU and memory data (average and peak)
        cpu_avgs = [r.get("cpu_percent_avg", 0) for r in results]
        cpu_maxs = [r.get("cpu_percent_max", 0) for r in results]
        memory_avgs = [r.get("memory_mb_avg", 0) for r in results]
        memory_maxs = [r.get("memory_mb_max", 0) for r in results]

        # Create figure with six subplots (2x3)
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(20, 12))

        # Chart 1: Ingestion Time (with MV refresh stacked)
        x_pos = range(len(strategies))

        # Stack MV refresh on top of ingestion time
        ax1.bar(x_pos, ingestion_times, label="Ingestion Time", color="steelblue")
        ax1.bar(
            x_pos,
            mv_refresh_times,
            bottom=ingestion_times,
            label="MV Refresh Time",
            color="coral",
        )

        ax1.set_xlabel("Strategy", fontsize=10, fontweight="bold")
        ax1.set_ylabel("Time (seconds)", fontsize=10, fontweight="bold")
        ax1.set_title(
            f"Ingestion Time Comparison ({test_data_size:,} records)",
            fontsize=12,
            fontweight="bold",
        )
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(strategies, rotation=45, ha="right", fontsize=8)
        ax1.legend()
        ax1.grid(axis="y", alpha=0.3)

        # Chart 2: Throughput (records/sec)
        colors = ["steelblue" if mv == 0 else "coral" for mv in mv_refresh_times]
        ax2.bar(x_pos, throughputs, color=colors)

        ax2.set_xlabel("Strategy", fontsize=10, fontweight="bold")
        ax2.set_ylabel("Records/Second", fontsize=10, fontweight="bold")
        ax2.set_title(
            f"Throughput Comparison ({test_data_size:,} records)",
            fontsize=12,
            fontweight="bold",
        )
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(strategies, rotation=45, ha="right", fontsize=8)
        ax2.grid(axis="y", alpha=0.3)

        # Add legend for throughput chart
        base_patch = mpatches.Patch(color="steelblue", label="Base Strategy")
        mv_patch = mpatches.Patch(color="coral", label="With Materialized Views")
        ax2.legend(handles=[base_patch, mv_patch])

        # Chart 3: Average CPU Usage
        ax3.bar(x_pos, cpu_avgs, color="mediumseagreen")
        ax3.set_xlabel("Strategy", fontsize=10, fontweight="bold")
        ax3.set_ylabel("CPU Usage (%)", fontsize=10, fontweight="bold")
        ax3.set_title(
            f"Average CPU Usage ({test_data_size:,} records)",
            fontsize=12,
            fontweight="bold",
        )
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(strategies, rotation=45, ha="right", fontsize=8)
        ax3.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for i, v in enumerate(cpu_avgs):
            if v > 0:
                ax3.text(
                    i, v + max(cpu_avgs) * 0.02, f"{v:.1f}%", ha="center", fontsize=8
                )

        # Chart 4: Peak CPU Usage
        ax4.bar(x_pos, cpu_maxs, color="darkseagreen")
        ax4.set_xlabel("Strategy", fontsize=10, fontweight="bold")
        ax4.set_ylabel("CPU Usage (%)", fontsize=10, fontweight="bold")
        ax4.set_title(
            f"Peak CPU Usage ({test_data_size:,} records)",
            fontsize=12,
            fontweight="bold",
        )
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(strategies, rotation=45, ha="right", fontsize=8)
        ax4.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for i, v in enumerate(cpu_maxs):
            if v > 0:
                ax4.text(
                    i, v + max(cpu_maxs) * 0.02, f"{v:.1f}%", ha="center", fontsize=8
                )

        # Chart 5: Average Memory Usage
        ax5.bar(x_pos, memory_avgs, color="mediumpurple")
        ax5.set_xlabel("Strategy", fontsize=10, fontweight="bold")
        ax5.set_ylabel("Memory Usage (MB)", fontsize=10, fontweight="bold")
        ax5.set_title(
            f"Average Memory Usage ({test_data_size:,} records)",
            fontsize=12,
            fontweight="bold",
        )
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels(strategies, rotation=45, ha="right", fontsize=8)
        ax5.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for i, v in enumerate(memory_avgs):
            if v > 0:
                ax5.text(
                    i, v + max(memory_avgs) * 0.02, f"{v:.0f}", ha="center", fontsize=8
                )

        # Chart 6: Peak Memory Usage
        ax6.bar(x_pos, memory_maxs, color="rebeccapurple")
        ax6.set_xlabel("Strategy", fontsize=10, fontweight="bold")
        ax6.set_ylabel("Memory Usage (MB)", fontsize=10, fontweight="bold")
        ax6.set_title(
            f"Peak Memory Usage ({test_data_size:,} records)",
            fontsize=12,
            fontweight="bold",
        )
        ax6.set_xticks(x_pos)
        ax6.set_xticklabels(strategies, rotation=45, ha="right", fontsize=8)
        ax6.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for i, v in enumerate(memory_maxs):
            if v > 0:
                ax6.text(
                    i, v + max(memory_maxs) * 0.02, f"{v:.0f}", ha="center", fontsize=8
                )

        plt.tight_layout()

        # Save the figure
        os.makedirs("results/{test_data_size}", exist_ok=True)
        output_file = f"results/{test_data_size}/ingestion_benchmark_results_bulk.png"
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"📊 Visualizations saved to: {output_file}\n")

        # Optionally display the plot (commented out for non-interactive environments)
        # plt.show()

    def generate_comparison_visualization(
        self,
        bulk_test_size: int,
        batch_size: int,
        batches_per_second: int,
        duration: int,
    ):
        """Generate comparison visualization between bulk and high-frequency benchmarks"""
        if not self.results or not self.high_freq_results:
            print("⚠ Insufficient data for comparison visualization")
            return

        # Match strategies between bulk and high-freq results
        bulk_dict = {r["strategy"]: r for r in self.results}
        high_freq_dict = {r["strategy"]: r for r in self.high_freq_results}

        # Find common strategies
        common_strategies = set(bulk_dict.keys()) & set(high_freq_dict.keys())
        if not common_strategies:
            print("⚠ No common strategies found between benchmarks")
            return

        # Sort strategies alphabetically for consistent display
        sorted_strategies = sorted(common_strategies)

        # Extract data for comparison
        bulk_throughputs = [
            bulk_dict[s]["throughput_records_per_sec"] for s in sorted_strategies
        ]
        high_freq_throughputs = [
            high_freq_dict[s]["throughput"] for s in sorted_strategies
        ]
        high_freq_latencies = [
            high_freq_dict[s]["p95_latency_ms"] for s in sorted_strategies
        ]

        # Create figure with 3 subplots
        fig = plt.figure(figsize=(20, 6))
        gs = fig.add_gridspec(1, 3, hspace=0.3, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])

        # Chart 1: Throughput Comparison (Bulk vs High-Freq)
        x = range(len(sorted_strategies))
        width = 0.35

        bars1 = ax1.bar(
            [i - width / 2 for i in x],
            bulk_throughputs,
            width,
            label="Bulk Ingestion",
            color="steelblue",
            alpha=0.8,
        )
        bars2 = ax1.bar(
            [i + width / 2 for i in x],
            high_freq_throughputs,
            width,
            label="High-Frequency",
            color="coral",
            alpha=0.8,
        )

        ax1.set_xlabel("Strategy", fontsize=10, fontweight="bold")
        ax1.set_ylabel("Throughput (records/sec)", fontsize=10, fontweight="bold")
        ax1.set_title(
            "Throughput: Bulk vs High-Frequency", fontsize=12, fontweight="bold"
        )
        ax1.set_xticks(x)
        ax1.set_xticklabels(sorted_strategies, rotation=45, ha="right", fontsize=8)
        ax1.legend(loc="upper left")
        ax1.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax1.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f"{int(height):,}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        rotation=0,
                    )

        # Chart 2: Efficiency Ratio (Bulk Throughput / High-Freq Throughput)
        efficiency_ratios = [
            bulk_throughputs[i] / high_freq_throughputs[i]
            if high_freq_throughputs[i] > 0
            else 0
            for i in range(len(sorted_strategies))
        ]

        colors = [
            "green" if r < 2 else "orange" if r < 5 else "red"
            for r in efficiency_ratios
        ]
        bars3 = ax2.bar(x, efficiency_ratios, color=colors, alpha=0.7)

        ax2.set_xlabel("Strategy", fontsize=10, fontweight="bold")
        ax2.set_ylabel("Bulk / High-Freq Ratio", fontsize=10, fontweight="bold")
        ax2.set_title(
            "Efficiency Ratio (Lower = Better for Streaming)",
            fontsize=12,
            fontweight="bold",
        )
        ax2.set_xticks(x)
        ax2.set_xticklabels(sorted_strategies, rotation=45, ha="right", fontsize=8)
        ax2.axhline(
            y=1,
            color="black",
            linestyle="--",
            linewidth=1,
            alpha=0.5,
            label="Equal Performance",
        )
        ax2.legend()
        ax2.grid(axis="y", alpha=0.3)

        # Add value labels
        for bar in bars3:
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.1f}x",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        # Chart 3: High-Frequency P95 Latency
        bars4 = ax3.bar(x, high_freq_latencies, color="purple", alpha=0.7)

        ax3.set_xlabel("Strategy", fontsize=10, fontweight="bold")
        ax3.set_ylabel("P95 Latency (ms)", fontsize=10, fontweight="bold")
        ax3.set_title(
            f"High-Frequency P95 Latency ({batch_size} rec/batch)",
            fontsize=12,
            fontweight="bold",
        )
        ax3.set_xticks(x)
        ax3.set_xticklabels(sorted_strategies, rotation=45, ha="right", fontsize=8)
        ax3.grid(axis="y", alpha=0.3)

        # Add value labels
        for bar in bars4:
            height = bar.get_height()
            ax3.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.0f}ms",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        # Add overall title with configuration info
        fig.suptitle(
            f"Ingestion Benchmark Comparison\n"
            f"Bulk: {bulk_test_size:,} records | High-Freq: {batch_size} rec/batch × {batches_per_second} batch/sec × {duration}s",
            fontsize=14,
            fontweight="bold",
            y=0.98,
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Save the figure
        os.makedirs("results/{bulk_test_size}", exist_ok=True)
        output_file = f"results/{bulk_test_size}/ingestion_comparison_results.png"
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"📊 Comparison visualizations saved to: {output_file}")

        # Print summary insights
        print("\n" + "=" * 80)
        print("BENCHMARK COMPARISON INSIGHTS")
        print("=" * 80)

        # Find best performers
        best_bulk_idx = bulk_throughputs.index(max(bulk_throughputs))
        best_hf_idx = high_freq_throughputs.index(max(high_freq_throughputs))
        lowest_latency_idx = high_freq_latencies.index(min(high_freq_latencies))
        best_ratio_idx = efficiency_ratios.index(min(efficiency_ratios))

        print(f"\n🏆 Best Bulk Throughput: {sorted_strategies[best_bulk_idx]}")
        print(f"   {bulk_throughputs[best_bulk_idx]:,.0f} records/sec")

        print(f"\n🏆 Best High-Frequency Throughput: {sorted_strategies[best_hf_idx]}")
        print(f"   {high_freq_throughputs[best_hf_idx]:,.0f} records/sec")

        print(f"\n⚡ Lowest P95 Latency: {sorted_strategies[lowest_latency_idx]}")
        print(f"   {high_freq_latencies[lowest_latency_idx]:.1f}ms per batch")

        print(f"\n💡 Most Consistent (Best Ratio): {sorted_strategies[best_ratio_idx]}")
        print(f"   {efficiency_ratios[best_ratio_idx]:.2f}x bulk/high-freq ratio")

        print("\n" + "=" * 80 + "\n")
