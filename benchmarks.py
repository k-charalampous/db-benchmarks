import os
from typing import List, Tuple

import clickhouse_connect
import numpy as np
import pandas as pd
import psycopg
import pymongo
import seaborn as sns
from matplotlib import pyplot as plt
from tabulate import tabulate

from clickhouse_flat_join_mv_strategy import ClickHouseFlatJoinMVStrategy
from clickhouse_flat_join_strategy import ClickHouseFlatJoinStrategy
from clickhouse_flat_mv_strategy import ClickHouseFlatMVStrategy
from clickhouse_flat_strategy import ClickhouseStrategy
from clickhouse_nested_mv_strategy import ClickHouseNestedMVStrategy
from clickhouse_nested_strategy import ClickHouseNestedStrategy
from data_generation import DataFileManager
from duckdb_parquet_flat_strategy import DuckDBParquetMinIOStrategy
from mongodb_strategy import MongoDBStrategy
from pg_duckdb_strategy import PgDuckDBStrategy
from postgres_flat_join_strategy import FlatPostgreSQLJoinStrategy
from postgres_flat_strategy import FlatPostgreSQLStrategy
from postgres_flat_strategy_hydra import HydraFlatPostgreSQLStrategy
from postgres_flat_strategy_paradeDB import ParadeDbFlatPostgreSQLStrategy
from postgres_jsonb_stratergy import NativePostgreSQLStrategy


class Benchmark:
    """Main benchmark orchestrator for queries"""

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
        Initialize the benchmark.

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
        self.results = []
        self.storage_stats = []
        self.ingestion_times = []  # Track ingestion time when data is actually loaded
        self.strategy_resources = []  # Track strategy-level aggregate resource usage

        # Docker container names for server-side monitoring
        self.pg_docker_container = pg_docker_container
        self.pg_18_docker_container = pg_18_docker_container
        self.clickhouse_docker_container = clickhouse_docker_container
        self.mongo_docker_container = mongo_docker_container

        # Idle resource baselines
        self.idle_resources = {}

    def measure_idle_resources(self, duration_seconds: float = 5.0):
        """
        Measure idle resource usage for all configured database containers.

        Args:
            duration_seconds: How long to monitor idle resources (default: 5 seconds)
        """
        import time

        from resource_monitor import monitor_docker_container

        print("\n" + "=" * 80)
        print("MEASURING IDLE RESOURCE BASELINES")
        print("=" * 80)
        print(
            f"Monitoring each database for {duration_seconds} seconds while idle...\n"
        )

        containers = {
            "PostgreSQL 17 (pg_duckdb)": self.pg_docker_container,
            "PostgreSQL 18": self.pg_18_docker_container,
            "ClickHouse": self.clickhouse_docker_container,
            "MongoDB": self.mongo_docker_container,
        }

        for db_name, container_name in containers.items():
            if container_name:
                print(
                    f"  Measuring {db_name} ({container_name})...", end=" ", flush=True
                )
                try:
                    monitor = monitor_docker_container(container_name, interval=0.5)
                    monitor.start()
                    time.sleep(duration_seconds)
                    resources = monitor.stop()

                    self.idle_resources[db_name] = resources.to_dict()
                    print(
                        f"✓ CPU: {resources.cpu_percent_avg:.1f}% avg, Memory: {resources.memory_mb_avg:.1f} MB avg"
                    )
                except Exception as e:
                    print(f"✗ Failed: {e}")
                    self.idle_resources[db_name] = {
                        "cpu_percent_avg": 0.0,
                        "cpu_percent_max": 0.0,
                        "memory_mb_avg": 0.0,
                        "memory_mb_max": 0.0,
                        "memory_percent_avg": 0.0,
                        "memory_percent_max": 0.0,
                        "samples": 0,
                    }

        print("\n" + "=" * 80 + "\n")

    def get_pg_storage_stats(self, conn: str, table_name: str, strategy_name: str):
        """Get PostgreSQL table and index sizes with individual index breakdown"""
        try:
            conn = psycopg.connect(conn)

            def bytes_to_human(bytes_val):
                for unit in ["B", "KB", "MB", "GB", "TB"]:
                    if bytes_val < 1024.0:
                        return f"{bytes_val:.2f} {unit}"
                    bytes_val /= 1024.0
                return f"{bytes_val:.2f} PB"

            with conn.cursor() as cur:
                # Get table size
                cur.execute(f"""
                    SELECT pg_relation_size('{table_name}') as table_bytes
                """)
                result = cur.fetchone()
                table_bytes = result[0] if result else 0

                # Add table data storage stat
                self.storage_stats.append(
                    {
                        "strategy": strategy_name,
                        "object": f"{table_name} (table)",
                        "total_size": bytes_to_human(table_bytes),
                        "table_size": bytes_to_human(table_bytes),
                        "index_size": bytes_to_human(0),
                        "total_bytes": table_bytes,
                        "table_bytes": table_bytes,
                        "index_bytes": 0,
                    }
                )

                # Get individual index sizes
                cur.execute(f"""
                    SELECT
                        indexname,
                        pg_relation_size(quote_ident(schemaname) || '.' || quote_ident(indexname)) as index_bytes
                    FROM pg_indexes
                    WHERE tablename = '{table_name}'
                    ORDER BY index_bytes DESC
                """)
                indexes = cur.fetchall()

                for index_name, index_bytes in indexes:
                    self.storage_stats.append(
                        {
                            "strategy": strategy_name,
                            "object": f"{table_name} ({index_name})",
                            "total_size": bytes_to_human(index_bytes),
                            "table_size": bytes_to_human(0),
                            "index_size": bytes_to_human(index_bytes),
                            "total_bytes": index_bytes,
                            "table_bytes": 0,
                            "index_bytes": index_bytes,
                        }
                    )
                    print(f"      {index_name}: {bytes_to_human(index_bytes)}")

            conn.close()
        except Exception as e:
            print(f"    Warning: Could not get storage stats for {table_name}: {e}")

    def get_pg_flat_storage_stats(
        self, conn_string: str, table_name: str, strategy_name: str
    ):
        """Get PostgreSQL storage stats for flat tables with individual index breakdown"""
        try:
            conn = psycopg.connect(conn_string)

            def bytes_to_human(bytes_val):
                for unit in ["B", "KB", "MB", "GB", "TB"]:
                    if bytes_val < 1024.0:
                        return f"{bytes_val:.2f} {unit}"
                    bytes_val /= 1024.0
                return f"{bytes_val:.2f} PB"

            with conn.cursor() as cur:
                # Get table size
                cur.execute(f"""
                    SELECT pg_relation_size('{table_name}') as table_bytes
                """)
                result = cur.fetchone()
                table_bytes = result[0] if result else 0

                # Add table data storage stat
                self.storage_stats.append(
                    {
                        "strategy": strategy_name,
                        "object": f"{table_name} (table)",
                        "total_size": bytes_to_human(table_bytes),
                        "table_size": bytes_to_human(table_bytes),
                        "index_size": bytes_to_human(0),
                        "total_bytes": table_bytes,
                        "table_bytes": table_bytes,
                        "index_bytes": 0,
                    }
                )

                # Get individual index sizes for this table
                cur.execute(f"""
                    SELECT
                        indexname,
                        pg_relation_size(quote_ident(schemaname) || '.' || quote_ident(indexname)) as index_bytes
                    FROM pg_indexes
                    WHERE tablename = '{table_name}'
                    ORDER BY index_bytes DESC
                """)
                indexes = cur.fetchall()

                for index_name, index_bytes in indexes:
                    self.storage_stats.append(
                        {
                            "strategy": strategy_name,
                            "object": f"{table_name} ({index_name})",
                            "total_size": bytes_to_human(index_bytes),
                            "table_size": bytes_to_human(0),
                            "index_size": bytes_to_human(index_bytes),
                            "total_bytes": index_bytes,
                            "table_bytes": 0,
                            "index_bytes": index_bytes,
                        }
                    )
                    print(f"      {index_name}: {bytes_to_human(index_bytes)}")

            conn.close()
        except Exception as e:
            print(f"    Warning: Could not get storage stats for {table_name}: {e}")

    def get_pg_join_storage_stats(
        self, conn_string: str, table_prefix: str, strategy_name: str
    ):
        """Get PostgreSQL storage stats for normalized join tables with individual index breakdown"""
        try:
            conn = psycopg.connect(conn_string)

            def bytes_to_human(bytes_val):
                for unit in ["B", "KB", "MB", "GB", "TB"]:
                    if bytes_val < 1024.0:
                        return f"{bytes_val:.2f} {unit}"
                    bytes_val /= 1024.0
                return f"{bytes_val:.2f} PB"

            with conn.cursor() as cur:
                # Get all tables with this prefix
                cur.execute(
                    """
                    SELECT tablename
                    FROM pg_tables
                    WHERE tablename LIKE %s
                    ORDER BY tablename
                """,
                    (f"{table_prefix}_%",),
                )

                tables = [row[0] for row in cur.fetchall()]

                for table_name in tables:
                    # Get table size
                    cur.execute(f"""
                        SELECT pg_relation_size('{table_name}') as table_bytes
                    """)
                    result = cur.fetchone()
                    table_bytes = result[0] if result else 0

                    # Add table data storage stat
                    self.storage_stats.append(
                        {
                            "strategy": strategy_name,
                            "object": f"{table_name} (table)",
                            "total_size": bytes_to_human(table_bytes),
                            "table_size": bytes_to_human(table_bytes),
                            "index_size": bytes_to_human(0),
                            "total_bytes": table_bytes,
                            "table_bytes": table_bytes,
                            "index_bytes": 0,
                        }
                    )
                    print(f"      {table_name}: {bytes_to_human(table_bytes)}")

                    # Get individual index sizes for this table
                    cur.execute(f"""
                        SELECT
                            indexname,
                            pg_relation_size(quote_ident(schemaname) || '.' || quote_ident(indexname)) as index_bytes
                        FROM pg_indexes
                        WHERE tablename = '{table_name}'
                        ORDER BY index_bytes DESC
                    """)
                    indexes = cur.fetchall()

                    for index_name, index_bytes in indexes:
                        self.storage_stats.append(
                            {
                                "strategy": strategy_name,
                                "object": f"{table_name} ({index_name})",
                                "total_size": bytes_to_human(index_bytes),
                                "table_size": bytes_to_human(0),
                                "index_size": bytes_to_human(index_bytes),
                                "total_bytes": index_bytes,
                                "table_bytes": 0,
                                "index_bytes": index_bytes,
                            }
                        )
                        print(f"        {index_name}: {bytes_to_human(index_bytes)}")

            conn.close()
        except Exception as e:
            print(f"    Warning: Could not get storage stats for {table_prefix}_*: {e}")

    def get_mongodb_storage_stats(self, collection_name: str, strategy_name: str):
        """Get MongoDB collection storage stats"""
        try:
            client = pymongo.MongoClient(self.mongo_conn)
            db = client[self.mongo_db]
            stats = db.command("collStats", collection_name)

            def bytes_to_human(bytes_val):
                for unit in ["B", "KB", "MB", "GB", "TB"]:
                    if bytes_val < 1024.0:
                        return f"{bytes_val:.2f} {unit}"
                    bytes_val /= 1024.0
                return f"{bytes_val:.2f} PB"

            storage_size = stats.get("storageSize", 0)
            total_index_size = stats.get("totalIndexSize", 0)
            total_size = storage_size + total_index_size

            self.storage_stats.append(
                {
                    "strategy": strategy_name,
                    "object": collection_name,
                    "total_size": bytes_to_human(total_size),
                    "table_size": bytes_to_human(storage_size),
                    "index_size": bytes_to_human(total_index_size),
                    "total_bytes": total_size,
                    "table_bytes": storage_size,
                    "index_bytes": total_index_size,
                }
            )
            client.close()
        except Exception as e:
            print(
                f"    Warning: Could not get storage stats for {collection_name}: {e}"
            )

    def get_pg_duckdb_metadata_stats(self, strategy_name: str):
        """Check for pg_duckdb internal tables and metadata storage"""
        try:
            conn = psycopg.connect(self.pg_conn)

            def bytes_to_human(bytes_val):
                for unit in ["B", "KB", "MB", "GB", "TB"]:
                    if bytes_val < 1024.0:
                        return f"{bytes_val:.2f} {unit}"
                    bytes_val /= 1024.0
                return f"{bytes_val:.2f} PB"

            with conn.cursor() as cur:
                # Check for DuckDB internal tables
                cur.execute("""
                    SELECT
                        schemaname || '.' || tablename as full_name,
                        pg_total_relation_size(quote_ident(schemaname) || '.' || quote_ident(tablename)) as total_bytes
                    FROM pg_tables
                    WHERE tablename LIKE '%duckdb%'
                       OR tablename LIKE 'ddb_%'
                       OR schemaname = 'duckdb'
                    ORDER BY total_bytes DESC
                """)
                duckdb_tables = cur.fetchall()

                total_duckdb_overhead = 0
                if duckdb_tables:
                    print("\n  pg_duckdb Internal Tables Found:")
                    for table_name, size_bytes in duckdb_tables:
                        print(f"    {table_name}: {bytes_to_human(size_bytes)}")
                        total_duckdb_overhead += size_bytes

                        # Add to storage stats
                        self.storage_stats.append(
                            {
                                "strategy": f"{strategy_name} (metadata)",
                                "object": table_name,
                                "total_size": bytes_to_human(size_bytes),
                                "table_size": bytes_to_human(size_bytes),
                                "index_size": bytes_to_human(0),
                                "total_bytes": size_bytes,
                                "table_bytes": size_bytes,
                                "index_bytes": 0,
                            }
                        )

                    if total_duckdb_overhead > 0:
                        print(
                            f"  Total pg_duckdb metadata overhead: {bytes_to_human(total_duckdb_overhead)}"
                        )

            conn.close()
        except Exception as e:
            print(f"    Warning: Could not get pg_duckdb metadata stats: {e}")

    def get_clickhouse_storage_stats(self, table_name: str, strategy_name: str):
        """Get ClickHouse storage stats for flat tables"""
        try:
            client = clickhouse_connect.get_client(
                host="localhost",
                port=8123,
                username="benchmark_user",
                password="benchmark_pass",
                database="benchmark_db",
            )

            total_bytes = 0
            data_compressed_bytes = 0
            data_uncompressed_bytes = 0
            marks_bytes = 0  # Primary key index (marks files)

            result = client.query(f"""
                SELECT
                    sum(bytes) as total_bytes,
                    sum(data_compressed_bytes) as data_compressed,
                    sum(data_uncompressed_bytes) as data_uncompressed,
                    sum(marks_bytes) as marks_bytes
                FROM system.parts
                WHERE database = 'benchmark_db'
                    AND table = '{table_name}'
                    AND active = 1
            """)
            if result.result_rows:
                row = result.result_rows[0]
                if row[0]:
                    total_bytes += row[0] or 0
                    data_compressed_bytes += row[1] or 0
                    data_uncompressed_bytes += row[2] or 0
                    marks_bytes += row[3] or 0

            def bytes_to_human(bytes_val):
                for unit in ["B", "KB", "MB", "GB", "TB"]:
                    if bytes_val < 1024.0:
                        return f"{bytes_val:.2f} {unit}"
                    bytes_val /= 1024.0
                return f"{bytes_val:.2f} PB"

            # In ClickHouse:
            # - total_bytes = everything (data + indexes + metadata)
            # - data_compressed_bytes = compressed column data
            # - marks_bytes = primary key index (marks files)
            # Approximate index size = total - data_compressed
            index_and_metadata = total_bytes - data_compressed_bytes

            self.storage_stats.append(
                {
                    "strategy": strategy_name,
                    "object": f"{table_name}_* (all tables)",
                    "total_size": bytes_to_human(total_bytes),
                    "table_size": bytes_to_human(data_compressed_bytes),
                    "index_size": bytes_to_human(index_and_metadata),
                    "total_bytes": total_bytes,
                    "table_bytes": data_compressed_bytes,
                    "index_bytes": index_and_metadata,
                }
            )

            # Print detailed breakdown for debugging
            print("    ClickHouse storage breakdown:")
            print(f"      Total size: {bytes_to_human(total_bytes)}")
            print(f"      Compressed data: {bytes_to_human(data_compressed_bytes)}")
            print(f"      Uncompressed data: {bytes_to_human(data_uncompressed_bytes)}")
            print(f"      Primary key marks: {bytes_to_human(marks_bytes)}")
            print(f"      Index + metadata: {bytes_to_human(index_and_metadata)}")
            print(
                f"      Compression ratio: {data_uncompressed_bytes / data_compressed_bytes if data_compressed_bytes > 0 else 0:.2f}x"
            )

            client.close()
        except Exception as e:
            print(
                f"    Warning: Could not get storage stats for ClickHouse {table_name}: {e}"
            )

    def get_duckdb_parquet_storage_stats(self, table_name: str, strategy_name: str):
        """Get DuckDB Parquet file sizes from MinIO/local storage"""
        try:
            from pathlib import Path

            def bytes_to_human(bytes_val):
                for unit in ["B", "KB", "MB", "GB", "TB"]:
                    if bytes_val < 1024.0:
                        return f"{bytes_val:.2f} {unit}"
                    bytes_val /= 1024.0
                return f"{bytes_val:.2f} PB"

            # Get parquet file sizes from local directory
            parquet_dir = Path("./benchmark_data/parquet")
            total_bytes = 0

            if parquet_dir.exists():
                for table_suffix in ["flat"]:
                    parquet_file = parquet_dir / f"{table_name}.parquet"
                    if parquet_file.exists():
                        file_size = parquet_file.stat().st_size
                        total_bytes += file_size
                        print(
                            f"      {table_suffix}.parquet: {bytes_to_human(file_size)}"
                        )

            self.storage_stats.append(
                {
                    "strategy": strategy_name,
                    "object": f"{table_name} (parquet files)",
                    "total_size": bytes_to_human(total_bytes),
                    "table_size": bytes_to_human(
                        total_bytes
                    ),  # Parquet files are data, no separate indexes
                    "index_size": bytes_to_human(0),  # No separate index files
                    "total_bytes": total_bytes,
                    "table_bytes": total_bytes,
                    "index_bytes": 0,
                }
            )
            print(f"    DuckDB Parquet total size: {bytes_to_human(total_bytes)}")
        except Exception as e:
            print(
                f"    Warning: Could not get storage stats for DuckDB Parquet {table_name}: {e}"
            )

    def run_full_benchmark(self, dataset_size: int = 10000):
        """Run complete query benchmark"""
        print(f"\n{'=' * 80}")
        print(f"QUERY BENCHMARK - {dataset_size:,} records")
        print(f"{'=' * 80}\n")

        # Measure idle resource baselines before running benchmarks
        self.measure_idle_resources(duration_seconds=5.0)

        # Generate and save data to files
        file_manager = DataFileManager()
        file_paths = file_manager.generate_and_save(
            dataset_size, prefix="non_agg_benchmark"
        )

        # If file_paths is None (data already existed), construct paths manually
        if not file_paths:
            file_paths = {
                "nested_jsonl": f"benchmark_data/{dataset_size}_non_agg_benchmark_nested.jsonl",
                "csv_flat": "benchmark_data/1000_non_agg_benchmark_flat.csv",
            }

        query_types = [
            ("simple_where", 0),
            ("complex_where", 0),
            ("pagination_early", 10),  # Page 10
            ("pagination_deep", 1000),  # Page 1000
            ("nested_array_filter", 0),
            # Aggregation queries
            ("simple_nested_agg", 0),
            ("deep_nested_agg", 0),
            ("array_aggregation", 0),
            ("complex_where_agg", 0),
            ("seller_rating_agg", 0),
        ]

        # For flat strategies, pass CSV path directly
        # self.run_hydra(file_paths["csv_flat"], query_types, dataset_size)
        # self.run_paradedb(file_paths["csv_flat"], query_types, dataset_size)
        # self.run_native_pg(file_paths["nested_jsonl"], query_types, dataset_size)
        # self.run_generated_columns(
        #     file_paths["nested_jsonl"], query_types, dataset_size
        # )

        self.run_flat_postgresql(
            file_paths["csv_flat"],
            query_types,
            dataset_size,
            conn=self.pg_conn,
            strategy_name="PostgreSQL17 Flat",
        )
        self.run_flat_postgresql(
            file_paths["csv_flat"],
            query_types,
            dataset_size,
            conn=self.pg_conn_18,
            strategy_name="PostgreSQL18 Flat",
        )
        self.run_flat_postgresql_join(
            file_paths["csv_flat"],
            query_types,
            dataset_size,
            conn=self.pg_conn_18,
            strategy_name="PostgreSQL18 Flat Join",
            table_prefix=f"norm_{dataset_size}",
        )
        self.run_pg_duckdb(file_paths["csv_flat"], query_types, dataset_size)
        self.run_clickhouse_flat(file_paths["csv_flat"], query_types, dataset_size)
        self.run_clickhouse_flat_mv(file_paths["csv_flat"], query_types, dataset_size)
        self.run_clickhouse_flat_join(file_paths["csv_flat"], query_types, dataset_size)
        self.run_clickhouse_flat_join_mv(
            file_paths["csv_flat"], query_types, dataset_size
        )
        # self.run_clickhouse_nested(
        #     file_paths["nested_jsonl"], query_types, dataset_size
        # )
        # self.run_clickhouse_nested_mv(
        #     file_paths["nested_jsonl"], query_types, dataset_size
        # )
        self.run_mongodb(file_paths["nested_jsonl"], query_types, dataset_size)
        # self.run_duckdb(file_paths["nested_jsonl"], query_types, dataset_size)
        self.run_duckdb_parquet_minio(file_paths["csv_flat"], query_types, dataset_size)

        # Print storage summary first, then query performance results
        self.print_storage_summary()
        self.print_results()
        self.visualize_results(dataset_size)
        self.visualize_ingestion_times(dataset_size)

    def run_native_pg(
        self, jsonl_path: str, query_types: List[Tuple[str, int]], dataset_size: int
    ):
        """Strategy 1: JSONB PostgreSQL"""
        print("--- Strategy 1: JSONB PostgreSQL JSONB ---")

        table_name = f"nested_native_pg_{dataset_size}"
        strategy = NativePostgreSQLStrategy(self.pg_conn, table_name=table_name)
        strategy.docker_container = self.pg_docker_container
        strategy.connect()

        print("  Setting up table...")
        ingestion_time = strategy.setup(jsonl_path)

        # Track ingestion time if data was actually loaded
        if ingestion_time is not None:
            self.ingestion_times.append(
                {
                    "strategy": "PostgreSQL JSONB",
                    "time_seconds": ingestion_time,
                    "dataset_size": dataset_size,
                }
            )

        # Start monitoring for entire strategy run
        strategy.start_monitoring()

        for query_type, page in query_types:
            elapsed, rows, resources = strategy.run_query(query_type)
            label = f"{query_type} (page {page})" if page > 0 else query_type
            print(f"  {label}: {elapsed * 1000:.2f}ms ({rows} rows)")

            self.results.append(
                {
                    "strategy": "JSONB PostgreSQL",
                    "query_type": label,
                    "time_ms": elapsed * 1000,
                    "rows": rows,
                    **resources,
                }
            )

        # Stop monitoring and get aggregate resources
        aggregate_resources = strategy.stop_monitoring()
        print(
            f"  Strategy aggregate - CPU: {aggregate_resources['cpu_percent_avg']:.1f}% avg, {aggregate_resources['cpu_percent_max']:.1f}% max | Memory: {aggregate_resources['memory_mb_avg']:.1f} MB avg, {aggregate_resources['memory_mb_max']:.1f} MB max"
        )

        # Store strategy-level aggregate resources
        self.strategy_resources.append(
            {"strategy": "JSONB PostgreSQL", **aggregate_resources}
        )

        strategy.close()

        # Get storage stats
        self.get_pg_storage_stats(
            table_name=table_name, strategy_name="JSONB PostgreSQL", conn=self.pg_conn
        )
        print()

    # def run_generated_columns(
    #     self, jsonl_path: str, query_types: List[Tuple[str, int]], dataset_size: int
    # ):
    #     """Strategy 2: PostgreSQL Generated Columns"""
    #     print("--- Strategy 2: PostgreSQL PostgreSQL Generated Columns ---")

    #     strategy = GeneratedColumnsStrategy(self.pg_conn)
    #     strategy.connect()

    #     table_name = f"nested_native_pg_{dataset_size}"
    #     print("  Setting up table with PostgreSQL Generated Columns...")
    #     strategy.setup(table_name, jsonl_path)

    #     for query_type, page in query_types:
    #         try:
    #             elapsed, rows = strategy.run_query(table_name, query_type, page)
    #             label = f"{query_type} (page {page})" if page > 0 else query_type
    #             print(f"  {label}: {elapsed * 1000:.2f}ms ({rows} rows)")

    #             self.results.append(
    #                 {
    #                     "strategy": "PostgreSQL Generated Columns",
    #                     "query_type": label,
    #                     "time_ms": elapsed * 1000,
    #                     "rows": rows,
    #                 }
    #             )
    #         except Exception as e:
    #             label = f"{query_type} (page {page})" if page > 0 else query_type
    #             print(f"  {label}: Error - {str(e)[:50]}")

    #     strategy.close()
    #     print()

    def run_pg_duckdb(
        self,
        csv_path: str,
        query_types: List[Tuple[str, int]],
        dataset_size: int,
    ):
        """Strategy 3: PostgreSQL with pg_duckdb Extension (Flat Tables)"""
        print("--- Strategy 3: PostgreSQL with pg_duckdb Extension (Flat Tables) ---")

        table_name = f"flat_pg_{dataset_size}"
        strategy = PgDuckDBStrategy(self.pg_conn, table_name=table_name)
        strategy.docker_container = self.pg_docker_container
        strategy.connect()

        print("  Using flat table with pg_duckdb acceleration...")
        strategy.setup(csv_path)

        # Start monitoring for entire strategy run
        strategy.start_monitoring()

        for query_type, page in query_types:
            try:
                elapsed, rows, resources = strategy.run_query(query_type)
                label = f"{query_type} (page {page})" if page > 0 else query_type
                print(f"  {label}: {elapsed * 1000:.2f}ms ({rows} rows)")

                self.results.append(
                    {
                        "strategy": "PostgreSQL pg_duckdb Flat",
                        "query_type": label,
                        "time_ms": elapsed * 1000,
                        "rows": rows,
                        **resources,
                    }
                )
            except Exception as e:
                label = f"{query_type} (page {page})" if page > 0 else query_type
                print(f"  {label}: Error - {str(e)[:50]}")

        # Stop monitoring and get aggregate resources
        aggregate_resources = strategy.stop_monitoring()
        print(
            f"  Strategy aggregate - CPU: {aggregate_resources['cpu_percent_avg']:.1f}% avg, {aggregate_resources['cpu_percent_max']:.1f}% max | Memory: {aggregate_resources['memory_mb_avg']:.1f} MB avg, {aggregate_resources['memory_mb_max']:.1f} MB max"
        )

        # Store strategy-level aggregate resources
        self.strategy_resources.append(
            {"strategy": "PostgreSQL pg_duckdb Flat", **aggregate_resources}
        )

        strategy.close()

        # Get storage stats
        # Note: Storage for flat tables is shared with FlatPostgreSQLStrategy
        # But check for pg_duckdb internal metadata/cache tables
        print("  Note: Flat table storage shared with PostgreSQL Flat strategy")
        self.get_pg_duckdb_metadata_stats("PostgreSQL pg_duckdb")
        print()

    def run_hydra(
        self,
        csv_path: str,
        query_types: List[Tuple[str, int]],
        dataset_size: int,
    ):
        """Strategy 4: PostgreSQL with Hydra Columnar Extension (Flat Tables)"""
        print(
            "--- Strategy 4: PostgreSQL with Hydra Columnar Extension (Flat Tables) ---"
        )

        table_name = f"flat_pg_{dataset_size}"
        conn_string = "host=localhost port=5434 dbname=benchmark_db user=benchmark_user password=benchmark_pass"
        strategy = HydraFlatPostgreSQLStrategy(
            conn_string,
            table_name=table_name,
        )
        strategy.docker_container = self.pg_docker_container
        strategy.connect()

        print("  Using flat table with Hydra columnar optimization...")
        ingestion_time = strategy.setup(csv_path)

        # Track ingestion time if data was actually loaded
        if ingestion_time is not None:
            self.ingestion_times.append(
                {
                    "strategy": "PostgreSQL Hydra Flat",
                    "time_seconds": ingestion_time,
                    "dataset_size": dataset_size,
                }
            )

        # Start monitoring for entire strategy run
        strategy.start_monitoring()

        for query_type, page in query_types:
            try:
                elapsed, rows, resources = strategy.run_query(query_type)
                label = f"{query_type} (page {page})" if page > 0 else query_type
                print(f"  {label}: {elapsed * 1000:.2f}ms ({rows} rows)")

                self.results.append(
                    {
                        "strategy": "PostgreSQL Hydra Flat",
                        "query_type": label,
                        "time_ms": elapsed * 1000,
                        "rows": rows,
                        **resources,
                    }
                )
            except Exception as e:
                label = f"{query_type} (page {page})" if page > 0 else query_type
                print(f"  {label}: Error - {str(e)[:50]}")

        # Stop monitoring and get aggregate resources
        aggregate_resources = strategy.stop_monitoring()
        print(
            f"  Strategy aggregate - CPU: {aggregate_resources['cpu_percent_avg']:.1f}% avg, {aggregate_resources['cpu_percent_max']:.1f}% max | Memory: {aggregate_resources['memory_mb_avg']:.1f} MB avg, {aggregate_resources['memory_mb_max']:.1f} MB max"
        )

        # Store strategy-level aggregate resources
        self.strategy_resources.append(
            {"strategy": "PostgreSQL Hydra Flat", **aggregate_resources}
        )

        strategy.close()

        # Get storage stats
        self.get_pg_flat_storage_stats(
            conn_string, table_name, strategy_name="PostgreSQL Hydra Flat"
        )

    def run_paradedb(
        self,
        csv_path: str,
        query_types: List[Tuple[str, int]],
        dataset_size: int,
    ):
        """Strategy 5: PostgreSQL with ParadeDB Extension (Flat Tables)"""
        print("--- Strategy 5: PostgreSQL with ParadeDB Extension (Flat Tables) ---")

        table_name = f"flat_pg_{dataset_size}"
        conn_string = "host=localhost port=5435 dbname=benchmark_db user=benchmark_user password=benchmark_pass"
        strategy = ParadeDbFlatPostgreSQLStrategy(
            conn_string,
            table_name=table_name,
        )
        strategy.docker_container = self.pg_docker_container
        strategy.connect()

        print("  Using flat table with ParadeDB analytical acceleration...")
        ingestion_time = strategy.setup(csv_path)

        # Track ingestion time if data was actually loaded
        if ingestion_time is not None:
            self.ingestion_times.append(
                {
                    "strategy": "PostgreSQL ParadeDB Flat",
                    "time_seconds": ingestion_time,
                    "dataset_size": dataset_size,
                }
            )

        # Start monitoring for entire strategy run
        strategy.start_monitoring()

        for query_type, page in query_types:
            try:
                elapsed, rows, resources = strategy.run_query(query_type)
                label = f"{query_type} (page {page})" if page > 0 else query_type
                print(f"  {label}: {elapsed * 1000:.2f}ms ({rows} rows)")

                self.results.append(
                    {
                        "strategy": "PostgreSQL ParadeDB Flat",
                        "query_type": label,
                        "time_ms": elapsed * 1000,
                        "rows": rows,
                        **resources,
                    }
                )
            except Exception as e:
                label = f"{query_type} (page {page})" if page > 0 else query_type
                print(f"  {label}: Error - {str(e)[:50]}")

        # Stop monitoring and get aggregate resources
        aggregate_resources = strategy.stop_monitoring()
        print(
            f"  Strategy aggregate - CPU: {aggregate_resources['cpu_percent_avg']:.1f}% avg, {aggregate_resources['cpu_percent_max']:.1f}% max | Memory: {aggregate_resources['memory_mb_avg']:.1f} MB avg, {aggregate_resources['memory_mb_max']:.1f} MB max"
        )

        # Store strategy-level aggregate resources
        self.strategy_resources.append(
            {"strategy": "PostgreSQL ParadeDB Flat", **aggregate_resources}
        )

        strategy.close()

        # Get storage stats
        self.get_pg_flat_storage_stats(
            conn_string, table_name, strategy_name="PostgreSQL ParadeDB Flat"
        )

    def run_mongodb(
        self, jsonl_path: str, query_types: List[Tuple[str, int]], dataset_size: int
    ):
        """Strategy 6: MongoDB"""
        print("--- Strategy 6: MongoDB ---")
        collection_name = f"nested_benchmark_{dataset_size}"

        strategy = MongoDBStrategy(
            self.mongo_conn, self.mongo_db, collection_name=collection_name
        )
        strategy.docker_container = self.mongo_docker_container

        print("  Setting up collection...")
        ingestion_time = strategy.setup(jsonl_path)

        # Track ingestion time if data was actually loaded
        if ingestion_time is not None:
            self.ingestion_times.append(
                {
                    "strategy": "MongoDB",
                    "time_seconds": ingestion_time,
                    "dataset_size": dataset_size,
                }
            )

        # Start monitoring for entire strategy run
        strategy.start_monitoring()

        for query_type, page in query_types:
            try:
                elapsed, rows, resources = strategy.run_query(query_type)
                label = f"{query_type} (page {page})" if page > 0 else query_type
                print(f"  {label}: {elapsed * 1000:.2f}ms ({rows} rows)")

                self.results.append(
                    {
                        "strategy": "MongoDB",
                        "query_type": label,
                        "time_ms": elapsed * 1000,
                        "rows": rows,
                        **resources,
                    }
                )
            except Exception as e:
                label = f"{query_type} (page {page})" if page > 0 else query_type
                print(f"  {label}: Error - {str(e)[:50]}")

        # Stop monitoring and get aggregate resources
        aggregate_resources = strategy.stop_monitoring()
        print(
            f"  Strategy aggregate - CPU: {aggregate_resources['cpu_percent_avg']:.1f}% avg, {aggregate_resources['cpu_percent_max']:.1f}% max | Memory: {aggregate_resources['memory_mb_avg']:.1f} MB avg, {aggregate_resources['memory_mb_max']:.1f} MB max"
        )

        # Store strategy-level aggregate resources
        self.strategy_resources.append({"strategy": "MongoDB", **aggregate_resources})

        strategy.close()

        # Get storage stats
        self.get_mongodb_storage_stats(collection_name, "MongoDB")
        print()

    # def run_duckdb(
    #     self, jsonl_path: str, query_types: List[Tuple[str, int]], dataset_size: int
    # ):
    #     """Strategy 4: DuckDB"""
    #     print("--- Strategy 4: DuckDB Standalone ---")

    #     strategy = DuckDBStrategy()

    #     table_name = f"nested_benchmark_{dataset_size}"
    #     print("  Setting up table...")
    #     strategy.setup(table_name, jsonl_path)

    #     for query_type, page in query_types:
    #         elapsed, rows = strategy.run_query(table_name, query_type, page)
    #         label = f"{query_type} (page {page})" if page > 0 else query_type
    #         if elapsed > 0:
    #             print(f"  {label}: {elapsed * 1000:.2f}ms ({rows} rows)")

    #             self.results.append(
    #                 {
    #                     "strategy": "DuckDB",
    #                     "query_type": label,
    #                     "time_ms": elapsed * 1000,
    #                     "rows": rows,
    #                 }
    #             )
    #         else:
    #             print(f"  {label}: N/A (not implemented)")

    #     strategy.close()
    #     print()

    def run_duckdb_parquet_minio(
        self,
        csv_path: str,
        query_types: List[Tuple[str, int]],
        dataset_size: int,
    ):
        """Strategy 5: DuckDB with Parquet on MinIO (single flat Parquet file)"""
        print(
            "--- Strategy 5: DuckDB with Parquet on MinIO (single flat Parquet file) ---"
        )

        table_name = f"parquet_benchmark_{dataset_size}"
        strategy = DuckDBParquetMinIOStrategy(table_name=table_name)
        strategy.docker_container = None

        print("  Setting up flat Parquet file in MinIO...")
        strategy.setup(csv_path)

        # Start monitoring for entire strategy run
        strategy.start_monitoring()

        for query_type, page in query_types:
            elapsed, rows, resources = strategy.run_query(query_type)
            label = f"{query_type} (page {page})" if page > 0 else query_type
            if elapsed > 0:
                print(f"  {label}: {elapsed * 1000:.2f}ms ({rows} rows)")

                self.results.append(
                    {
                        "strategy": "DuckDB Parquet MinIO",
                        "query_type": label,
                        "time_ms": elapsed * 1000,
                        "rows": rows,
                        **resources,
                    }
                )
            else:
                print(f"  {label}: N/A (not implemented)")

        # Stop monitoring and get aggregate resources
        aggregate_resources = strategy.stop_monitoring()
        print(
            f"  Strategy aggregate - CPU: {aggregate_resources['cpu_percent_avg']:.1f}% avg, {aggregate_resources['cpu_percent_max']:.1f}% max | Memory: {aggregate_resources['memory_mb_avg']:.1f} MB avg, {aggregate_resources['memory_mb_max']:.1f} MB max"
        )

        # Store strategy-level aggregate resources
        self.strategy_resources.append(
            {"strategy": "DuckDB Parquet MinIO", **aggregate_resources}
        )

        # Get storage stats for DuckDB Parquet files
        print("  Getting storage stats...")
        self.get_duckdb_parquet_storage_stats(table_name, "DuckDB Parquet MinIO")

        strategy.close()
        print()

    def run_flat_postgresql(
        self,
        csv_path: str,
        query_types: List[Tuple[str, int]],
        dataset_size: int,
        conn: str,
        strategy_name: str,
    ):
        """Strategy 5: PostgreSQL Flat (single denormalized table)"""
        print(f"--- Strategy 5: {strategy_name} (single denormalized table) ---")

        table_name = f"flat_pg_{dataset_size}"
        strategy = FlatPostgreSQLStrategy(conn, table_name=table_name)
        strategy.docker_container = self.pg_18_docker_container
        strategy.connect()

        print("  Setting up single denormalized table...")
        ingestion_time = strategy.setup(csv_path)

        # Track ingestion time if data was actually loaded
        if ingestion_time is not None:
            self.ingestion_times.append(
                {
                    "strategy": strategy_name,
                    "time_seconds": ingestion_time,
                    "dataset_size": dataset_size,
                }
            )

        # Start monitoring for entire strategy run
        strategy.start_monitoring()

        for query_type, page in query_types:
            elapsed, rows, resources = strategy.run_query(query_type)
            label = f"{query_type} (page {page})" if page > 0 else query_type
            print(f"  {label}: {elapsed * 1000:.2f}ms ({rows} rows)")

            self.results.append(
                {
                    "strategy": strategy_name,
                    "query_type": label,
                    "time_ms": elapsed * 1000,
                    "rows": rows,
                    **resources,
                }
            )

        # Stop monitoring and get aggregate resources
        aggregate_resources = strategy.stop_monitoring()
        print(
            f"  Strategy aggregate - CPU: {aggregate_resources['cpu_percent_avg']:.1f}% avg, {aggregate_resources['cpu_percent_max']:.1f}% max | Memory: {aggregate_resources['memory_mb_avg']:.1f} MB avg, {aggregate_resources['memory_mb_max']:.1f} MB max"
        )

        # Store strategy-level aggregate resources
        self.strategy_resources.append(
            {"strategy": strategy_name, **aggregate_resources}
        )

        strategy.close()

        # Get storage stats
        self.get_pg_flat_storage_stats(conn, table_name, strategy_name=strategy_name)
        print()

    def run_flat_postgresql_join(
        self,
        csv_path: str,
        query_types: List[Tuple[str, int]],
        dataset_size: int,
        conn: str,
        strategy_name: str,
        table_prefix: str = "norm",
    ):
        """Strategy: PostgreSQL Flat Join (multiple normalized tables with JOINs)"""
        print(f"--- Strategy: {strategy_name} (normalized tables with JOINs) ---")

        strategy = FlatPostgreSQLJoinStrategy(conn, table_prefix=table_prefix)
        strategy.docker_container = self.pg_18_docker_container
        strategy.connect()

        print("  Setting up normalized tables...")
        ingestion_time = strategy.setup(csv_path)

        # Track ingestion time if data was actually loaded
        if ingestion_time is not None:
            self.ingestion_times.append(
                {
                    "strategy": strategy_name,
                    "time_seconds": ingestion_time,
                    "dataset_size": dataset_size,
                }
            )

        # Start monitoring for entire strategy run
        strategy.start_monitoring()

        for query_type, page in query_types:
            try:
                elapsed, rows, resources = strategy.run_query(query_type)
                label = f"{query_type} (page {page})" if page > 0 else query_type
                print(f"  {label}: {elapsed * 1000:.2f}ms ({rows} rows) [with JOINs]")

                self.results.append(
                    {
                        "strategy": strategy_name,
                        "query_type": label,
                        "time_ms": elapsed * 1000,
                        "rows": rows,
                        **resources,
                    }
                )
            except Exception as e:
                label = f"{query_type} (page {page})" if page > 0 else query_type
                print(f"  {label}: Error - {str(e)[:100]}")

        # Stop monitoring and get aggregate resources
        aggregate_resources = strategy.stop_monitoring()
        print(
            f"  Strategy aggregate - CPU: {aggregate_resources['cpu_percent_avg']:.1f}% avg, {aggregate_resources['cpu_percent_max']:.1f}% max | Memory: {aggregate_resources['memory_mb_avg']:.1f} MB avg, {aggregate_resources['memory_mb_max']:.1f} MB max"
        )

        # Store strategy-level aggregate resources
        self.strategy_resources.append(
            {"strategy": strategy_name, **aggregate_resources}
        )

        strategy.close()

        # Get storage stats for all normalized tables
        self.get_pg_join_storage_stats(conn, table_prefix, strategy_name=strategy_name)
        print()

    def run_clickhouse_flat_join(
        self,
        csv_path: str,
        query_types: List[Tuple[str, int]],
        dataset_size: int,
    ):
        """Strategy: ClickHouse Flat Join (normalized tables with JOINs)"""
        print("--- Strategy: ClickHouse Flat Join (normalized tables with JOINs) ---")

        try:
            table_prefix = f"ch_norm_{dataset_size}"
            strategy = ClickHouseFlatJoinStrategy(table_prefix=table_prefix)
            strategy.docker_container = self.clickhouse_docker_container

            print("  Setting up normalized tables...")
            ingestion_time = strategy.setup(csv_path)

            # Track ingestion time if data was actually loaded
            if ingestion_time is not None:
                self.ingestion_times.append(
                    {
                        "strategy": "ClickHouse Flat Join",
                        "time_seconds": ingestion_time,
                        "dataset_size": dataset_size,
                    }
                )

            # Start monitoring for entire strategy run
            strategy.start_monitoring()

            for query_type, page in query_types:
                try:
                    elapsed, rows, resources = strategy.run_query(query_type)
                    label = f"{query_type} (page {page})" if page > 0 else query_type
                    print(
                        f"  {label}: {elapsed * 1000:.2f}ms ({rows} rows) [with JOINs]"
                    )

                    self.results.append(
                        {
                            "strategy": "ClickHouse Flat Join",
                            "query_type": label,
                            "time_ms": elapsed * 1000,
                            "rows": rows,
                            **resources,
                        }
                    )
                except Exception as e:
                    label = f"{query_type} (page {page})" if page > 0 else query_type
                    print(f"  {label}: Error - {str(e)[:100]}")

            # Stop monitoring and get aggregate resources
            aggregate_resources = strategy.stop_monitoring()
            print(
                f"  Strategy aggregate - CPU: {aggregate_resources['cpu_percent_avg']:.1f}% avg, {aggregate_resources['cpu_percent_max']:.1f}% max | Memory: {aggregate_resources['memory_mb_avg']:.1f} MB avg, {aggregate_resources['memory_mb_max']:.1f} MB max"
            )

            # Store strategy-level aggregate resources
            self.strategy_resources.append(
                {"strategy": "ClickHouse Flat Join", **aggregate_resources}
            )

            # Get storage stats for all normalized tables
            # Note: ClickHouse storage stats need special handling for multiple tables
            print("  Getting storage stats for normalized tables...")
            for table_suffix in [
                "orders",
                "customers",
                "products",
                "payments",
                "shipping",
                "sellers",
                "order_items",
            ]:
                table_name = f"{table_prefix}_{table_suffix}"
                self.get_clickhouse_storage_stats(table_name, "ClickHouse Flat Join")

            strategy.close()
        except Exception as e:
            print(f"  ⚠ ClickHouse Flat Join test failed: {e}")
            print("  Make sure ClickHouse is running: docker-compose ps clickhouse")

        print()

    def run_clickhouse_nested(
        self,
        jsonl_path: str,
        query_types: List[Tuple[str, int]],
        dataset_size: int,
    ):
        """Strategy: ClickHouse Nested (with Tuple and Array types)"""
        print("--- Strategy: ClickHouse Nested (with Tuple and Array types) ---")

        try:
            table_name = f"nested_clickhouse_{dataset_size}"
            strategy = ClickHouseNestedStrategy(table_name=table_name)
            strategy.docker_container = self.clickhouse_docker_container

            print("  Setting up nested table with Tuple/Array types...")
            ingestion_time = strategy.setup(jsonl_path)

            # Track ingestion time if data was actually loaded
            if ingestion_time is not None:
                self.ingestion_times.append(
                    {
                        "strategy": "ClickHouse Nested",
                        "time_seconds": ingestion_time,
                        "dataset_size": dataset_size,
                    }
                )

            # Start monitoring for entire strategy run
            strategy.start_monitoring()

            for query_type, page in query_types:
                try:
                    elapsed, rows, resources = strategy.run_query(query_type)
                    label = f"{query_type} (page {page})" if page > 0 else query_type
                    print(
                        f"  {label}: {elapsed * 1000:.2f}ms ({rows} rows) [nested structures]"
                    )

                    self.results.append(
                        {
                            "strategy": "ClickHouse Nested",
                            "query_type": label,
                            "time_ms": elapsed * 1000,
                            "rows": rows,
                            **resources,
                        }
                    )
                except Exception as e:
                    label = f"{query_type} (page {page})" if page > 0 else query_type
                    print(f"  {label}: Error - {str(e)[:100]}")

            # Stop monitoring and get aggregate resources
            aggregate_resources = strategy.stop_monitoring()
            print(
                f"  Strategy aggregate - CPU: {aggregate_resources['cpu_percent_avg']:.1f}% avg, {aggregate_resources['cpu_percent_max']:.1f}% max | Memory: {aggregate_resources['memory_mb_avg']:.1f} MB avg, {aggregate_resources['memory_mb_max']:.1f} MB max"
            )

            # Store strategy-level aggregate resources
            self.strategy_resources.append(
                {"strategy": "ClickHouse Nested", **aggregate_resources}
            )

            # Get storage stats
            self.get_clickhouse_storage_stats(table_name, "ClickHouse Nested")

            strategy.close()
        except Exception as e:
            print(f"  ⚠ ClickHouse Nested test failed: {e}")
            print("  Make sure ClickHouse is running: docker-compose ps clickhouse")

        print()

    def run_clickhouse_flat(
        self,
        csv_path: str,
        query_types: List[Tuple[str, int]],
        dataset_size: int,
    ):
        """Strategy 6: ClickHouse Flat (single denormalized table)"""
        print("--- Strategy 6: ClickHouse Flat (single denormalized table) ---")

        try:
            table_name = f"flat_clickhouse_{dataset_size}"
            strategy = ClickhouseStrategy(table_name=table_name)
            strategy.docker_container = self.clickhouse_docker_container

            print("  Setting up single denormalized table...")
            ingestion_time = strategy.setup(csv_path)

            # Track ingestion time if data was actually loaded
            if ingestion_time is not None:
                self.ingestion_times.append(
                    {
                        "strategy": "ClickHouse Flat",
                        "time_seconds": ingestion_time,
                        "dataset_size": dataset_size,
                    }
                )

            # Start monitoring for entire strategy run
            strategy.start_monitoring()

            for query_type, page in query_types:
                elapsed, rows, resources = strategy.run_query(query_type)
                label = f"{query_type} (page {page})" if page > 0 else query_type
                print(f"  {label}: {elapsed * 1000:.2f}ms ({rows} rows) ")

                self.results.append(
                    {
                        "strategy": "ClickHouse Flat",
                        "query_type": label,
                        "time_ms": elapsed * 1000,
                        "rows": rows,
                        **resources,
                    }
                )

            # Stop monitoring and get aggregate resources
            aggregate_resources = strategy.stop_monitoring()
            print(
                f"  Strategy aggregate - CPU: {aggregate_resources['cpu_percent_avg']:.1f}% avg, {aggregate_resources['cpu_percent_max']:.1f}% max | Memory: {aggregate_resources['memory_mb_avg']:.1f} MB avg, {aggregate_resources['memory_mb_max']:.1f} MB max"
            )

            # Store strategy-level aggregate resources
            self.strategy_resources.append(
                {"strategy": "ClickHouse Flat", **aggregate_resources}
            )

            # Get storage stats
            self.get_clickhouse_storage_stats(table_name, "ClickHouse Flat")

            strategy.close()
        except Exception as e:
            print(e)
            print(f"  ⚠ ClickHouse test failed: {e}")
            print("  Make sure ClickHouse is running: docker-compose ps clickhouse")

        print()

    def run_clickhouse_flat_mv(
        self,
        csv_path: str,
        query_types: List[Tuple[str, int]],
        dataset_size: int,
    ):
        """Strategy: ClickHouse Flat with Materialized Views (aggregations only)"""
        print(
            "--- Strategy: ClickHouse Flat + Materialized Views (pre-computed aggregations) ---"
        )

        try:
            table_name = f"flat_clickhouse_mv_{dataset_size}"
            strategy = ClickHouseFlatMVStrategy(table_name=table_name)
            strategy.docker_container = self.clickhouse_docker_container

            print("  Setting up table and materialized views...")
            ingestion_time = strategy.setup(csv_path)

            # Track ingestion time if data was actually loaded
            if ingestion_time is not None:
                self.ingestion_times.append(
                    {
                        "strategy": "ClickHouse Flat+MV",
                        "time_seconds": ingestion_time,
                        "dataset_size": dataset_size,
                    }
                )

            # Only run aggregation queries (MVs are designed for these)
            agg_queries = [
                "simple_nested_agg",
                "deep_nested_agg",
                "array_aggregation",
                "complex_where_agg",
                "seller_rating_agg",
            ]

            # Start monitoring for entire strategy run
            strategy.start_monitoring()

            for query_type, page in query_types:
                if query_type not in agg_queries:
                    continue  # Skip non-aggregation queries

                elapsed, rows, resources = strategy.run_query(query_type)
                label = f"{query_type} (page {page})" if page > 0 else query_type
                print(f"  {label}: {elapsed * 1000:.2f}ms ({rows} rows) [from MV]")

                self.results.append(
                    {
                        "strategy": "ClickHouse Flat+MV",
                        "query_type": label,
                        "time_ms": elapsed * 1000,
                        "rows": rows,
                        **resources,
                    }
                )

            # Stop monitoring and get aggregate resources
            aggregate_resources = strategy.stop_monitoring()
            print(
                f"  Strategy aggregate - CPU: {aggregate_resources['cpu_percent_avg']:.1f}% avg, {aggregate_resources['cpu_percent_max']:.1f}% max | Memory: {aggregate_resources['memory_mb_avg']:.1f} MB avg, {aggregate_resources['memory_mb_max']:.1f} MB max"
            )

            # Store strategy-level aggregate resources
            self.strategy_resources.append(
                {"strategy": "ClickHouse Flat+MV", **aggregate_resources}
            )

            # Get storage stats (includes MVs)
            self.get_clickhouse_storage_stats(table_name, "ClickHouse Flat+MV")

            strategy.close()
        except Exception as e:
            print(f"  ⚠ ClickHouse Flat+MV test failed: {e}")
            print("  Make sure ClickHouse is running: docker-compose ps clickhouse")

        print()

    def run_clickhouse_flat_join_mv(
        self,
        csv_path: str,
        query_types: List[Tuple[str, int]],
        dataset_size: int,
    ):
        """Strategy: ClickHouse Flat Join with Materialized Views (aggregations only)"""
        print(
            "--- Strategy: ClickHouse Flat Join + Materialized Views (pre-computed aggregations) ---"
        )

        try:
            table_prefix = f"ch_norm_mv_{dataset_size}"
            strategy = ClickHouseFlatJoinMVStrategy(table_prefix=table_prefix)
            strategy.docker_container = self.clickhouse_docker_container

            print("  Setting up normalized tables and materialized views...")
            ingestion_time = strategy.setup(csv_path)

            # Track ingestion time if data was actually loaded
            if ingestion_time is not None:
                self.ingestion_times.append(
                    {
                        "strategy": "ClickHouse Join+MV",
                        "time_seconds": ingestion_time,
                        "dataset_size": dataset_size,
                    }
                )

            # Only run aggregation queries (MVs are designed for these)
            agg_queries = [
                "simple_nested_agg",
                "deep_nested_agg",
                "array_aggregation",
                "complex_where_agg",
                "seller_rating_agg",
            ]

            # Start monitoring for entire strategy run
            strategy.start_monitoring()

            for query_type, page in query_types:
                if query_type not in agg_queries:
                    continue  # Skip non-aggregation queries

                try:
                    elapsed, rows, resources = strategy.run_query(query_type)
                    label = f"{query_type} (page {page})" if page > 0 else query_type
                    print(f"  {label}: {elapsed * 1000:.2f}ms ({rows} rows) [from MV]")

                    self.results.append(
                        {
                            "strategy": "ClickHouse Join+MV",
                            "query_type": label,
                            "time_ms": elapsed * 1000,
                            "rows": rows,
                            **resources,
                        }
                    )
                except Exception as e:
                    label = f"{query_type} (page {page})" if page > 0 else query_type
                    print(f"  {label}: Error - {str(e)[:100]}")

            # Stop monitoring and get aggregate resources
            aggregate_resources = strategy.stop_monitoring()
            print(
                f"  Strategy aggregate - CPU: {aggregate_resources['cpu_percent_avg']:.1f}% avg, {aggregate_resources['cpu_percent_max']:.1f}% max | Memory: {aggregate_resources['memory_mb_avg']:.1f} MB avg, {aggregate_resources['memory_mb_max']:.1f} MB max"
            )

            # Store strategy-level aggregate resources
            self.strategy_resources.append(
                {"strategy": "ClickHouse Join+MV", **aggregate_resources}
            )

            # Get storage stats for all tables including MVs
            print("  Getting storage stats for normalized tables and MVs...")
            for table_suffix in [
                "orders",
                "customers",
                "products",
                "payments",
                "shipping",
                "sellers",
                "order_items",
            ]:
                table_name = f"{table_prefix}_{table_suffix}"
                self.get_clickhouse_storage_stats(table_name, "ClickHouse Join+MV")

            strategy.close()
        except Exception as e:
            print(f"  ⚠ ClickHouse Join+MV test failed: {e}")
            print("  Make sure ClickHouse is running: docker-compose ps clickhouse")

        print()

    def run_clickhouse_nested_mv(
        self,
        jsonl_path: str,
        query_types: List[Tuple[str, int]],
        dataset_size: int,
    ):
        """Strategy: ClickHouse Nested with Materialized Views (aggregations only)"""
        print(
            "--- Strategy: ClickHouse Nested + Materialized Views (pre-computed aggregations) ---"
        )

        try:
            table_name = f"nested_clickhouse_mv_{dataset_size}"
            strategy = ClickHouseNestedMVStrategy(table_name=table_name)
            strategy.docker_container = self.clickhouse_docker_container

            print("  Setting up nested table and materialized views...")
            ingestion_time = strategy.setup(jsonl_path)

            # Track ingestion time if data was actually loaded
            if ingestion_time is not None:
                self.ingestion_times.append(
                    {
                        "strategy": "ClickHouse Nested+MV",
                        "time_seconds": ingestion_time,
                        "dataset_size": dataset_size,
                    }
                )

            # Only run aggregation queries (MVs are designed for these)
            agg_queries = [
                "simple_nested_agg",
                "deep_nested_agg",
                "array_aggregation",
                "complex_where_agg",
                "seller_rating_agg",
            ]

            # Start monitoring for entire strategy run
            strategy.start_monitoring()

            for query_type, page in query_types:
                if query_type not in agg_queries:
                    continue  # Skip non-aggregation queries

                try:
                    elapsed, rows, resources = strategy.run_query(query_type)
                    label = f"{query_type} (page {page})" if page > 0 else query_type
                    print(f"  {label}: {elapsed * 1000:.2f}ms ({rows} rows) [from MV]")

                    self.results.append(
                        {
                            "strategy": "ClickHouse Nested+MV",
                            "query_type": label,
                            "time_ms": elapsed * 1000,
                            "rows": rows,
                            **resources,
                        }
                    )
                except Exception as e:
                    label = f"{query_type} (page {page})" if page > 0 else query_type
                    print(f"  {label}: Error - {str(e)[:100]}")

            # Stop monitoring and get aggregate resources
            aggregate_resources = strategy.stop_monitoring()
            print(
                f"  Strategy aggregate - CPU: {aggregate_resources['cpu_percent_avg']:.1f}% avg, {aggregate_resources['cpu_percent_max']:.1f}% max | Memory: {aggregate_resources['memory_mb_avg']:.1f} MB avg, {aggregate_resources['memory_mb_max']:.1f} MB max"
            )

            # Store strategy-level aggregate resources
            self.strategy_resources.append(
                {"strategy": "ClickHouse Nested+MV", **aggregate_resources}
            )

            # Get storage stats (includes MVs)
            self.get_clickhouse_storage_stats(table_name, "ClickHouse Nested+MV")

            strategy.close()
        except Exception as e:
            print(f"  ⚠ ClickHouse Nested+MV test failed: {e}")
            print("  Make sure ClickHouse is running: docker-compose ps clickhouse")

        print()

    def print_storage_summary(self):
        """Print storage statistics summary"""
        if not self.storage_stats:
            return

        print(f"\n{'=' * 80}")
        print("STORAGE SUMMARY - Disk Space Required")
        print(f"{'=' * 80}\n")

        # Create DataFrame for storage stats
        df_storage = pd.DataFrame(self.storage_stats)

        # Print detailed table
        print("Detailed Storage Breakdown:")
        print(
            tabulate(
                df_storage[
                    ["strategy", "object", "total_size", "table_size", "index_size"]
                ],
                headers=[
                    "Strategy",
                    "Object",
                    "Total Size",
                    "Table Size",
                    "Index Size",
                ],
                tablefmt="grid",
            )
        )

        # Print summary by strategy
        print(f"\n{'=' * 80}")
        print("Storage by Strategy (Total)")
        print(f"{'=' * 80}\n")

        strategy_totals = df_storage.groupby("strategy").agg(
            {"total_bytes": "sum", "table_bytes": "sum", "index_bytes": "sum"}
        )

        def bytes_to_human(bytes_val):
            for unit in ["B", "KB", "MB", "GB", "TB"]:
                if bytes_val < 1024.0:
                    return f"{bytes_val:.2f} {unit}"
                bytes_val /= 1024.0
            return f"{bytes_val:.2f} PB"

        strategy_summary = []
        for strategy in strategy_totals.index:
            total = strategy_totals.loc[strategy, "total_bytes"]
            table = strategy_totals.loc[strategy, "table_bytes"]
            index = strategy_totals.loc[strategy, "index_bytes"]
            strategy_summary.append(
                {
                    "Strategy": strategy,
                    "Total Size": bytes_to_human(total),
                    "Table Data": bytes_to_human(table),
                    "Indexes": bytes_to_human(index),
                    "Total Bytes": total,
                }
            )

        df_summary = pd.DataFrame(strategy_summary).sort_values(
            "Total Bytes", ascending=False
        )

        # ANSI color codes
        GREEN = "\033[92m"
        RED = "\033[91m"
        RESET = "\033[0m"

        # Color the smallest and largest
        min_size = df_summary["Total Bytes"].min()
        max_size = df_summary["Total Bytes"].max()

        colored_rows = []
        for _, row in df_summary.iterrows():
            if row["Total Bytes"] == min_size and min_size != max_size:
                colored_rows.append(
                    [
                        f"{GREEN}{row['Strategy']}{RESET}",
                        f"{GREEN}{row['Total Size']}{RESET}",
                        f"{GREEN}{row['Table Data']}{RESET}",
                        f"{GREEN}{row['Indexes']}{RESET}",
                    ]
                )
            elif row["Total Bytes"] == max_size and min_size != max_size:
                colored_rows.append(
                    [
                        f"{RED}{row['Strategy']}{RESET}",
                        f"{RED}{row['Total Size']}{RESET}",
                        f"{RED}{row['Table Data']}{RESET}",
                        f"{RED}{row['Indexes']}{RESET}",
                    ]
                )
            else:
                colored_rows.append(
                    [
                        row["Strategy"],
                        row["Total Size"],
                        row["Table Data"],
                        row["Indexes"],
                    ]
                )

        print(
            tabulate(
                colored_rows,
                headers=["Strategy", "Total Size", "Table Data", "Indexes"],
                tablefmt="grid",
            )
        )

        # Calculate compression ratios (comparing to baseline)
        if len(df_summary) > 1:
            print(f"\n{'=' * 80}")
            print("Storage Efficiency vs Largest Strategy")
            print(f"{'=' * 80}\n")

            baseline = max_size
            for _, row in df_summary.iterrows():
                if row["Total Bytes"] > 0 and row["Total Bytes"] != baseline:
                    ratio = baseline / row["Total Bytes"]
                    savings = ((baseline - row["Total Bytes"]) / baseline) * 100
                    symbol = "💾" if savings > 20 else ""
                    print(
                        f"  {row['Strategy']:35s}: {ratio:.2f}x smaller ({savings:.1f}% savings) {symbol}"
                    )

    def print_results(self):
        """Print formatted results with comparisons"""
        print(f"\n{'=' * 80}")
        print("QUERY PERFORMANCE RESULTS SUMMARY")
        print(f"{'=' * 80}\n")

        df = pd.DataFrame(self.results)

        pivot = df.pivot(index="query_type", columns="strategy", values="time_ms")

        # ANSI color codes
        GREEN = "\033[92m"
        RED = "\033[91m"
        RESET = "\033[0m"

        # Create colored version of the pivot table
        pivot_colored = pivot.copy()
        for idx in pivot.index:
            row = pivot.loc[idx]
            valid_values = row.dropna()
            if len(valid_values) > 0:
                min_val = valid_values.min()
                max_val = valid_values.max()

                # Color the values in this row
                for col in pivot.columns:
                    val = pivot.loc[idx, col]
                    if pd.notna(val):
                        if val == min_val and min_val != max_val:
                            pivot_colored.loc[idx, col] = f"{GREEN}{val:.2f}{RESET}"
                        elif val == max_val and min_val != max_val:
                            pivot_colored.loc[idx, col] = f"{RED}{val:.2f}{RESET}"
                        else:
                            pivot_colored.loc[idx, col] = f"{val:.2f}"

        print(tabulate(pivot_colored, headers="keys", tablefmt="grid"))

        print(f"\n{'=' * 80}")
        print("SPEEDUP vs JSONB PostgreSQL (Baseline)")
        print(f"{'=' * 80}\n")

        for query_type in pivot.index:
            if "JSONB PostgreSQL" not in pivot.columns:
                continue

            baseline = pivot.loc[query_type, "JSONB PostgreSQL"]
            if pd.isna(baseline) or baseline == 0:
                continue

            print(f"\n{query_type}:")

            for strategy in pivot.columns:
                if strategy == "JSONB PostgreSQL":
                    continue

                value = pivot.loc[query_type, strategy]
                if pd.notna(value) and value > 0:
                    speedup = baseline / value
                    symbol = "⚡" if speedup > 1.5 else "🐌" if speedup < 0.7 else ""
                    print(f"  {strategy:35s}: {speedup:5.2f}x {symbol}")

    def visualize_ingestion_times(self, dataset_size: int):
        """Generate separate visualization for ingestion times"""
        if not self.ingestion_times:
            print("No ingestion times to visualize (data already existed)")
            return

        print("\n" + "=" * 80)
        print("INGESTION TIME VISUALIZATION")
        print("=" * 80)

        df = pd.DataFrame(self.ingestion_times)

        # Set style
        sns.set_style("whitegrid")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: Ingestion Time in seconds
        strategies = df["strategy"].tolist()
        times = df["time_seconds"].tolist()
        colors = sns.color_palette("husl", len(strategies))

        bars1 = ax1.bar(strategies, times, color=colors)
        ax1.set_xlabel("Strategy", fontsize=11)
        ax1.set_ylabel("Time (seconds)", fontsize=11)
        ax1.set_title(
            f"Data Ingestion Time\nDataset: {dataset_size:,} records",
            fontsize=12,
            fontweight="bold",
        )
        ax1.tick_params(axis="x", rotation=45)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}s",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        # Plot 2: Throughput (records per second)
        throughputs = [dataset_size / t for t in times]
        bars2 = ax2.bar(strategies, throughputs, color=colors)
        ax2.set_xlabel("Strategy", fontsize=11)
        ax2.set_ylabel("Records/Second", fontsize=11)
        ax2.set_title(
            f"Ingestion Throughput\nDataset: {dataset_size:,} records",
            fontsize=12,
            fontweight="bold",
        )
        ax2.tick_params(axis="x", rotation=45)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

        # Add value labels on bars
        for bar in bars2:
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:,.0f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.tight_layout()

        output_file = f"results/{dataset_size}/ingestion_times.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"📊 Ingestion visualization saved to: {output_file}")
        plt.close()

    def visualize_results(self, dataset_size: int):
        """Generate visualizations comparing benchmark results"""
        if not self.results:
            print("No results to visualize")
            return

        os.makedirs(f"results/{dataset_size}", exist_ok=True)
        df = pd.DataFrame(self.results)
        pivot = df.pivot(index="query_type", columns="strategy", values="time_ms")

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = (16, 10)

        # Create a figure with multiple subplots
        has_storage = bool(self.storage_stats)
        # Updated layout to accommodate CPU and memory plots (including peak metrics)
        subplot_layout = (5, 2) if has_storage else (4, 2)
        plt.figure(figsize=(20, 24 if has_storage else 20))

        # 1. Bar chart comparing all strategies across query types
        ax1 = plt.subplot(*subplot_layout, 1)
        pivot.plot(kind="bar", ax=ax1, width=0.8)
        ax1.set_title(
            f"Query Performance Comparison ({dataset_size:,} records)",
            fontsize=14,
            fontweight="bold",
        )
        ax1.set_xlabel("Query Type", fontsize=12)
        ax1.set_ylabel("Time (ms)", fontsize=12)
        ax1.legend(
            title="Strategy", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9
        )
        ax1.tick_params(axis="x", rotation=45)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

        # 2. Heatmap showing relative performance
        ax2 = plt.subplot(*subplot_layout, 2)
        # Normalize by row (each query type) to show relative performance
        pivot_normalized = pivot.div(pivot.min(axis=1), axis=0)

        # Create heatmap with per-row color normalization
        # This ensures each benchmark (row) has its own color scale
        data_to_plot = pivot_normalized.T

        # Create a custom heatmap with row-wise normalization
        im_data = []
        for col_idx, col in enumerate(data_to_plot.columns):
            col_data = data_to_plot[col].values
            # Normalize each column (which represents a query type) independently
            col_min, col_max = col_data.min(), col_data.max()
            if col_max > col_min:
                normalized_col = (col_data - col_min) / (col_max - col_min)
            else:
                normalized_col = col_data * 0  # All same value, set to 0
            im_data.append(normalized_col)

        # Stack the normalized columns
        im_array = np.column_stack(im_data)

        # Create the heatmap with the normalized data for colors
        # but annotate with the actual relative performance values
        cmap = sns.color_palette("RdYlGn_r", as_cmap=True)
        im = ax2.imshow(im_array, cmap=cmap, aspect="auto")

        # Add annotations with actual values
        for i in range(len(data_to_plot.index)):
            for j in range(len(data_to_plot.columns)):
                value = data_to_plot.iloc[i, j]
                text = ax2.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=9,
                )

        # Set ticks and labels
        ax2.set_xticks(np.arange(len(data_to_plot.columns)))
        ax2.set_yticks(np.arange(len(data_to_plot.index)))
        ax2.set_xticklabels(data_to_plot.columns, rotation=45, ha="right")
        ax2.set_yticklabels(data_to_plot.index)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label(
            "Relative Performance\n(per benchmark, lower is better)", fontsize=10
        )

        ax2.set_title(
            "Relative Performance Heatmap\n(colors normalized per query type)",
            fontsize=14,
            fontweight="bold",
        )
        ax2.set_xlabel("Query Type", fontsize=12)
        ax2.set_ylabel("Strategy", fontsize=12)

        # 3. Speedup comparison (vs Native PostgreSQL if available)
        # ax3 = plt.subplot(*subplot_layout, 3)
        # if "Native PostgreSQL" in pivot.columns:
        #     speedup_df = pivot.div(pivot["Native PostgreSQL"], axis=0)
        #     speedup_df = speedup_df.drop(columns=["Native PostgreSQL"])
        #     speedup_df.plot(kind="bar", ax=ax3, width=0.8)
        #     ax3.axhline(
        #         y=1.0,
        #         color="r",
        #         linestyle="--",
        #         linewidth=2,
        #         label="Baseline (Native PG)",
        #     )
        #     ax3.set_title(
        #         "Speedup vs Native PostgreSQL\n(higher is better)",
        #         fontsize=14,
        #         fontweight="bold",
        #     )
        #     ax3.set_xlabel("Query Type", fontsize=12)
        #     ax3.set_ylabel("Speedup Factor (x)", fontsize=12)
        #     ax3.legend(
        #         title="Strategy", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9
        #     )
        #     plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha="right")
        # else:
        #     ax3.text(
        #         0.5,
        #         0.5,
        #         "Native PostgreSQL results not available",
        #         ha="center",
        #         va="center",
        #         transform=ax3.transAxes,
        #     )

        # 4. Box plots showing performance distribution split by query type (queries vs aggregations)
        ax4_left = plt.subplot(*subplot_layout, 3)
        ax4_right = plt.subplot(*subplot_layout, 4)

        # Define aggregation query types
        agg_query_types = [
            "simple_nested_agg",
            "deep_nested_agg",
            "array_aggregation",
            "complex_where_agg",
            "seller_rating_agg",
        ]

        # Split data into queries and aggregations
        df_queries = df[
            ~df["query_type"].str.contains("|".join(agg_query_types), regex=True)
        ]
        df_aggs = df[
            df["query_type"].str.contains("|".join(agg_query_types), regex=True)
        ]

        # Left plot: Queries distribution
        if not df_queries.empty:
            sns.boxplot(data=df_queries, y="strategy", x="time_ms", ax=ax4_left)
            ax4_left.set_title(
                "Performance Distribution - Queries", fontsize=14, fontweight="bold"
            )
            ax4_left.set_xlabel("Time (ms)", fontsize=12)
            ax4_left.set_ylabel("Strategy", fontsize=12)
        else:
            ax4_left.text(
                0.5,
                0.5,
                "No query data",
                ha="center",
                va="center",
                transform=ax4_left.transAxes,
            )

        # Right plot: Aggregations distribution
        if not df_aggs.empty:
            sns.boxplot(data=df_aggs, y="strategy", x="time_ms", ax=ax4_right)
            ax4_right.set_title(
                "Performance Distribution - Aggregations",
                fontsize=14,
                fontweight="bold",
            )
            ax4_right.set_xlabel("Time (ms)", fontsize=12)
            ax4_right.set_ylabel("Strategy", fontsize=12)
        else:
            ax4_right.text(
                0.5,
                0.5,
                "No aggregation data",
                ha="center",
                va="center",
                transform=ax4_right.transAxes,
            )

        # 5. CPU Usage by Strategy (use strategy-level aggregates with idle baseline)
        ax5_left = plt.subplot(*subplot_layout, 5)
        if self.strategy_resources:
            df_strategy_res = pd.DataFrame(self.strategy_resources)
            df_cpu = df_strategy_res.set_index("strategy")[
                "cpu_percent_avg"
            ].sort_values(ascending=False)
        else:
            # Fallback to old method if strategy_resources not available
            df_cpu = (
                df.groupby("strategy")["cpu_percent_avg"]
                .mean()
                .sort_values(ascending=False)
            )

        # Map strategy names to database types for idle baseline
        strategy_to_db = {
            "JSONB PostgreSQL": "PostgreSQL 17",
            "PostgreSQL pg_duckdb Flat": "PostgreSQL 17",
            "PostgreSQL Hydra Flat": "PostgreSQL 17",
            "PostgreSQL ParadeDB Flat": "PostgreSQL 17",
            "PostgreSQL17 Flat": "PostgreSQL 17",
            "PostgreSQL17 Flat Join": "PostgreSQL 17",
            "PostgreSQL18 Flat": "PostgreSQL 18",
            "PostgreSQL18 Flat Join": "PostgreSQL 18",
            "ClickHouse Flat": "ClickHouse",
            "ClickHouse Flat+MV": "ClickHouse",
            "ClickHouse Flat Join": "ClickHouse",
            "ClickHouse Join+MV": "ClickHouse",
            "ClickHouse Nested": "ClickHouse",
            "ClickHouse Nested+MV": "ClickHouse",
            "MongoDB": "MongoDB",
        }

        # Create stacked bar chart with idle baseline
        idle_cpu = []
        active_cpu = []
        for strategy in df_cpu.index:
            db_type = strategy_to_db.get(strategy)
            idle_val = 0.0
            if db_type and db_type in self.idle_resources:
                idle_val = self.idle_resources[db_type]["cpu_percent_avg"]
            idle_cpu.append(idle_val)
            active_cpu.append(
                max(0, df_cpu[strategy] - idle_val)
            )  # Active = Total - Idle

        bars_idle = ax5_left.barh(
            df_cpu.index, idle_cpu, color="lightgray", label="Idle Baseline"
        )
        bars_active = ax5_left.barh(
            df_cpu.index,
            active_cpu,
            left=idle_cpu,
            color="mediumseagreen",
            label="Active Increase",
        )

        ax5_left.set_title(
            "Average CPU Usage by Strategy", fontsize=14, fontweight="bold"
        )
        ax5_left.set_xlabel("CPU (%)", fontsize=12)
        ax5_left.set_ylabel("Strategy", fontsize=12)
        ax5_left.grid(axis="x", alpha=0.3)
        ax5_left.legend(loc="lower right", fontsize=9)

        # Add value labels on bars (total)
        for i, (strategy, total) in enumerate(df_cpu.items()):
            ax5_left.text(
                total,
                i,
                f"{total:.1f}%",
                ha="left",
                va="center",
                fontsize=9,
            )

        # 6. Memory Usage by Strategy (use strategy-level aggregates with idle baseline)
        ax5_right = plt.subplot(*subplot_layout, 6)
        if self.strategy_resources:
            df_memory = df_strategy_res.set_index("strategy")[
                "memory_mb_avg"
            ].sort_values(ascending=False)
        else:
            # Fallback to old method if strategy_resources not available
            df_memory = (
                df.groupby("strategy")["memory_mb_avg"]
                .mean()
                .sort_values(ascending=False)
            )

        # Create stacked bar chart with idle baseline
        idle_mem = []
        active_mem = []
        for strategy in df_memory.index:
            db_type = strategy_to_db.get(strategy)
            idle_val = 0.0
            if db_type and db_type in self.idle_resources:
                idle_val = self.idle_resources[db_type]["memory_mb_avg"]
            idle_mem.append(idle_val)
            active_mem.append(
                max(0, df_memory[strategy] - idle_val)
            )  # Active = Total - Idle

        ax5_right.barh(
            df_memory.index, idle_mem, color="lightgray", label="Idle Baseline"
        )
        ax5_right.barh(
            df_memory.index,
            active_mem,
            left=idle_mem,
            color="mediumpurple",
            label="Active Increase",
        )

        ax5_right.set_title(
            "Average Memory Usage by Strategy", fontsize=14, fontweight="bold"
        )
        ax5_right.set_xlabel("Memory (MB)", fontsize=12)
        ax5_right.set_ylabel("Strategy", fontsize=12)
        ax5_right.grid(axis="x", alpha=0.3)
        ax5_right.legend(loc="lower right", fontsize=9)

        # Add value labels on bars (total)
        for i, (strategy, total) in enumerate(df_memory.items()):
            ax5_right.text(
                total,
                i,
                f"{total:.1f}",
                ha="left",
                va="center",
                fontsize=9,
            )

        # 7. Peak CPU Usage by Strategy (use strategy-level aggregates with idle baseline)
        ax7_left = plt.subplot(*subplot_layout, 7)
        if self.strategy_resources:
            df_cpu_peak = df_strategy_res.set_index("strategy")[
                "cpu_percent_max"
            ].sort_values(ascending=False)
        else:
            # Fallback to old method if strategy_resources not available
            df_cpu_peak = (
                df.groupby("strategy")["cpu_percent_max"]
                .mean()
                .sort_values(ascending=False)
            )

        # Create stacked bar chart with idle baseline (using max idle for peak comparison)
        idle_cpu_peak = []
        active_cpu_peak = []
        for strategy in df_cpu_peak.index:
            db_type = strategy_to_db.get(strategy)
            idle_val = 0.0
            if db_type and db_type in self.idle_resources:
                idle_val = self.idle_resources[db_type][
                    "cpu_percent_max"
                ]  # Use max idle for peak
            idle_cpu_peak.append(idle_val)
            active_cpu_peak.append(max(0, df_cpu_peak[strategy] - idle_val))

        ax7_left.barh(
            df_cpu_peak.index, idle_cpu_peak, color="lightgray", label="Idle Peak"
        )
        ax7_left.barh(
            df_cpu_peak.index,
            active_cpu_peak,
            left=idle_cpu_peak,
            color="darkseagreen",
            label="Active Peak Increase",
        )

        ax7_left.set_title("Peak CPU Usage by Strategy", fontsize=14, fontweight="bold")
        ax7_left.set_xlabel("CPU (%)", fontsize=12)
        ax7_left.set_ylabel("Strategy", fontsize=12)
        ax7_left.grid(axis="x", alpha=0.3)
        ax7_left.legend(loc="lower right", fontsize=9)

        # Add value labels on bars (total)
        for i, (strategy, total) in enumerate(df_cpu_peak.items()):
            ax7_left.text(
                total,
                i,
                f"{total:.1f}%",
                ha="left",
                va="center",
                fontsize=9,
            )

        # 8. Peak Memory Usage by Strategy (use strategy-level aggregates with idle baseline)
        ax7_right = plt.subplot(*subplot_layout, 8)
        if self.strategy_resources:
            df_memory_peak = df_strategy_res.set_index("strategy")[
                "memory_mb_max"
            ].sort_values(ascending=False)
        else:
            # Fallback to old method if strategy_resources not available
            df_memory_peak = (
                df.groupby("strategy")["memory_mb_max"]
                .mean()
                .sort_values(ascending=False)
            )

        # Create stacked bar chart with idle baseline (using max idle for peak comparison)
        idle_mem_peak = []
        active_mem_peak = []
        for strategy in df_memory_peak.index:
            db_type = strategy_to_db.get(strategy)
            idle_val = 0.0
            if db_type and db_type in self.idle_resources:
                idle_val = self.idle_resources[db_type][
                    "memory_mb_max"
                ]  # Use max idle for peak
            idle_mem_peak.append(idle_val)
            active_mem_peak.append(max(0, df_memory_peak[strategy] - idle_val))

        ax7_right.barh(
            df_memory_peak.index, idle_mem_peak, color="lightgray", label="Idle Peak"
        )
        ax7_right.barh(
            df_memory_peak.index,
            active_mem_peak,
            left=idle_mem_peak,
            color="rebeccapurple",
            label="Active Peak Increase",
        )

        ax7_right.set_title(
            "Peak Memory Usage by Strategy", fontsize=14, fontweight="bold"
        )
        ax7_right.set_xlabel("Memory (MB)", fontsize=12)
        ax7_right.set_ylabel("Strategy", fontsize=12)
        ax7_right.grid(axis="x", alpha=0.3)
        ax7_right.legend(loc="lower right", fontsize=9)

        # Add value labels on bars (total)
        for i, (strategy, total) in enumerate(df_memory_peak.items()):
            ax7_right.text(
                total,
                i,
                f"{total:.1f}",
                ha="left",
                va="center",
                fontsize=9,
            )

        # 9. Storage comparison (if available)
        if has_storage:
            ax9 = plt.subplot(*subplot_layout, 9)
            df_storage = pd.DataFrame(self.storage_stats)

            # Aggregate by strategy
            storage_by_strategy = df_storage.groupby("strategy").agg(
                {"total_bytes": "sum", "table_bytes": "sum", "index_bytes": "sum"}
            )

            # Convert to MB for easier reading
            storage_mb = storage_by_strategy / (1024 * 1024)
            storage_mb.plot(kind="bar", stacked=False, ax=ax9)
            ax9.set_title(
                "Storage Requirements by Strategy", fontsize=14, fontweight="bold"
            )
            ax9.set_xlabel("Strategy", fontsize=12)
            ax9.set_ylabel("Size (MB)", fontsize=12)
            ax9.legend(["Total", "Table Data", "Indexes"], fontsize=9)
            plt.setp(ax9.xaxis.get_majorticklabels(), rotation=45, ha="right")

            # 10. Storage breakdown pie chart
            ax10 = plt.subplot(*subplot_layout, 10)
            total_by_strategy = df_storage.groupby("strategy")["total_bytes"].sum()
            colors = plt.cm.Set3(range(len(total_by_strategy)))
            wedges, texts, autotexts = ax10.pie(
                total_by_strategy.values,
                labels=total_by_strategy.index,
                autopct="%1.1f%%",
                colors=colors,
                startangle=90,
            )
            ax10.set_title(
                "Storage Distribution by Strategy", fontsize=14, fontweight="bold"
            )
            for autotext in autotexts:
                autotext.set_color("white")
                autotext.set_fontsize(9)
                autotext.set_fontweight("bold")

        plt.tight_layout()

        # Save the figure
        output_file = f"results/{dataset_size}/benchmarks.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"\n📊 Visualizations saved to: {output_file}")

        # Create detailed storage breakdown chart showing individual components
        if has_storage and len(self.storage_stats) > 0:
            df_storage_detailed = pd.DataFrame(self.storage_stats)

            # Create a figure for detailed storage breakdown
            plt.figure(figsize=(16, 10))

            # Group by strategy and create stacked bar chart showing all components
            strategies = df_storage_detailed["strategy"].unique()

            for idx, strategy in enumerate(strategies):
                ax = plt.subplot(len(strategies), 1, idx + 1)

                # Filter data for this strategy
                strategy_data = df_storage_detailed[
                    df_storage_detailed["strategy"] == strategy
                ].copy()

                # Sort by total bytes descending
                strategy_data = strategy_data.sort_values(
                    "total_bytes", ascending=False
                )

                # Create separate bars for table and index components
                objects = strategy_data["object"].tolist()
                table_mb = (strategy_data["table_bytes"] / (1024 * 1024)).tolist()
                index_mb = (strategy_data["index_bytes"] / (1024 * 1024)).tolist()

                x = range(len(objects))
                width = 0.8

                # Create stacked bars
                ax.bar(x, table_mb, width, label="Table Data", color="steelblue")
                ax.bar(
                    x, index_mb, width, bottom=table_mb, label="Index", color="orange"
                )

                ax.set_title(
                    f"{strategy} - Storage Breakdown", fontsize=12, fontweight="bold"
                )
                ax.set_ylabel("Size (MB)", fontsize=10)
                ax.set_xticks(x)
                ax.set_xticklabels(objects, rotation=45, ha="right", fontsize=8)
                ax.legend(fontsize=9)
                ax.grid(axis="y", alpha=0.3)

                # Add value labels on bars
                for i, (table, index) in enumerate(zip(table_mb, index_mb)):
                    total = table + index
                    if total > 0:
                        ax.text(
                            i,
                            total,
                            f"{total:.1f}",
                            ha="center",
                            va="bottom",
                            fontsize=7,
                        )

            plt.tight_layout()
            output_file_storage = f"results/{dataset_size}/storage_detailed.png"
            plt.savefig(output_file_storage, dpi=300, bbox_inches="tight")
            print(f"📊 Detailed storage breakdown saved to: {output_file_storage}")

        # Also create individual query type comparison
        fig2, axes = plt.subplots(
            len(pivot.index), 1, figsize=(12, 4 * len(pivot.index))
        )
        if len(pivot.index) == 1:
            axes = [axes]

        for idx, query_type in enumerate(pivot.index):
            ax = axes[idx]
            data = pivot.loc[query_type].dropna().sort_values()
            colors = [
                "green"
                if val == data.min()
                else "red"
                if val == data.max()
                else "steelblue"
                for val in data.values
            ]
            data.plot(kind="barh", ax=ax, color=colors)
            ax.set_title(f"{query_type}", fontsize=12, fontweight="bold")
            ax.set_xlabel("Time (ms)", fontsize=10)
            ax.set_ylabel("")

            # Add value labels on bars
            for i, (strategy, value) in enumerate(data.items()):
                ax.text(value, i, f" {value:.2f}ms", va="center", fontsize=9)

        plt.tight_layout()
        output_file2 = f"results/{dataset_size}/benchmarks-by-query.png"
        plt.savefig(output_file2, dpi=300, bbox_inches="tight")
        print(f"📊 Query-specific visualizations saved to: {output_file2}")

        plt.close("all")
