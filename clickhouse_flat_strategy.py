import time

import clickhouse_connect

from postgres_flat_strategy import FlatPostgreSQLStrategy
from timer import BenchmarkTimer


class ClickhouseStrategy(FlatPostgreSQLStrategy):
    """PostgreSQL with flattened tables and nested reconstruction"""

    def __init__(self, table_name: str, page_size: int = 100):
        self.client = clickhouse_connect.get_client(
            host="localhost",
            port=8123,
            username="benchmark_user",
            password="benchmark_pass",
            database="benchmark_db",
        )
        self.table_name = table_name
        self.queries = {
            "simple_where": f"""
                SELECT *
                FROM {table_name}
                WHERE customer_tier = 'gold'
                LIMIT {page_size}
            """,
            "complex_where": f"""
                SELECT *
                FROM {table_name}
                WHERE payment_amount > 100
                  AND shipping_status = 'delivered'
                  AND customer_tier IN ('gold', 'platinum')
                  AND payment_status = 'completed'
                LIMIT {page_size}
            """,
            "pagination_early": f"""
                SELECT *
                FROM {table_name}
                WHERE customer_tier = 'gold'
                ORDER BY order_id
                LIMIT {page_size} OFFSET 100
            """,
            "pagination_deep": f"""
                SELECT *
                FROM {table_name}
                WHERE customer_tier = 'gold'
                ORDER BY order_id
                LIMIT {page_size} OFFSET 10000
            """,
            "nested_array_filter": f"""
                SELECT *
                FROM {table_name}
                WHERE product_id LIKE 'PROD-0%'
                LIMIT {page_size}
            """,
            # Aggregation queries - these benefit most from pg_duckdb acceleration
            "simple_nested_agg": f"""
                SELECT
                    customer_tier,
                    COUNT(DISTINCT order_id) as order_count,
                    AVG(payment_amount) as avg_amount
                FROM {table_name}
                GROUP BY customer_tier
            """,
            "deep_nested_agg": f"""
                SELECT
                    category_main,
                    COUNT(DISTINCT order_id) as count,
                    AVG(shipping_lat) as avg_lat
                FROM {table_name}
                WHERE category_main IS NOT NULL
                GROUP BY category_main
            """,
            "array_aggregation": f"""
                SELECT
                    product_id,
                    COUNT(*) as times_ordered,
                    SUM(quantity) as total_quantity,
                    AVG(price) as avg_price
                FROM {table_name}
                GROUP BY product_id
                ORDER BY times_ordered DESC
                LIMIT 100
            """,
            "complex_where_agg": f"""
                SELECT
                    customer_tier,
                    payment_status,
                    COUNT(DISTINCT order_id) as count,
                    SUM(payment_amount) as total_amount
                FROM {table_name}
                WHERE payment_amount > 100
                  AND shipping_status = 'delivered'
                  AND customer_tier IN ('gold', 'platinum')
                GROUP BY customer_tier, payment_status
            """,
            "seller_rating_agg": f"""
                SELECT
                    seller_name as seller,
                    AVG(seller_rating_score) as avg_rating,
                    COUNT(*) as product_count,
                    SUM(quantity) as total_sold
                FROM {table_name}
                GROUP BY seller_name
                HAVING COUNT(*) > 5
                ORDER BY avg_rating DESC
                LIMIT 50
            """,
        }

    def close(self):
        self.client.close()

    def setup(self, csv_path: str):
        """Setup single denormalized ClickHouse table from CSV file. Returns ingestion time in seconds if data was loaded, None otherwise."""
        # Check if table exists
        result = self.client.query(f"""
            SELECT count() FROM system.tables
            WHERE database = 'benchmark_db'
            AND name = '{self.table_name}'
        """)
        existing_tables = result.result_rows[0][0]

        if existing_tables == 1:
            # Table exists, check if it has data
            result = self.client.query(f"SELECT count() FROM {self.table_name}")
            row_count = result.result_rows[0][0]
            if row_count > 0:
                print(
                    f"    ✓ ClickHouse table {self.table_name} already exists with {row_count:,} rows, skipping setup"
                )
                return None

        print(f"    Creating single denormalized ClickHouse table {self.table_name}...")
        # Start timing ingestion
        ingestion_start = time.perf_counter()
        self.client.command(f"DROP TABLE IF EXISTS {self.table_name}")

        # Create single denormalized table with ALL data
        self.client.command(f"""
            CREATE TABLE {self.table_name} (
                order_id String,
                timestamp String,
                customer_id String,
                customer_name String,
                customer_email String,
                customer_tier String,
                customer_lifetime_value Float64,
                payment_method String,
                payment_status String,
                payment_amount Float64,
                payment_processor String,
                payment_fee Float64,
                shipping_status String,
                shipping_method String,
                shipping_cost Float64,
                shipping_city String,
                shipping_state String,
                shipping_country String,
                shipping_lat Float64,
                shipping_lon Float64,
                product_id String,
                product_name String,
                category_main String,
                category_sub String,
                price Float64,
                quantity UInt32,
                discount_applied UInt8,
                discount_percentage Float64,
                seller_id String,
                seller_name String,
                seller_rating_score Float64,
                seller_rating_count UInt32
            ) ENGINE = MergeTree()
            ORDER BY (customer_tier, order_id, product_id)
        """)

        # Load data from CSV in batches
        print(f"    Loading data from {csv_path}...")
        import pandas as pd

        batch_size = 10000
        total_rows = 0

        for chunk in pd.read_csv(csv_path, chunksize=batch_size):
            # Replace NaN values with appropriate defaults for ClickHouse
            # For string columns, use empty string; for numeric, keep NaN (ClickHouse handles it)
            string_columns = [
                "order_id",
                "timestamp",
                "customer_id",
                "customer_name",
                "customer_email",
                "customer_tier",
                "payment_method",
                "payment_status",
                "payment_processor",
                "shipping_status",
                "shipping_method",
                "shipping_city",
                "shipping_state",
                "shipping_country",
                "product_id",
                "product_name",
                "category_main",
                "category_sub",
                "seller_id",
                "seller_name",
            ]
            for col in string_columns:
                if col in chunk.columns:
                    chunk[col] = chunk[col].fillna("")

            self.client.insert_df(f"{self.table_name}", chunk)
            total_rows += len(chunk)
            if total_rows % 50000 == 0:
                print(f"      Loaded {total_rows:,} rows so far...")

        print(f"    ✓ Loaded {total_rows:,} rows total")

        ingestion_time = time.perf_counter() - ingestion_start
        print(f"    ✓ Ingestion completed in {ingestion_time:.2f}s")

        return ingestion_time

    def execute_query(self, query_type: str):
        query = self.queries.get(query_type)
        with BenchmarkTimer(query_type) as timer:
            result = self.client.query(query)
            results = result.result_rows
        return timer.elapsed, len(results)
