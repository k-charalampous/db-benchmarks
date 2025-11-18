import json
import time

import clickhouse_connect

from strategy import Strategy
from timer import BenchmarkTimer


class ClickHouseNestedStrategy(Strategy):
    """ClickHouse with nested/JSON data structures using Tuple and Array types"""

    def __init__(self, table_name: str, page_size: int = 100):
        self.client = clickhouse_connect.get_client(
            host="localhost",
            port=8123,
            username="benchmark_user",
            password="benchmark_pass",
            database="benchmark_db",
        )
        self.table_name = table_name
        self.page_size = page_size

        # Queries using nested field access with dot notation
        self.queries = {
            "simple_where": f"""
                SELECT *
                FROM {table_name}
                WHERE customer.tier = 'gold'
                LIMIT {page_size}
            """,
            "complex_where": f"""
                SELECT *
                FROM {table_name}
                WHERE payment.amount > 100
                  AND shipping.status = 'delivered'
                  AND customer.tier IN ('gold', 'platinum')
                  AND payment.status = 'completed'
                LIMIT {page_size}
            """,
            "pagination_early": f"""
                SELECT *
                FROM {table_name}
                WHERE customer.tier = 'gold'
                ORDER BY order_id
                LIMIT {page_size} OFFSET 100
            """,
            "pagination_deep": f"""
                SELECT *
                FROM {table_name}
                WHERE customer.tier = 'gold'
                ORDER BY order_id
                LIMIT {page_size} OFFSET 10000
            """,
            "nested_array_filter": f"""
                SELECT *
                FROM {table_name}
                ARRAY JOIN items
                WHERE items.product_id LIKE 'PROD-0%'
                LIMIT {page_size}
            """,
            # Aggregation queries using nested fields
            "simple_nested_agg": f"""
                SELECT
                    customer.tier as customer_tier,
                    COUNT(DISTINCT order_id) as order_count,
                    AVG(payment.amount) as avg_amount
                FROM {table_name}
                GROUP BY customer.tier
            """,
            "deep_nested_agg": f"""
                SELECT
                    items.category.main as category_main,
                    COUNT(DISTINCT order_id) as count,
                    AVG(shipping.address.coordinates.lat) as avg_lat
                FROM {table_name}
                ARRAY JOIN items
                WHERE items.category.main != ''
                GROUP BY items.category.main
            """,
            "array_aggregation": f"""
                SELECT
                    items.product_id as product_id,
                    COUNT(*) as times_ordered,
                    SUM(items.quantity) as total_quantity,
                    AVG(items.price) as avg_price
                FROM {table_name}
                ARRAY JOIN items
                GROUP BY items.product_id
                ORDER BY times_ordered DESC
                LIMIT 100
            """,
            "complex_where_agg": f"""
                SELECT
                    customer.tier as customer_tier,
                    payment.status as payment_status,
                    COUNT(DISTINCT order_id) as count,
                    SUM(payment.amount) as total_amount
                FROM {table_name}
                WHERE payment.amount > 100
                  AND shipping.status = 'delivered'
                  AND customer.tier IN ('gold', 'platinum')
                GROUP BY customer.tier, payment.status
            """,
            "seller_rating_agg": f"""
                SELECT
                    items.seller.name as seller,
                    AVG(items.seller.rating.score) as avg_rating,
                    COUNT(*) as product_count,
                    SUM(items.quantity) as total_sold
                FROM {table_name}
                ARRAY JOIN items
                GROUP BY items.seller.name
                HAVING COUNT(*) > 5
                ORDER BY avg_rating DESC
                LIMIT 50
            """,
        }

    def close(self):
        self.client.close()

    def setup(self, jsonl_path: str):
        """Setup ClickHouse table with nested structures from JSONL file. Returns ingestion time in seconds if data was loaded, None otherwise."""
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
                    f"    ✓ ClickHouse nested table {self.table_name} already exists with {row_count:,} rows, skipping setup"
                )
                return None

        print(f"    Creating ClickHouse table with nested structures {self.table_name}...")
        # Start timing ingestion
        ingestion_start = time.perf_counter()
        self.client.command(f"DROP TABLE IF EXISTS {self.table_name}")

        # Create table with nested/tuple types
        # ClickHouse uses Tuple for nested objects and Array(Tuple(...)) for arrays of objects
        self.client.command(f"""
            CREATE TABLE {self.table_name} (
                order_id String,
                timestamp String,
                customer Tuple(
                    id String,
                    name String,
                    email String,
                    tier String,
                    lifetime_value Float64
                ),
                payment Tuple(
                    method String,
                    status String,
                    amount Float64,
                    processor Tuple(
                        name String,
                        fee Float64
                    )
                ),
                shipping Tuple(
                    status String,
                    method String,
                    cost Float64,
                    address Tuple(
                        city String,
                        state String,
                        country String,
                        coordinates Tuple(
                            lat Float64,
                            lon Float64
                        )
                    )
                ),
                items Array(Tuple(
                    product_id String,
                    name String,
                    category Tuple(
                        main String,
                        sub String
                    ),
                    price Float64,
                    quantity UInt32,
                    discount Tuple(
                        applied UInt8,
                        percentage Float64
                    ),
                    seller Tuple(
                        id String,
                        name String,
                        rating Tuple(
                            score Float64,
                            count UInt32
                        )
                    )
                ))
            ) ENGINE = MergeTree()
            ORDER BY (customer.tier, order_id)
        """)

        # Load data from JSONL file
        print(f"    Loading nested data from {jsonl_path}...")

        batch_size = 10000
        total_rows = 0
        batch = []

        with open(jsonl_path, 'r') as f:
            for line in f:
                record = json.loads(line.strip())

                # Transform nested JSON to ClickHouse tuple format
                transformed = self._transform_record(record)
                batch.append(transformed)

                if len(batch) >= batch_size:
                    self._insert_batch(batch)
                    total_rows += len(batch)
                    if total_rows % 50000 == 0:
                        print(f"      Loaded {total_rows:,} rows so far...")
                    batch = []

            # Insert remaining records
            if batch:
                self._insert_batch(batch)
                total_rows += len(batch)

        print(f"    ✓ Loaded {total_rows:,} rows total")

        ingestion_time = time.perf_counter() - ingestion_start
        print(f"    ✓ Ingestion completed in {ingestion_time:.2f}s")

        return ingestion_time

    def _transform_record(self, record):
        """Transform JSON record to ClickHouse tuple format"""
        # ClickHouse expects tuples as Python tuples
        return {
            'order_id': record['order_id'],
            'timestamp': record['timestamp'],
            'customer': (
                record['customer']['id'],
                record['customer']['name'],
                record['customer']['email'],
                record['customer']['tier'],
                record['customer']['lifetime_value']
            ),
            'payment': (
                record['payment']['method'],
                record['payment']['status'],
                record['payment']['amount'],
                (
                    record['payment']['processor']['name'],
                    record['payment']['processor']['fee']
                )
            ),
            'shipping': (
                record['shipping']['status'],
                record['shipping']['method'],
                record['shipping']['cost'],
                (
                    record['shipping']['address']['city'],
                    record['shipping']['address']['state'],
                    record['shipping']['address']['country'],
                    (
                        record['shipping']['address']['coordinates']['lat'],
                        record['shipping']['address']['coordinates']['lon']
                    )
                )
            ),
            'items': [
                (
                    item['product_id'],
                    item['name'],
                    (
                        item['category']['main'],
                        item['category']['sub']
                    ),
                    item['price'],
                    item['quantity'],
                    (
                        1 if item['discount']['applied'] else 0,
                        item['discount']['percentage']
                    ),
                    (
                        item['seller']['id'],
                        item['seller']['name'],
                        (
                            item['seller']['rating']['score'],
                            item['seller']['rating']['count']
                        )
                    )
                )
                for item in record['items']
            ]
        }

    def _insert_batch(self, batch):
        """Insert a batch of records into ClickHouse"""
        # Prepare data for insertion
        data = []
        for record in batch:
            data.append([
                record['order_id'],
                record['timestamp'],
                record['customer'],
                record['payment'],
                record['shipping'],
                record['items']
            ])

        self.client.insert(
            self.table_name,
            data,
            column_names=['order_id', 'timestamp', 'customer', 'payment', 'shipping', 'items']
        )

    def execute_query(self, query_type: str):
        query = self.queries.get(query_type)
        with BenchmarkTimer(query_type) as timer:
            result = self.client.query(query)
            results = result.result_rows
        return timer.elapsed, len(results)
