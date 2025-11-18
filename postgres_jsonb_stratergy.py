import json
import time

import psycopg
from psycopg import sql

from strategy import Strategy
from timer import BenchmarkTimer


class NativePostgreSQLStrategy(Strategy):
    """JSONB PostgreSQL with JSONB queries"""

    def __init__(self, conn_string: str, table_name: str, page_size: int = 100):
        self.conn_string = conn_string
        self.conn = None
        self.table_name = table_name

        self.queries = {
            "simple_where": f"""
                SELECT data
                FROM {table_name}
                WHERE data->'customer'->>'tier' = 'gold'
                LIMIT {page_size}
            """,
            "complex_where": f"""
                SELECT data
                FROM {table_name}
                WHERE (data->'payment'->>'amount')::numeric > 100
                  AND data->'shipping'->>'status' = 'delivered'
                  AND data->'customer'->>'tier' IN ('gold', 'platinum')
                  AND data->'payment'->>'status' = 'completed'
                LIMIT {page_size}
            """,
            "pagination_early": f"""
                SELECT data
                FROM {table_name}
                WHERE data->'customer'->>'tier' = 'gold'
                ORDER BY id
                LIMIT {page_size} OFFSET {100}
            """,
            "pagination_deep": f"""
                SELECT data
                FROM {table_name}
                WHERE data->'customer'->>'tier' = 'gold'
                ORDER BY id
                LIMIT {page_size} OFFSET {100000}
            """,
            "nested_array_filter": f"""
                SELECT data
                FROM {table_name},
                jsonb_array_elements(data->'items') as item
                WHERE item->>'product_id' LIKE 'PROD-0%'
                LIMIT {page_size}
            """,
            # Aggregation queries
            "simple_nested_agg": f"""
                SELECT
                    data->'customer'->>'tier' as tier,
                    COUNT(*) as order_count,
                    AVG((data->'payment'->>'amount')::numeric) as avg_amount
                FROM {table_name}
                GROUP BY data->'customer'->>'tier'
            """,
            "deep_nested_agg": f"""
                SELECT
                    data->'items'->0->'category'->>'main' as category,
                    COUNT(*) as count,
                    AVG((data->'shipping'->'address'->'coordinates'->>'lat')::numeric) as avg_lat
                FROM {table_name}
                WHERE data->'items'->0->'category'->>'main' IS NOT NULL
                GROUP BY data->'items'->0->'category'->>'main'
            """,
            "array_aggregation": f"""
                SELECT
                    item->>'product_id' as product_id,
                    COUNT(*) as times_ordered,
                    SUM((item->>'quantity')::integer) as total_quantity,
                    AVG((item->>'price')::numeric) as avg_price
                FROM {table_name},
                jsonb_array_elements(data->'items') as item
                GROUP BY item->>'product_id'
                ORDER BY times_ordered DESC
                LIMIT 100
            """,
            "complex_where_agg": f"""
                SELECT
                    data->'customer'->>'tier' as tier,
                    data->'payment'->>'status' as payment_status,
                    COUNT(*) as count,
                    SUM((data->'payment'->>'amount')::numeric) as total_amount
                FROM {table_name}
                WHERE (data->'payment'->>'amount')::numeric > 100
                  AND data->'shipping'->>'status' = 'delivered'
                  AND data->'customer'->>'tier' IN ('gold', 'platinum')
                GROUP BY data->'customer'->>'tier', data->'payment'->>'status'
            """,
            "seller_rating_agg": f"""
                SELECT
                    item->'seller'->>'name' as seller,
                    AVG((item->'seller'->'rating'->>'score')::numeric) as avg_rating,
                    COUNT(*) as product_count,
                    SUM((item->>'quantity')::integer) as total_sold
                FROM {table_name},
                jsonb_array_elements(data->'items') as item
                GROUP BY item->'seller'->>'name'
                HAVING COUNT(*) > 5
                ORDER BY avg_rating DESC
                LIMIT 50
            """,
        }

    def connect(self):
        self.conn = psycopg.connect(self.conn_string)
        self.conn.autocommit = False

    def close(self):
        if self.conn:
            self.conn.close()

    def setup(self, jsonl_path: str):
        """Setup table and load data from JSONL file. Returns ingestion time in seconds if data was loaded, None otherwise."""
        with self.conn.cursor() as cur:
            # Check if table exists
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = %s
                )
            """,
                (self.table_name,),
            )
            table_exists = cur.fetchone()[0]

            if table_exists:
                print(f"    ✓ Table {self.table_name} already exists, skipping setup")
                return None

            print(f"    Creating table {self.table_name}...")

            # Start timing ingestion
            ingestion_start = time.perf_counter()

            cur.execute(
                sql.SQL("""
                CREATE TABLE {} (
                    id SERIAL PRIMARY KEY,
                    data JSONB NOT NULL
                )
            """).format(sql.Identifier(self.table_name))
            )

            # Create GIN index
            # cur.execute(
            #     sql.SQL(f"""
            #     CREATE INDEX idx_{table_name}_gin ON {table_name} USING GIN (data)
            # """)
            # )

            # Additional indexes for common access patterns
            cur.execute(
                sql.SQL(f"""
                CREATE INDEX idx_{self.table_name}_tier ON {self.table_name} ((data->'customer'->>'tier'))
            """)
            )

            cur.execute(
                sql.SQL(f"""
                CREATE INDEX idx_{self.table_name}_status ON {self.table_name} ((data->'payment'->>'status'))
            """)
            )

            # Bulk insert from file
            print(f"    Loading data from {jsonl_path}...")
            insert_query = sql.SQL("""
                INSERT INTO {} (data) VALUES (%s)
            """).format(sql.Identifier(self.table_name))

            batch = []
            batch_size = 1000
            count = 0

            with open(jsonl_path, "r") as f:
                for line in f:
                    record = json.loads(line.strip())
                    batch.append((json.dumps(record),))
                    count += 1

                    if len(batch) >= batch_size:
                        cur.executemany(insert_query, batch)
                        batch = []

                if batch:
                    cur.executemany(insert_query, batch)

            self.conn.commit()

            ingestion_time = time.perf_counter() - ingestion_start
            print(f"    ✓ Loaded {count:,} records in {ingestion_time:.2f}s")

            return ingestion_time

    def execute_query(self, query_type: str):
        query = self.queries.get(query_type)
        with self.conn.cursor() as cur:
            with BenchmarkTimer(query_type) as timer:
                cur.execute(query)
                results = cur.fetchall()
        return timer.elapsed, len(results)
