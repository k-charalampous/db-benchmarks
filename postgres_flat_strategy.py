import time

import psycopg
from psycopg import sql

from strategy import Strategy
from timer import BenchmarkTimer


class FlatPostgreSQLStrategy(Strategy):
    """PostgreSQL with flattened tables and nested reconstruction"""

    def __init__(self, conn_string: str, table_name: str, page_size: int = 100):
        self.conn_string = conn_string
        self.conn = None
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

    def connect(self):
        self.conn = psycopg.connect(self.conn_string)
        self.conn.autocommit = False

    def close(self):
        if self.conn:
            self.conn.close()

    def setup(self, csv_path: str):
        """Setup single denormalized flat table from CSV file. Returns ingestion time in seconds if data was loaded, None otherwise."""
        with self.conn.cursor() as cur:
            # Check if table already exists
            cur.execute(
                """
                SELECT COUNT(*) FROM information_schema.tables
                WHERE table_name = %s
            """,
                (f"{self.table_name}",),
            )
            existing_tables = cur.fetchone()[0]

            if existing_tables == 1:
                # Table exists, check if it has data
                cur.execute(f"SELECT COUNT(*) FROM {self.table_name}")
                row_count = cur.fetchone()[0]
                if row_count > 0:
                    print(
                        f"    ✓ Table {self.table_name} already exists with {row_count:,} rows, skipping setup"
                    )
                    return None

            print(f"    Creating single denormalized table {self.table_name}...")
            # Start timing ingestion
            ingestion_start = time.perf_counter()
            cur.execute(
                sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(
                    sql.Identifier(f"{self.table_name}")
                )
            )

            # Create single denormalized table with ALL data
            cur.execute(
                sql.SQL("""
                CREATE TABLE {} (
                    order_id TEXT,
                    timestamp TEXT,
                    customer_id TEXT,
                    customer_name TEXT,
                    customer_email TEXT,
                    customer_tier TEXT,
                    customer_lifetime_value NUMERIC,
                    payment_method TEXT,
                    payment_status TEXT,
                    payment_amount NUMERIC,
                    payment_processor TEXT,
                    payment_fee NUMERIC,
                    shipping_status TEXT,
                    shipping_method TEXT,
                    shipping_cost NUMERIC,
                    shipping_city TEXT,
                    shipping_state TEXT,
                    shipping_country TEXT,
                    shipping_lat NUMERIC,
                    shipping_lon NUMERIC,
                    product_id TEXT,
                    product_name TEXT,
                    category_main TEXT,
                    category_sub TEXT,
                    price NUMERIC,
                    quantity INTEGER,
                    discount_applied BOOLEAN,
                    discount_percentage NUMERIC,
                    seller_id TEXT,
                    seller_name TEXT,
                    seller_rating_score NUMERIC,
                    seller_rating_count INTEGER
                )
            """).format(sql.Identifier(f"{self.table_name}"))
            )

            # Load data from CSV using COPY
            print(f"    Loading data from {csv_path}...")
            with open(csv_path, "r") as f:
                with cur.copy(
                    f"COPY {self.table_name} FROM STDIN WITH (FORMAT csv, HEADER true)",
                ) as copy:
                    for line in f:
                        copy.write(line)

            count = cur.execute(f"SELECT COUNT(*) FROM {self.table_name}").fetchone()[0]
            print(f"    ✓ Loaded {count:,} rows total")

            # Create indexes on commonly queried columns
            print("    Creating indexes...")
            cur.execute(
                sql.SQL("CREATE INDEX ON {} (customer_tier)").format(
                    sql.Identifier(f"{self.table_name}")
                )
            )
            cur.execute(
                sql.SQL("CREATE INDEX ON {} (payment_amount)").format(
                    sql.Identifier(f"{self.table_name}")
                )
            )
            cur.execute(
                sql.SQL("CREATE INDEX ON {} (product_id)").format(
                    sql.Identifier(f"{self.table_name}")
                )
            )
            cur.execute(
                sql.SQL("CREATE INDEX ON {} (order_id)").format(
                    sql.Identifier(f"{self.table_name}")
                )
            )

            self.conn.commit()

            ingestion_time = time.perf_counter() - ingestion_start
            print(f"    ✓ Ingestion completed in {ingestion_time:.2f}s")

            return ingestion_time

    def execute_query(self, query_type: str):
        query = self.queries.get(query_type)
        with self.conn.cursor() as cur:
            with BenchmarkTimer(query_type) as timer:
                cur.execute(query)
                results = cur.fetchall()
        return timer.elapsed, len(results)
