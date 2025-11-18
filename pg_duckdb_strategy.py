import psycopg

from strategy import Strategy
from timer import BenchmarkTimer


class PgDuckDBStrategy(Strategy):
    """PostgreSQL with pg_duckdb extension using flat tables for accelerated analytical queries"""

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
        """Use existing flat table from FlatPostgreSQLStrategy and enable pg_duckdb"""
        with self.conn.cursor() as cur:
            # Check if pg_duckdb extension is available
            try:
                cur.execute("CREATE EXTENSION IF NOT EXISTS pg_duckdb")
                self.conn.commit()
                print("    ✓ pg_duckdb extension enabled")
            except Exception as e:
                print(f"    Warning: Could not enable pg_duckdb extension: {e}")
                print("    Continuing with standard PostgreSQL...")

            # Check if the flat table already exists (from FlatPostgreSQLStrategy)
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
                        f"    ✓ Using existing flat table {self.table_name} with {row_count:,} rows"
                    )
                    return

            # Note: If table doesn't exist, it should be created by FlatPostgreSQLStrategy first
            # This strategy just uses the existing table with pg_duckdb acceleration
            print(
                f"    Warning: Flat table {self.table_name} does not exist. "
                f"It should be created by FlatPostgreSQLStrategy first."
            )

    def execute_query(self, query_type: str):
        query = self.queries.get(query_type)
        with self.conn.cursor() as cur:
            with BenchmarkTimer(query_type) as timer:
                cur.execute(query)
                results = cur.fetchall()
        return timer.elapsed, len(results)
