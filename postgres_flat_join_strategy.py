import time
from typing import Optional

import psycopg
from psycopg import sql

from strategy import Strategy
from timer import BenchmarkTimer


class FlatPostgreSQLJoinStrategy(Strategy):
    """PostgreSQL with multiple flat tables and joins for queries"""

    def __init__(self, conn_string: str, table_prefix: str, page_size: int = 100):
        self.conn_string = conn_string
        self.conn = None
        self.table_prefix = table_prefix

        # Define table names to match ClickHouse naming
        self.orders_table = f"{table_prefix}_orders"
        self.customers_table = f"{table_prefix}_customers"
        self.products_table = f"{table_prefix}_products"
        self.payments_table = f"{table_prefix}_payments"
        self.shipping_table = f"{table_prefix}_shipping"
        self.sellers_table = f"{table_prefix}_sellers"
        self.order_items_table = f"{table_prefix}_order_items"

        self.queries = {
            "simple_where": f"""
                SELECT o.*, c.*, p.*, pay.*, s.*, sel.*
                FROM {self.orders_table} o
                JOIN {self.customers_table} c ON o.customer_id = c.customer_id
                JOIN {self.order_items_table} oi ON o.order_id = oi.order_id
                JOIN {self.products_table} p ON oi.product_id = p.product_id
                JOIN {self.payments_table} pay ON o.order_id = pay.order_id
                JOIN {self.shipping_table} s ON o.order_id = s.order_id
                JOIN {self.sellers_table} sel ON p.seller_id = sel.seller_id
                WHERE c.customer_tier = 'gold'
                LIMIT {page_size}
            """,
            "complex_where": f"""
                SELECT o.*, c.*, p.*, pay.*, s.*, sel.*
                FROM {self.orders_table} o
                JOIN {self.customers_table} c ON o.customer_id = c.customer_id
                JOIN {self.order_items_table} oi ON o.order_id = oi.order_id
                JOIN {self.products_table} p ON oi.product_id = p.product_id
                JOIN {self.payments_table} pay ON o.order_id = pay.order_id
                JOIN {self.shipping_table} s ON o.order_id = s.order_id
                JOIN {self.sellers_table} sel ON p.seller_id = sel.seller_id
                WHERE pay.payment_amount > 100
                  AND s.shipping_status = 'delivered'
                  AND c.customer_tier IN ('gold', 'platinum')
                  AND pay.payment_status = 'completed'
                LIMIT {page_size}
            """,
            "pagination_early": f"""
                SELECT o.*, c.*, p.*, pay.*, s.*, sel.*
                FROM {self.orders_table} o
                JOIN {self.customers_table} c ON o.customer_id = c.customer_id
                JOIN {self.order_items_table} oi ON o.order_id = oi.order_id
                JOIN {self.products_table} p ON oi.product_id = p.product_id
                JOIN {self.payments_table} pay ON o.order_id = pay.order_id
                JOIN {self.shipping_table} s ON o.order_id = s.order_id
                JOIN {self.sellers_table} sel ON p.seller_id = sel.seller_id
                WHERE c.customer_tier = 'gold'
                ORDER BY o.order_id
                LIMIT {page_size} OFFSET 100
            """,
            "pagination_deep": f"""
                SELECT o.*, c.*, p.*, pay.*, s.*, sel.*
                FROM {self.orders_table} o
                JOIN {self.customers_table} c ON o.customer_id = c.customer_id
                JOIN {self.order_items_table} oi ON o.order_id = oi.order_id
                JOIN {self.products_table} p ON oi.product_id = p.product_id
                JOIN {self.payments_table} pay ON o.order_id = pay.order_id
                JOIN {self.shipping_table} s ON o.order_id = s.order_id
                JOIN {self.sellers_table} sel ON p.seller_id = sel.seller_id
                WHERE c.customer_tier = 'gold'
                ORDER BY o.order_id
                LIMIT {page_size} OFFSET 10000
            """,
            "nested_array_filter": f"""
                SELECT o.*, c.*, p.*, pay.*, s.*, sel.*
                FROM {self.orders_table} o
                JOIN {self.customers_table} c ON o.customer_id = c.customer_id
                JOIN {self.order_items_table} oi ON o.order_id = oi.order_id
                JOIN {self.products_table} p ON oi.product_id = p.product_id
                JOIN {self.payments_table} pay ON o.order_id = pay.order_id
                JOIN {self.shipping_table} s ON o.order_id = s.order_id
                JOIN {self.sellers_table} sel ON p.seller_id = sel.seller_id
                WHERE p.product_id LIKE 'PROD-0%%'
                LIMIT {page_size}
            """,
            "simple_nested_agg": f"""
                SELECT
                    c.customer_tier,
                    COUNT(DISTINCT o.order_id) as order_count,
                    AVG(pay.payment_amount) as avg_amount
                FROM {self.orders_table} o
                JOIN {self.customers_table} c ON o.customer_id = c.customer_id
                JOIN {self.payments_table} pay ON o.order_id = pay.order_id
                GROUP BY c.customer_tier
            """,
            "deep_nested_agg": f"""
                SELECT
                    p.category_main,
                    COUNT(DISTINCT o.order_id) as count,
                    AVG(s.shipping_lat) as avg_lat
                FROM {self.orders_table} o
                JOIN {self.order_items_table} oi ON o.order_id = oi.order_id
                JOIN {self.products_table} p ON oi.product_id = p.product_id
                JOIN {self.shipping_table} s ON o.order_id = s.order_id
                WHERE p.category_main IS NOT NULL
                GROUP BY p.category_main
            """,
            "array_aggregation": f"""
                SELECT
                    p.product_id,
                    COUNT(*) as times_ordered,
                    SUM(oi.quantity) as total_quantity,
                    AVG(p.price) as avg_price
                FROM {self.order_items_table} oi
                JOIN {self.products_table} p ON oi.product_id = p.product_id
                GROUP BY p.product_id
                ORDER BY times_ordered DESC
                LIMIT 100
            """,
            "complex_where_agg": f"""
                SELECT
                    c.customer_tier,
                    pay.payment_status,
                    COUNT(DISTINCT o.order_id) as count,
                    SUM(pay.payment_amount) as total_amount
                FROM {self.orders_table} o
                JOIN {self.customers_table} c ON o.customer_id = c.customer_id
                JOIN {self.payments_table} pay ON o.order_id = pay.order_id
                JOIN {self.shipping_table} s ON o.order_id = s.order_id
                WHERE pay.payment_amount > 100
                  AND s.shipping_status = 'delivered'
                  AND c.customer_tier IN ('gold', 'platinum')
                GROUP BY c.customer_tier, pay.payment_status
            """,
            "seller_rating_agg": f"""
                SELECT
                    sel.seller_name as seller,
                    AVG(sel.seller_rating_score) as avg_rating,
                    COUNT(*) as product_count,
                    SUM(oi.quantity) as total_sold
                FROM {self.order_items_table} oi
                JOIN {self.products_table} p ON oi.product_id = p.product_id
                JOIN {self.sellers_table} sel ON p.seller_id = sel.seller_id
                GROUP BY sel.seller_name, sel.seller_rating_score
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

    def _create_table_schemas(self, cur):
        """Create the table schemas for all required tables"""
        # Orders table - core order information
        cur.execute(
            sql.SQL("""
            CREATE TABLE {} (
                order_id TEXT PRIMARY KEY,
                timestamp TEXT,
                customer_id TEXT
            )
        """).format(sql.Identifier(f"{self.table_prefix}_orders"))
        )

        # Customers table
        cur.execute(
            sql.SQL("""
            CREATE TABLE {} (
                customer_id TEXT PRIMARY KEY,
                customer_name TEXT,
                customer_email TEXT,
                customer_tier TEXT,
                customer_lifetime_value NUMERIC
            )
        """).format(sql.Identifier(f"{self.table_prefix}_customers"))
        )

        # Payments table
        cur.execute(
            sql.SQL("""
            CREATE TABLE {} (
                order_id TEXT PRIMARY KEY,
                payment_method TEXT,
                payment_status TEXT,
                payment_amount NUMERIC,
                payment_processor TEXT,
                payment_fee NUMERIC
            )
        """).format(sql.Identifier(f"{self.table_prefix}_payments"))
        )

        # Shipping table
        cur.execute(
            sql.SQL("""
            CREATE TABLE {} (
                order_id TEXT PRIMARY KEY,
                shipping_status TEXT,
                shipping_method TEXT,
                shipping_cost NUMERIC,
                shipping_city TEXT,
                shipping_state TEXT,
                shipping_country TEXT,
                shipping_lat NUMERIC,
                shipping_lon NUMERIC
            )
        """).format(sql.Identifier(f"{self.table_prefix}_shipping"))
        )

        # Products table
        cur.execute(
            sql.SQL("""
            CREATE TABLE {} (
                product_id TEXT PRIMARY KEY,
                product_name TEXT,
                category_main TEXT,
                category_sub TEXT,
                price NUMERIC,
                seller_id TEXT
            )
        """).format(sql.Identifier(f"{self.table_prefix}_products"))
        )

        # Order-Products junction table
        cur.execute(
            sql.SQL("""
            CREATE TABLE {} (
                order_id TEXT,
                product_id TEXT,
                quantity INTEGER,
                discount_applied BOOLEAN,
                discount_percentage NUMERIC,
                PRIMARY KEY (order_id, product_id)
            )
        """).format(sql.Identifier(f"{self.table_prefix}_order_items"))
        )

        # Sellers table
        cur.execute(
            sql.SQL("""
            CREATE TABLE {} (
                seller_id TEXT PRIMARY KEY,
                seller_name TEXT,
                seller_rating_score NUMERIC,
                seller_rating_count INTEGER
            )
        """).format(sql.Identifier(f"{self.table_prefix}_sellers"))
        )

    def _create_indexes(self, cur):
        """Create indexes on commonly queried columns"""
        # Orders indexes
        cur.execute(
            sql.SQL("CREATE INDEX ON {} (customer_id)").format(
                sql.Identifier(f"{self.table_prefix}_orders")
            )
        )

        # Customers indexes
        cur.execute(
            sql.SQL("CREATE INDEX ON {} (customer_tier)").format(
                sql.Identifier(f"{self.table_prefix}_customers")
            )
        )

        # Payments indexes
        cur.execute(
            sql.SQL("CREATE INDEX ON {} (payment_amount)").format(
                sql.Identifier(f"{self.table_prefix}_payments")
            )
        )
        cur.execute(
            sql.SQL("CREATE INDEX ON {} (payment_status)").format(
                sql.Identifier(f"{self.table_prefix}_payments")
            )
        )

        # Shipping indexes
        cur.execute(
            sql.SQL("CREATE INDEX ON {} (shipping_status)").format(
                sql.Identifier(f"{self.table_prefix}_shipping")
            )
        )

        # Products indexes
        cur.execute(
            sql.SQL("CREATE INDEX ON {} (category_main)").format(
                sql.Identifier(f"{self.table_prefix}_products")
            )
        )
        cur.execute(
            sql.SQL("CREATE INDEX ON {} (seller_id)").format(
                sql.Identifier(f"{self.table_prefix}_products")
            )
        )

        # Order-Products indexes
        cur.execute(
            sql.SQL("CREATE INDEX ON {} (product_id)").format(
                sql.Identifier(f"{self.table_prefix}_order_items")
            )
        )

    def setup(self, csv_path: str) -> Optional[float]:
        """Setup multiple flat tables from CSV file with proper relationships"""
        with self.conn.cursor() as cur:
            # Check if tables already exist and have data
            tables = [
                f"{self.table_prefix}_orders",
                f"{self.table_prefix}_customers",
                f"{self.table_prefix}_payments",
                f"{self.table_prefix}_shipping",
                f"{self.table_prefix}_products",
                f"{self.table_prefix}_order_items",
                f"{self.table_prefix}_sellers",
            ]

            tables_exist = True
            for table in tables:
                cur.execute(
                    """
                    SELECT COUNT(*) FROM information_schema.tables
                    WHERE table_name = %s
                    """,
                    (table,),
                )
                if cur.fetchone()[0] == 0:
                    tables_exist = False
                    break

            if tables_exist:
                # Check if tables have data
                cur.execute(f"SELECT COUNT(*) FROM {self.table_prefix}_orders")
                row_count = cur.fetchone()[0]
                if row_count > 0:
                    print(
                        f"    ✓ Tables with prefix {self.table_prefix} already exist with {row_count:,} orders, skipping setup"
                    )
                    return None

            print(f"    Creating normalized tables with prefix {self.table_prefix}...")
            # Drop existing tables if they exist
            for table in reversed(tables):
                cur.execute(
                    sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(
                        sql.Identifier(table)
                    )
                )

            # Create table schemas
            self._create_table_schemas(cur)

            # Start timing ingestion
            ingestion_start = time.perf_counter()

            # Create temporary staging table to load CSV
            cur.execute(
                sql.SQL("""
                CREATE TEMPORARY TABLE staging (
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
            """)
            )

            # Load data from CSV into staging
            print(f"    Loading data from {csv_path}...")
            with open(csv_path, "r") as f:
                with cur.copy(
                    "COPY staging FROM STDIN WITH (FORMAT csv, HEADER true)"
                ) as copy:
                    for line in f:
                        copy.write(line)

            # Insert data into normalized tables
            print("    Distributing data into normalized tables...")

            # Insert into customers (distinct customers only - deduplicate by customer_id)
            cur.execute(
                sql.SQL("""
                INSERT INTO {} (
                    customer_id, customer_name, customer_email,
                    customer_tier, customer_lifetime_value
                )
                SELECT DISTINCT ON (customer_id)
                    customer_id, customer_name, customer_email,
                    customer_tier, customer_lifetime_value
                FROM staging
                ORDER BY customer_id
            """).format(sql.Identifier(f"{self.table_prefix}_customers"))
            )

            # Insert into sellers (distinct sellers only - deduplicate by seller_id)
            cur.execute(
                sql.SQL("""
                INSERT INTO {} (
                    seller_id, seller_name,
                    seller_rating_score, seller_rating_count
                )
                SELECT DISTINCT ON (seller_id)
                    seller_id, seller_name,
                    seller_rating_score, seller_rating_count
                FROM staging
                ORDER BY seller_id
            """).format(sql.Identifier(f"{self.table_prefix}_sellers"))
            )

            # Insert into products (distinct products only - deduplicate by product_id)
            cur.execute(
                sql.SQL("""
                INSERT INTO {} (
                    product_id, product_name,
                    category_main, category_sub,
                    price, seller_id
                )
                SELECT DISTINCT ON (product_id)
                    product_id, product_name,
                    category_main, category_sub,
                    price, seller_id
                FROM staging
                ORDER BY product_id
            """).format(sql.Identifier(f"{self.table_prefix}_products"))
            )

            # Insert into orders (deduplicate by order_id)
            cur.execute(
                sql.SQL("""
                INSERT INTO {} (order_id, timestamp, customer_id)
                SELECT DISTINCT ON (order_id)
                    order_id, timestamp, customer_id
                FROM staging
                ORDER BY order_id
            """).format(sql.Identifier(f"{self.table_prefix}_orders"))
            )

            # Insert into payments (deduplicate by order_id since it's 1:1)
            cur.execute(
                sql.SQL("""
                INSERT INTO {} (
                    order_id, payment_method, payment_status,
                    payment_amount, payment_processor, payment_fee
                )
                SELECT DISTINCT ON (order_id)
                    order_id, payment_method, payment_status,
                    payment_amount, payment_processor, payment_fee
                FROM staging
                ORDER BY order_id
            """).format(sql.Identifier(f"{self.table_prefix}_payments"))
            )

            # Insert into shipping (deduplicate by order_id since it's 1:1)
            cur.execute(
                sql.SQL("""
                INSERT INTO {} (
                    order_id, shipping_status, shipping_method,
                    shipping_cost, shipping_city, shipping_state,
                    shipping_country, shipping_lat, shipping_lon
                )
                SELECT DISTINCT ON (order_id)
                    order_id, shipping_status, shipping_method,
                    shipping_cost, shipping_city, shipping_state,
                    shipping_country, shipping_lat, shipping_lon
                FROM staging
                ORDER BY order_id
            """).format(sql.Identifier(f"{self.table_prefix}_shipping"))
            )

            # Insert into order_products (deduplicate by order_id, product_id composite key)
            cur.execute(
                sql.SQL("""
                INSERT INTO {} (
                    order_id, product_id, quantity,
                    discount_applied, discount_percentage
                )
                SELECT DISTINCT ON (order_id, product_id)
                    order_id, product_id, quantity,
                    discount_applied, discount_percentage
                FROM staging
                ORDER BY order_id, product_id
            """).format(sql.Identifier(f"{self.table_prefix}_order_items"))
            )

            # Create indexes
            print("    Creating indexes...")
            self._create_indexes(cur)

            # Drop staging table
            cur.execute("DROP TABLE staging")

            # Commit all changes
            self.conn.commit()

            # Get final order count
            cur.execute(f"SELECT COUNT(*) FROM {self.table_prefix}_orders")
            count = cur.fetchone()[0]
            print(f"    ✓ Loaded {count:,} orders in normalized tables")

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
