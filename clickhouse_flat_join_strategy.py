import time

import clickhouse_connect
import pandas as pd

from strategy import Strategy
from timer import BenchmarkTimer


class ClickHouseFlatJoinStrategy(Strategy):
    """ClickHouse with normalized flat tables and JOIN operations"""

    def __init__(self, table_prefix: str = "ch_norm", page_size: int = 100):
        self.client = clickhouse_connect.get_client(
            host="localhost",
            port=8123,
            username="benchmark_user",
            password="benchmark_pass",
            database="benchmark_db",
        )
        self.table_prefix = table_prefix
        self.page_size = page_size

        # Define table names
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
                WHERE p.product_id LIKE 'PROD-0%'
                LIMIT {page_size}
            """,
            # Aggregation queries with JOINs
            "simple_nested_agg": f"""
                SELECT
                    c.customer_tier,
                    uniq(o.order_id) as order_count,
                    avg(pay.payment_amount) as avg_amount
                FROM {self.orders_table} o
                JOIN {self.customers_table} c ON o.customer_id = c.customer_id
                JOIN {self.payments_table} pay ON o.order_id = pay.order_id
                GROUP BY c.customer_tier
            """,
            "deep_nested_agg": f"""
                SELECT
                    p.category_main,
                    uniq(o.order_id) as count,
                    avg(s.shipping_lat) as avg_lat
                FROM {self.orders_table} o
                JOIN {self.order_items_table} oi ON o.order_id = oi.order_id
                JOIN {self.products_table} p ON oi.product_id = p.product_id
                JOIN {self.shipping_table} s ON o.order_id = s.order_id
                WHERE p.category_main IS NOT NULL AND p.category_main != ''
                GROUP BY p.category_main
            """,
            "array_aggregation": f"""
                SELECT
                    p.product_id,
                    count(*) as times_ordered,
                    sum(oi.quantity) as total_quantity,
                    avg(p.price) as avg_price
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
                    uniq(o.order_id) as count,
                    sum(pay.payment_amount) as total_amount
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
                    avg(sel.seller_rating_score) as avg_rating,
                    count(*) as product_count,
                    sum(oi.quantity) as total_sold
                FROM {self.order_items_table} oi
                JOIN {self.products_table} p ON oi.product_id = p.product_id
                JOIN {self.sellers_table} sel ON p.seller_id = sel.seller_id
                GROUP BY sel.seller_name, sel.seller_rating_score
                HAVING count(*) > 5
                ORDER BY avg_rating DESC
                LIMIT 50
            """,
        }

    def close(self):
        self.client.close()

    def setup(self, csv_path: str):
        """Setup normalized ClickHouse tables from CSV file. Returns ingestion time in seconds if data was loaded, None otherwise."""
        # Check if tables already exist
        tables = [
            self.orders_table,
            self.customers_table,
            self.products_table,
            self.payments_table,
            self.shipping_table,
            self.sellers_table,
            self.order_items_table,
        ]

        result = self.client.query(f"""
            SELECT count() FROM system.tables
            WHERE database = 'benchmark_db'
            AND name IN {tuple(tables)}
        """)
        existing_tables = result.result_rows[0][0]

        if existing_tables == 7:
            # All tables exist, check if they have data
            result = self.client.query(f"SELECT count() FROM {self.orders_table}")
            row_count = result.result_rows[0][0]
            if row_count > 0:
                print(
                    f"    ✓ ClickHouse normalized tables with prefix '{self.table_prefix}' already exist with {row_count:,} orders, skipping setup"
                )
                return None

        print(
            f"    Creating ClickHouse normalized tables with prefix '{self.table_prefix}'..."
        )
        ingestion_start = time.perf_counter()

        # Drop existing tables
        for table in tables:
            self.client.command(f"DROP TABLE IF EXISTS {table}")

        # Create normalized tables
        print("    Creating table schemas...")

        # Customers table
        self.client.command(f"""
            CREATE TABLE {self.customers_table} (
                customer_id String,
                customer_name String,
                customer_email String,
                customer_tier String,
                customer_lifetime_value Float64
            ) ENGINE = MergeTree()
            ORDER BY customer_id
        """)

        # Sellers table
        self.client.command(f"""
            CREATE TABLE {self.sellers_table} (
                seller_id String,
                seller_name String,
                seller_rating_score Float64,
                seller_rating_count UInt32
            ) ENGINE = MergeTree()
            ORDER BY seller_id
        """)

        # Products table
        self.client.command(f"""
            CREATE TABLE {self.products_table} (
                product_id String,
                product_name String,
                category_main String,
                category_sub String,
                price Float64,
                seller_id String
            ) ENGINE = MergeTree()
            ORDER BY (product_id, seller_id)
        """)

        # Orders table
        self.client.command(f"""
            CREATE TABLE {self.orders_table} (
                order_id String,
                timestamp String,
                customer_id String
            ) ENGINE = MergeTree()
            ORDER BY (customer_id, order_id)
        """)

        # Order items table
        self.client.command(f"""
            CREATE TABLE {self.order_items_table} (
                order_id String,
                product_id String,
                price Float64,
                quantity UInt32,
                discount_applied UInt8,
                discount_percentage Float64
            ) ENGINE = MergeTree()
            ORDER BY (order_id, product_id)
        """)

        # Payments table
        self.client.command(f"""
            CREATE TABLE {self.payments_table} (
                order_id String,
                payment_method String,
                payment_status String,
                payment_amount Float64,
                payment_processor String,
                payment_fee Float64
            ) ENGINE = MergeTree()
            ORDER BY order_id
        """)

        # Shipping table
        self.client.command(f"""
            CREATE TABLE {self.shipping_table} (
                order_id String,
                shipping_status String,
                shipping_method String,
                shipping_cost Float64,
                shipping_city String,
                shipping_state String,
                shipping_country String,
                shipping_lat Float64,
                shipping_lon Float64
            ) ENGINE = MergeTree()
            ORDER BY order_id
        """)

        # Load CSV and distribute to normalized tables
        print(f"    Loading and distributing data from {csv_path}...")

        # Read CSV in chunks
        chunk_size = 50000
        total_rows = 0

        # Temporary storage for deduplication
        customers_dict = {}
        sellers_dict = {}
        products_dict = {}
        orders_dict = {}

        customers_batch = []
        sellers_batch = []
        products_batch = []
        orders_batch = []
        payments_batch = []
        shipping_batch = []
        order_items_batch = []

        for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
            # Fill NaN values
            chunk = chunk.fillna(
                {
                    "order_id": "",
                    "timestamp": "",
                    "customer_id": "",
                    "customer_name": "",
                    "customer_email": "",
                    "customer_tier": "",
                    "customer_lifetime_value": 0.0,
                    "payment_method": "",
                    "payment_status": "",
                    "payment_amount": 0.0,
                    "payment_processor": "",
                    "payment_fee": 0.0,
                    "shipping_status": "",
                    "shipping_method": "",
                    "shipping_cost": 0.0,
                    "shipping_city": "",
                    "shipping_state": "",
                    "shipping_country": "",
                    "shipping_lat": 0.0,
                    "shipping_lon": 0.0,
                    "product_id": "",
                    "product_name": "",
                    "category_main": "",
                    "category_sub": "",
                    "price": 0.0,
                    "quantity": 0,
                    "discount_applied": False,
                    "discount_percentage": 0.0,
                    "seller_id": "",
                    "seller_name": "",
                    "seller_rating_score": 0.0,
                    "seller_rating_count": 0,
                }
            )

            for _, row in chunk.iterrows():
                # Deduplicate customers
                if row["customer_id"] not in customers_dict:
                    customers_dict[row["customer_id"]] = True
                    customers_batch.append(
                        [
                            row["customer_id"],
                            row["customer_name"],
                            row["customer_email"],
                            row["customer_tier"],
                            float(row["customer_lifetime_value"]),
                        ]
                    )

                # Deduplicate sellers
                if row["seller_id"] not in sellers_dict:
                    sellers_dict[row["seller_id"]] = True
                    sellers_batch.append(
                        [
                            row["seller_id"],
                            row["seller_name"],
                            float(row["seller_rating_score"]),
                            int(row["seller_rating_count"]),
                        ]
                    )

                # Deduplicate products
                if row["product_id"] not in products_dict:
                    products_dict[row["product_id"]] = True
                    products_batch.append(
                        [
                            row["product_id"],
                            row["product_name"],
                            row["category_main"],
                            row["category_sub"],
                            float(row["price"]),
                            row["seller_id"],
                        ]
                    )

                # Deduplicate orders
                if row["order_id"] not in orders_dict:
                    orders_dict[row["order_id"]] = True
                    orders_batch.append(
                        [row["order_id"], row["timestamp"], row["customer_id"]]
                    )

                    # Add payment (1:1 with order)
                    payments_batch.append(
                        [
                            row["order_id"],
                            row["payment_method"],
                            row["payment_status"],
                            float(row["payment_amount"]),
                            row["payment_processor"],
                            float(row["payment_fee"]),
                        ]
                    )

                    # Add shipping (1:1 with order)
                    shipping_batch.append(
                        [
                            row["order_id"],
                            row["shipping_status"],
                            row["shipping_method"],
                            float(row["shipping_cost"]),
                            row["shipping_city"],
                            row["shipping_state"],
                            row["shipping_country"],
                            float(row["shipping_lat"]),
                            float(row["shipping_lon"]),
                        ]
                    )

                # Always add order items
                order_items_batch.append(
                    [
                        row["order_id"],
                        row["product_id"],
                        float(row["price"]),
                        int(row["quantity"]),
                        1 if row["discount_applied"] else 0,
                        float(row["discount_percentage"]),
                    ]
                )

            total_rows += len(chunk)

            # Insert batches periodically
            if len(customers_batch) >= 10000:
                self._insert_batches(
                    customers_batch,
                    sellers_batch,
                    products_batch,
                    orders_batch,
                    payments_batch,
                    shipping_batch,
                    order_items_batch,
                )
                customers_batch = []
                sellers_batch = []
                products_batch = []
                orders_batch = []
                payments_batch = []
                shipping_batch = []
                order_items_batch = []

            if total_rows % 100000 == 0:
                print(f"      Processed {total_rows:,} rows...")

        # Insert remaining batches
        if customers_batch or orders_batch or order_items_batch:
            self._insert_batches(
                customers_batch,
                sellers_batch,
                products_batch,
                orders_batch,
                payments_batch,
                shipping_batch,
                order_items_batch,
            )

        # Get final counts
        order_count = self.client.query(
            f"SELECT count() FROM {self.orders_table}"
        ).result_rows[0][0]
        items_count = self.client.query(
            f"SELECT count() FROM {self.order_items_table}"
        ).result_rows[0][0]
        print(f"    ✓ Loaded {order_count:,} orders with {items_count:,} order items")

        ingestion_time = time.perf_counter() - ingestion_start
        print(f"    ✓ Ingestion and normalization completed in {ingestion_time:.2f}s")

        return ingestion_time

    def _insert_batches(
        self, customers, sellers, products, orders, payments, shipping, order_items
    ):
        """Insert batches into normalized tables"""
        if customers:
            self.client.insert(
                self.customers_table,
                customers,
                column_names=[
                    "customer_id",
                    "customer_name",
                    "customer_email",
                    "customer_tier",
                    "customer_lifetime_value",
                ],
            )
        if sellers:
            self.client.insert(
                self.sellers_table,
                sellers,
                column_names=[
                    "seller_id",
                    "seller_name",
                    "seller_rating_score",
                    "seller_rating_count",
                ],
            )
        if products:
            self.client.insert(
                self.products_table,
                products,
                column_names=[
                    "product_id",
                    "product_name",
                    "category_main",
                    "category_sub",
                    "price",
                    "seller_id",
                ],
            )
        if orders:
            self.client.insert(
                self.orders_table,
                orders,
                column_names=["order_id", "timestamp", "customer_id"],
            )
        if payments:
            self.client.insert(
                self.payments_table,
                payments,
                column_names=[
                    "order_id",
                    "payment_method",
                    "payment_status",
                    "payment_amount",
                    "payment_processor",
                    "payment_fee",
                ],
            )
        if shipping:
            self.client.insert(
                self.shipping_table,
                shipping,
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
        if order_items:
            self.client.insert(
                self.order_items_table,
                order_items,
                column_names=[
                    "order_id",
                    "product_id",
                    "price",
                    "quantity",
                    "discount_applied",
                    "discount_percentage",
                ],
            )

    def execute_query(self, query_type: str):
        query = self.queries.get(query_type)
        with BenchmarkTimer(query_type) as timer:
            result = self.client.query(query)
            results = result.result_rows
        return timer.elapsed, len(results)
