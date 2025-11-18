import time

from clickhouse_flat_join_strategy import ClickHouseFlatJoinStrategy
from timer import BenchmarkTimer


class ClickHouseFlatJoinMVStrategy(ClickHouseFlatJoinStrategy):
    """ClickHouse Flat Join with Materialized Views for pre-computed aggregations"""

    def __init__(self, table_prefix: str = "ch_norm_mv", page_size: int = 100):
        super().__init__(table_prefix, page_size)

        # Override only aggregation queries to use materialized views
        self.queries.update(
            {
                "simple_nested_agg": f"""
                SELECT
                    customer_tier,
                    order_count,
                    avg_amount
                FROM {self.orders_table}_mv_simple_agg
            """,
                "deep_nested_agg": f"""
                SELECT
                    category_main,
                    count,
                    avg_lat
                FROM {self.orders_table}_mv_deep_agg
                WHERE category_main != ''
            """,
                "array_aggregation": f"""
                SELECT
                    product_id,
                    times_ordered,
                    total_quantity,
                    avg_price
                FROM {self.order_items_table}_mv_array_agg
                ORDER BY times_ordered DESC
                LIMIT 100
            """,
                "complex_where_agg": f"""
                SELECT
                    customer_tier,
                    payment_status,
                    count,
                    total_amount
                FROM {self.orders_table}_mv_complex_agg
            """,
                "seller_rating_agg": f"""
                SELECT
                    seller,
                    avg_rating,
                    product_count,
                    total_sold
                FROM {self.order_items_table}_mv_seller_agg
                HAVING product_count > 5
                ORDER BY avg_rating DESC
                LIMIT 50
            """,
            }
        )

    def setup(self, csv_path: str):
        """Setup normalized tables and create materialized views for aggregations"""
        # First, call parent setup to create tables and load data
        ingestion_time = super().setup(csv_path)

        # If data already existed, we might still need to create MVs
        if ingestion_time is None:
            # Check if MVs exist
            result = self.client.query(f"""
                SELECT count() FROM system.tables
                WHERE database = 'benchmark_db'
                AND name LIKE '{self.table_prefix}%_mv_%'
            """)
            existing_mvs = result.result_rows[0][0]

            if existing_mvs >= 5:
                print(
                    f"    ✓ Materialized views for {self.table_prefix} already exist, skipping MV creation"
                )
                return None

        print("    Creating materialized views for normalized tables...")
        mv_start = time.perf_counter()

        # Drop existing MVs if they exist
        self.client.command(f"DROP TABLE IF EXISTS {self.orders_table}_mv_simple_agg")
        self.client.command(f"DROP TABLE IF EXISTS {self.orders_table}_mv_deep_agg")
        self.client.command(
            f"DROP TABLE IF EXISTS {self.order_items_table}_mv_array_agg"
        )
        self.client.command(f"DROP TABLE IF EXISTS {self.orders_table}_mv_complex_agg")
        self.client.command(
            f"DROP TABLE IF EXISTS {self.order_items_table}_mv_seller_agg"
        )

        # MV 1: Simple nested aggregation (customer_tier)
        self.client.command(f"""
            CREATE MATERIALIZED VIEW {self.orders_table}_mv_simple_agg
            ENGINE = AggregatingMergeTree()
            ORDER BY customer_tier
            POPULATE
            AS SELECT
                c.customer_tier,
                uniq(o.order_id) as order_count,
                avg(pay.payment_amount) as avg_amount
            FROM {self.orders_table} o
            JOIN {self.customers_table} c ON o.customer_id = c.customer_id
            JOIN {self.payments_table} pay ON o.order_id = pay.order_id
            GROUP BY c.customer_tier
        """)

        # MV 2: Deep nested aggregation (category_main)
        self.client.command(f"""
            CREATE MATERIALIZED VIEW {self.orders_table}_mv_deep_agg
            ENGINE = AggregatingMergeTree()
            ORDER BY category_main
            POPULATE
            AS SELECT
                p.category_main,
                uniq(o.order_id) as count,
                avg(s.shipping_lat) as avg_lat
            FROM {self.orders_table} o
            JOIN {self.order_items_table} oi ON o.order_id = oi.order_id
            JOIN {self.products_table} p ON oi.product_id = p.product_id
            JOIN {self.shipping_table} s ON o.order_id = s.order_id
            WHERE p.category_main != ''
            GROUP BY p.category_main
        """)

        # MV 3: Array aggregation (product_id)
        self.client.command(f"""
            CREATE MATERIALIZED VIEW {self.order_items_table}_mv_array_agg
            ENGINE = SummingMergeTree()
            ORDER BY product_id
            POPULATE
            AS SELECT
                p.product_id as product_id,
                count() as times_ordered,
                sum(oi.quantity) as total_quantity,
                avg(p.price) as avg_price
            FROM {self.order_items_table} oi
            JOIN {self.products_table} p ON oi.product_id = p.product_id
            GROUP BY p.product_id
        """)

        # MV 4: Complex WHERE aggregation (customer_tier, payment_status)
        self.client.command(f"""
            CREATE MATERIALIZED VIEW {self.orders_table}_mv_complex_agg
            ENGINE = AggregatingMergeTree()
            ORDER BY (customer_tier, payment_status)
            POPULATE
            AS SELECT
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
        """)

        # MV 5: Seller rating aggregation (seller_name)
        self.client.command(f"""
            CREATE MATERIALIZED VIEW {self.order_items_table}_mv_seller_agg
            ENGINE = SummingMergeTree()
            ORDER BY seller
            POPULATE
            AS SELECT
                sel.seller_name as seller,
                avg(sel.seller_rating_score) as avg_rating,
                count() as product_count,
                sum(oi.quantity) as total_sold
            FROM {self.order_items_table} oi
            JOIN {self.products_table} p ON oi.product_id = p.product_id
            JOIN {self.sellers_table} sel ON p.seller_id = sel.seller_id
            GROUP BY sel.seller_name, sel.seller_rating_score
        """)

        mv_time = time.perf_counter() - mv_start
        print(f"    ✓ Created 5 materialized views with JOINs in {mv_time:.2f}s")

        # Return the original ingestion time if this was a fresh load
        return ingestion_time

    def execute_query(self, query_type: str):
        """Execute query - MVs are used automatically for aggregations"""
        query = self.queries.get(query_type)
        with BenchmarkTimer(query_type) as timer:
            result = self.client.query(query)
            results = result.result_rows
        return timer.elapsed, len(results)
