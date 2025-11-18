import time

import clickhouse_connect

from clickhouse_flat_strategy import ClickhouseStrategy
from timer import BenchmarkTimer


class ClickHouseFlatMVStrategy(ClickhouseStrategy):
    """ClickHouse Flat with Materialized Views for pre-computed aggregations"""

    def __init__(self, table_name: str, page_size: int = 100):
        super().__init__(table_name, page_size)

        # Override only aggregation queries to use materialized views
        self.queries.update({
            "simple_nested_agg": f"""
                SELECT
                    customer_tier,
                    order_count,
                    avg_amount
                FROM {table_name}_mv_simple_agg
            """,
            "deep_nested_agg": f"""
                SELECT
                    category_main,
                    count,
                    avg_lat
                FROM {table_name}_mv_deep_agg
                WHERE category_main != ''
            """,
            "array_aggregation": f"""
                SELECT
                    product_id,
                    times_ordered,
                    total_quantity,
                    avg_price
                FROM {table_name}_mv_array_agg
                ORDER BY times_ordered DESC
                LIMIT 100
            """,
            "complex_where_agg": f"""
                SELECT
                    customer_tier,
                    payment_status,
                    count,
                    total_amount
                FROM {table_name}_mv_complex_agg
            """,
            "seller_rating_agg": f"""
                SELECT
                    seller,
                    avg_rating,
                    product_count,
                    total_sold
                FROM {table_name}_mv_seller_agg
                HAVING product_count > 5
                ORDER BY avg_rating DESC
                LIMIT 50
            """,
        })

    def setup(self, csv_path: str):
        """Setup table and create materialized views for aggregations"""
        # First, call parent setup to create base table and load data
        ingestion_time = super().setup(csv_path)

        # If data already existed, we might still need to create MVs
        if ingestion_time is None:
            # Check if MVs exist
            result = self.client.query(f"""
                SELECT count() FROM system.tables
                WHERE database = 'benchmark_db'
                AND name LIKE '{self.table_name}_mv_%'
            """)
            existing_mvs = result.result_rows[0][0]

            if existing_mvs >= 5:
                print(f"    ✓ Materialized views for {self.table_name} already exist, skipping MV creation")
                return None

        print(f"    Creating materialized views for {self.table_name}...")
        mv_start = time.perf_counter()

        # Drop existing MVs if they exist
        self.client.command(f"DROP TABLE IF EXISTS {self.table_name}_mv_simple_agg")
        self.client.command(f"DROP TABLE IF EXISTS {self.table_name}_mv_deep_agg")
        self.client.command(f"DROP TABLE IF EXISTS {self.table_name}_mv_array_agg")
        self.client.command(f"DROP TABLE IF EXISTS {self.table_name}_mv_complex_agg")
        self.client.command(f"DROP TABLE IF EXISTS {self.table_name}_mv_seller_agg")

        # MV 1: Simple nested aggregation (customer_tier)
        self.client.command(f"""
            CREATE MATERIALIZED VIEW {self.table_name}_mv_simple_agg
            ENGINE = AggregatingMergeTree()
            ORDER BY customer_tier
            POPULATE
            AS SELECT
                customer_tier,
                uniq(order_id) as order_count,
                avg(payment_amount) as avg_amount
            FROM {self.table_name}
            GROUP BY customer_tier
        """)

        # MV 2: Deep nested aggregation (category_main)
        self.client.command(f"""
            CREATE MATERIALIZED VIEW {self.table_name}_mv_deep_agg
            ENGINE = AggregatingMergeTree()
            ORDER BY category_main
            POPULATE
            AS SELECT
                category_main,
                uniq(order_id) as count,
                avg(shipping_lat) as avg_lat
            FROM {self.table_name}
            WHERE category_main != ''
            GROUP BY category_main
        """)

        # MV 3: Array aggregation (product_id)
        self.client.command(f"""
            CREATE MATERIALIZED VIEW {self.table_name}_mv_array_agg
            ENGINE = SummingMergeTree()
            ORDER BY product_id
            POPULATE
            AS SELECT
                product_id,
                count() as times_ordered,
                sum(quantity) as total_quantity,
                avg(price) as avg_price
            FROM {self.table_name}
            GROUP BY product_id
        """)

        # MV 4: Complex WHERE aggregation (customer_tier, payment_status)
        self.client.command(f"""
            CREATE MATERIALIZED VIEW {self.table_name}_mv_complex_agg
            ENGINE = AggregatingMergeTree()
            ORDER BY (customer_tier, payment_status)
            POPULATE
            AS SELECT
                customer_tier,
                payment_status,
                uniq(order_id) as count,
                sum(payment_amount) as total_amount
            FROM {self.table_name}
            WHERE payment_amount > 100
              AND shipping_status = 'delivered'
              AND customer_tier IN ('gold', 'platinum')
            GROUP BY customer_tier, payment_status
        """)

        # MV 5: Seller rating aggregation (seller_name)
        self.client.command(f"""
            CREATE MATERIALIZED VIEW {self.table_name}_mv_seller_agg
            ENGINE = SummingMergeTree()
            ORDER BY seller
            POPULATE
            AS SELECT
                seller_name as seller,
                avg(seller_rating_score) as avg_rating,
                count() as product_count,
                sum(quantity) as total_sold
            FROM {self.table_name}
            GROUP BY seller_name
        """)

        mv_time = time.perf_counter() - mv_start
        print(f"    ✓ Created 5 materialized views in {mv_time:.2f}s")

        # Return the original ingestion time if this was a fresh load
        return ingestion_time

    def execute_query(self, query_type: str):
        """Execute query - MVs are used automatically for aggregations"""
        query = self.queries.get(query_type)
        with BenchmarkTimer(query_type) as timer:
            result = self.client.query(query)
            results = result.result_rows
        return timer.elapsed, len(results)
