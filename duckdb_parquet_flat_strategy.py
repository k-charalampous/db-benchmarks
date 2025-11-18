from pathlib import Path

import boto3
import duckdb
import pandas as pd
from botocore.client import Config

from strategy import Strategy
from timer import BenchmarkTimer


class DuckDBParquetMinIOStrategy(Strategy):
    """DuckDB querying Parquet files stored on MinIO (S3-compatible storage)"""

    def __init__(
        self,
        table_name: str,
        minio_endpoint: str = "http://localhost:9000",
        minio_access_key: str = "minioadmin",
        minio_secret_key: str = "minioadmin",
        bucket_name: str = "benchmark-data",
        page_size: int = 100,
    ):
        self.minio_endpoint = minio_endpoint
        self.minio_access_key = minio_access_key
        self.minio_secret_key = minio_secret_key
        self.bucket_name = bucket_name
        self.conn = duckdb.connect(":memory:")
        self.parquet_dir = Path("./benchmark_data/parquet")
        self.parquet_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = f"s3://{self.bucket_name}/{table_name}.parquet"
        self.table_name = table_name
        # Initialize S3 client for MinIO
        self.s3_client = boto3.client(
            "s3",
            endpoint_url=minio_endpoint,
            aws_access_key_id=minio_access_key,
            aws_secret_access_key=minio_secret_key,
            config=Config(signature_version="s3v4"),
        )

        self.queries = {
            "simple_where": f"""
                SELECT *
                FROM read_parquet('{self.file_path}')
                WHERE customer_tier = 'gold'
                LIMIT {page_size}
            """,
            "complex_where": f"""
                SELECT *
                FROM read_parquet('{self.file_path}')
                WHERE payment_amount > 100
                  AND shipping_status = 'delivered'
                  AND customer_tier IN ('gold', 'platinum')
                  AND payment_status = 'completed'
                LIMIT {page_size}
            """,
            "pagination_early": f"""
                SELECT *
                FROM read_parquet('{self.file_path}')
                WHERE customer_tier = 'gold'
                ORDER BY order_id
                LIMIT {page_size} OFFSET 100
            """,
            "pagination_deep": f"""
                SELECT *
                FROM read_parquet('{self.file_path}')
                WHERE customer_tier = 'gold'
                ORDER BY order_id
                LIMIT {page_size} OFFSET 10000
            """,
            "nested_array_filter": f"""
                SELECT *
                FROM read_parquet('{self.file_path}')
                WHERE product_id LIKE 'PROD-0%'
                LIMIT {page_size}
            """,
            # Aggregation queries
            "simple_nested_agg": f"""
                SELECT
                    customer_tier,
                    COUNT(DISTINCT order_id) as order_count,
                    AVG(payment_amount) as avg_amount
                FROM read_parquet('{self.file_path}')
                GROUP BY customer_tier
            """,
            "deep_nested_agg": f"""
                SELECT
                    category_main,
                    COUNT(*) as count,
                    AVG(shipping_lat) as avg_lat
                FROM read_parquet('{self.file_path}')
                GROUP BY category_main
            """,
            "array_aggregation": f"""
                SELECT
                    order_id,
                    COUNT(product_id) as item_count,
                    SUM(price * quantity) as total_value,
                    AVG(price) as avg_item_price
                FROM read_parquet('{self.file_path}')
                GROUP BY order_id
                LIMIT {page_size}
            """,
            "complex_where_agg": f"""
                SELECT
                    customer_tier,
                    payment_status,
                    COUNT(DISTINCT order_id) as count,
                    SUM(payment_amount) as total_amount
                FROM read_parquet('{self.file_path}')
                WHERE payment_amount > 100
                  AND shipping_status = 'delivered'
                  AND customer_tier IN ('gold', 'platinum')
                GROUP BY customer_tier, payment_status
            """,
            "seller_rating_agg": f"""
                SELECT
                    seller_id,
                    seller_name,
                    AVG(seller_rating_score) as avg_rating,
                    SUM(seller_rating_count) as total_ratings,
                    COUNT(DISTINCT order_id) as order_count,
                    SUM(price * quantity) as total_revenue
                FROM read_parquet('{self.file_path}')
                GROUP BY seller_id, seller_name
                HAVING COUNT(DISTINCT order_id) > 5
                ORDER BY total_revenue DESC
                LIMIT {page_size}
            """,
        }

    def close(self):
        self.conn.close()

    def _ensure_bucket_exists(self):
        """Create MinIO bucket if it doesn't exist"""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            print(f"    ✓ Bucket '{self.bucket_name}' already exists")
        except Exception:
            print(f"    Creating bucket '{self.bucket_name}'...")
            self.s3_client.create_bucket(Bucket=self.bucket_name)
            print("    ✓ Bucket created")

    def _convert_csv_to_parquet(self, csv_path: str) -> str:
        """Convert CSV file to Parquet format"""
        parquet_path = self.parquet_dir / f"{self.table_name}.parquet"

        # Check if parquet file already exists and is recent
        if parquet_path.exists():
            csv_mtime = Path(csv_path).stat().st_mtime
            parquet_mtime = parquet_path.stat().st_mtime
            if parquet_mtime >= csv_mtime:
                print(
                    f"    ✓ Parquet file for {self.table_name} already exists and is up-to-date"
                )
                return str(parquet_path)

        print(f"    Converting {self.table_name} CSV to Parquet...")
        df = pd.read_csv(csv_path)
        df.to_parquet(parquet_path, engine="pyarrow", compression="snappy", index=False)
        file_size = parquet_path.stat().st_size / (1024 * 1024)
        print(f"    ✓ Created {self.table_name}.parquet ({file_size:.2f} MB)")
        return str(parquet_path)

    def _upload_to_minio(self, local_path: str, s3_key: str):
        """Upload file to MinIO"""
        try:
            # Check if file already exists in MinIO with same size
            try:
                response = self.s3_client.head_object(
                    Bucket=self.bucket_name, Key=s3_key
                )
                remote_size = response["ContentLength"]
                local_size = Path(local_path).stat().st_size
                if remote_size == local_size:
                    print(
                        f"    ✓ {s3_key} already exists in MinIO with same size, skipping upload"
                    )
                    return
            except Exception:
                pass  # File doesn't exist, proceed with upload

            print(f"    Uploading {s3_key} to MinIO...")
            self.s3_client.upload_file(local_path, self.bucket_name, s3_key)
            print(f"    ✓ Uploaded {s3_key}")
        except Exception as e:
            print(f"    Error uploading {s3_key}: {e}")
            raise

    def setup(self, csv_path: str):
        """Setup single flat Parquet file in MinIO from flat CSV file"""
        print(f"    Setting up flat Parquet file in MinIO for {self.file_path}...")

        # Ensure bucket exists
        self._ensure_bucket_exists()

        # Convert single flat CSV to Parquet
        parquet_path = self._convert_csv_to_parquet(csv_path)

        # Upload to MinIO
        s3_key = f"{self.table_name}.parquet"
        self._upload_to_minio(parquet_path, s3_key)

        # Configure DuckDB to use MinIO
        print("    Configuring DuckDB S3 settings...")
        self.conn.execute("INSTALL httpfs;")
        self.conn.execute("LOAD httpfs;")
        self.conn.execute(
            f"SET s3_endpoint='{self.minio_endpoint.replace('http://', '')}';"
        )
        self.conn.execute(f"SET s3_access_key_id='{self.minio_access_key}';")
        self.conn.execute(f"SET s3_secret_access_key='{self.minio_secret_key}';")
        self.conn.execute("SET s3_use_ssl=false;")
        self.conn.execute("SET s3_url_style='path';")

        # Verify we can read the data
        try:
            flat_count = self.conn.execute(
                f"SELECT COUNT(*) FROM read_parquet('s3://{self.bucket_name}/{self.file_path}.parquet')"
            ).fetchone()[0]
            print("    ✓ Verified data in MinIO:")
            print(f"      - Flat table: {flat_count:,} rows")
        except Exception as e:
            print(f"    Warning: Could not verify data in MinIO: {e}")

    def execute_query(self, query_type: str):
        query = self.queries.get(query_type)
        with BenchmarkTimer(query_type) as timer:
            results = self.conn.execute(query).fetchall()
        return timer.elapsed, len(results)
