import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import pandas as pd
from faker import Faker

fake = Faker()


# ============================================================================
# DATA FILE MANAGER
# ============================================================================
class DataFileManager:
    """Manages generation and loading of benchmark data files"""

    def __init__(self, data_dir: str = "./benchmark_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

    def generate_and_save(
        self, count: int, prefix: str = "benchmark", starting_order_id: int = 0
    ) -> Dict[str, str]:
        """
        Generate data and save to files in batches to avoid memory issues
        Returns dict with file paths

        Args:
            count: Number of records to generate
            prefix: Prefix for the output files
            starting_order_id: Starting ID offset to avoid conflicts with existing data
        """
        # Use different filenames when starting_order_id is specified to avoid conflicts
        if starting_order_id > 0:
            jsonl_path = self.data_dir / f"{count}_{prefix}_offset{starting_order_id}_nested.jsonl"
            flat_csv_path = self.data_dir / f"{count}_{prefix}_offset{starting_order_id}_flat.csv"
        else:
            jsonl_path = self.data_dir / f"{count}_{prefix}_nested.jsonl"
            flat_csv_path = self.data_dir / f"{count}_{prefix}_flat.csv"

        # Check if data already exists
        if jsonl_path.exists():
            print(f"\nData for {count:,} records exist, skipping generation...")
            return {
                "nested_jsonl": str(jsonl_path),
                "csv_flat": str(flat_csv_path),
            }

        print(f"\nGenerating {count:,} records in batches to avoid memory issues...")
        if starting_order_id > 0:
            print(f"  Using starting ID offset: {starting_order_id:,}")

        first_batch = True

        # Generation batch size (generate this many records at a time)
        generation_batch_size = 10000
        total_processed = 0
        total_flat_rows = 0

        # Open files for writing
        with open(jsonl_path, "w") as jsonl_file:
            # Generate and process in batches
            for batch_start in range(0, count, generation_batch_size):
                batch_end = min(batch_start + generation_batch_size, count)
                batch_count = batch_end - batch_start

                # Generate this batch with offset
                print(f"  Generating records {batch_start:,} to {batch_end:,}...")
                batch_data = NestedDataGenerator.generate_ecommerce_data(
                    batch_count, start_id=starting_order_id + batch_start
                )

                # Write to JSONL immediately
                for record in batch_data:
                    json.dump(record, jsonl_file, default=self._json_default)
                    jsonl_file.write("\n")

                # Flatten the batch to single denormalized table
                flat_rows = FlatDataGenerator.flatten_to_single_table(batch_data)

                # Write flat CSV (one row per item with all data denormalized)
                if flat_rows:
                    df = pd.DataFrame(flat_rows)
                    df.to_csv(
                        flat_csv_path,
                        mode="a",
                        header=first_batch,
                        index=False,
                    )
                    total_flat_rows += len(flat_rows)

                first_batch = False
                total_processed += batch_count

                # Clear memory
                del batch_data
                del flat_rows

                print(
                    f"  ✓ Processed {total_processed:,} / {count:,} records ({total_processed * 100 / count:.1f}%)"
                )

        print(f"\n✓ Generated and saved {total_processed:,} records total")
        print(f"✓ JSONL file: {jsonl_path}")
        print(f"✓ Flat CSV: {flat_csv_path} ({total_flat_rows:,} rows)")

        print()

        return {
            "nested_jsonl": str(jsonl_path),
            "csv_flat": str(flat_csv_path),
        }

    def load_nested_from_jsonl(self, jsonl_path: str) -> List[Dict]:
        """Load nested data from JSONL file"""
        data = []
        with open(jsonl_path, "r") as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data

    def load_flat_from_csv(self, csv_path: str) -> List[Dict]:
        """Load flat data from CSV file"""
        df = pd.read_csv(csv_path)
        return df.to_dict("records")

    @staticmethod
    def _json_default(obj):
        """Handle non-serializable types for JSON"""
        if hasattr(obj, "isoformat"):
            return obj.isoformat()
        elif hasattr(obj, "__str__"):
            return str(obj)
        return None


# ============================================================================
# DATA GENERATION - Reused from aggregation benchmark
# ============================================================================
class FlatDataGenerator:
    """Flatten nested data for relational storage"""

    @staticmethod
    def flatten_for_relational(nested_data: List[Dict]) -> Dict[str, List]:
        """
        Flatten nested e-commerce data into relational tables
        Returns: {
            'orders': [...],
            'items': [...],
            'customers': [...]
        }
        """
        orders = []
        items = []
        customers = {}

        for record in nested_data:
            # Customer table (deduplicated)
            cust_id = record["customer"]["id"]
            if cust_id not in customers:
                customers[cust_id] = {
                    "customer_id": cust_id,
                    "name": record["customer"]["name"],
                    "email": record["customer"]["email"],
                    "tier": record["customer"]["tier"],
                    "lifetime_value": record["customer"]["lifetime_value"],
                }

            # Order table (flattened)
            orders.append(
                {
                    "order_id": record["order_id"],
                    "customer_id": cust_id,
                    "timestamp": record["timestamp"],
                    "customer_tier": record["customer"]["tier"],
                    "payment_method": record["payment"]["method"],
                    "payment_status": record["payment"]["status"],
                    "payment_amount": record["payment"]["amount"],
                    "payment_processor": record["payment"]["processor"]["name"],
                    "payment_fee": record["payment"]["processor"]["fee"],
                    "shipping_status": record["shipping"]["status"],
                    "shipping_method": record["shipping"]["method"],
                    "shipping_cost": record["shipping"]["cost"],
                    "shipping_city": record["shipping"]["address"]["city"],
                    "shipping_state": record["shipping"]["address"]["state"],
                    "shipping_country": record["shipping"]["address"]["country"],
                    "shipping_lat": record["shipping"]["address"]["coordinates"]["lat"],
                    "shipping_lon": record["shipping"]["address"]["coordinates"]["lon"],
                }
            )

            # Items table (exploded from array)
            for item in record["items"]:
                items.append(
                    {
                        "order_id": record["order_id"],
                        "product_id": item["product_id"],
                        "product_name": item["name"],
                        "category_main": item["category"]["main"],
                        "category_sub": item["category"]["sub"],
                        "price": item["price"],
                        "quantity": item["quantity"],
                        "discount_applied": item["discount"]["applied"],
                        "discount_percentage": item["discount"]["percentage"],
                        "seller_id": item["seller"]["id"],
                        "seller_name": item["seller"]["name"],
                        "seller_rating_score": item["seller"]["rating"]["score"],
                        "seller_rating_count": item["seller"]["rating"]["count"],
                    }
                )

        return {"orders": orders, "items": items, "customers": list(customers.values())}

    @staticmethod
    def flatten_to_single_table(nested_data: List[Dict]) -> List[Dict]:
        """
        Flatten nested e-commerce data into a single denormalized table
        One row per item with all order and customer data repeated
        """
        flat_rows = []

        for record in nested_data:
            # For each item in the order, create a fully denormalized row
            for item in record["items"]:
                flat_row = {
                    # Order fields
                    "order_id": record["order_id"],
                    "timestamp": record["timestamp"],
                    # Customer fields
                    "customer_id": record["customer"]["id"],
                    "customer_name": record["customer"]["name"],
                    "customer_email": record["customer"]["email"],
                    "customer_tier": record["customer"]["tier"],
                    "customer_lifetime_value": record["customer"]["lifetime_value"],
                    # Payment fields
                    "payment_method": record["payment"]["method"],
                    "payment_status": record["payment"]["status"],
                    "payment_amount": record["payment"]["amount"],
                    "payment_processor": record["payment"]["processor"]["name"],
                    "payment_fee": record["payment"]["processor"]["fee"],
                    # Shipping fields
                    "shipping_status": record["shipping"]["status"],
                    "shipping_method": record["shipping"]["method"],
                    "shipping_cost": record["shipping"]["cost"],
                    "shipping_city": record["shipping"]["address"]["city"],
                    "shipping_state": record["shipping"]["address"]["state"],
                    "shipping_country": record["shipping"]["address"]["country"],
                    "shipping_lat": record["shipping"]["address"]["coordinates"]["lat"],
                    "shipping_lon": record["shipping"]["address"]["coordinates"]["lon"],
                    # Item fields
                    "product_id": item["product_id"],
                    "product_name": item["name"],
                    "category_main": item["category"]["main"],
                    "category_sub": item["category"]["sub"],
                    "price": item["price"],
                    "quantity": item["quantity"],
                    "discount_applied": item["discount"]["applied"],
                    "discount_percentage": item["discount"]["percentage"],
                    "seller_id": item["seller"]["id"],
                    "seller_name": item["seller"]["name"],
                    "seller_rating_score": item["seller"]["rating"]["score"],
                    "seller_rating_count": item["seller"]["rating"]["count"],
                }
                flat_rows.append(flat_row)

        return flat_rows


class NestedDataGenerator:
    """Generate deeply nested JSON data for testing"""

    @staticmethod
    def generate_ecommerce_data(count: int, start_id: int = 0) -> List[Dict]:
        """
        Generate e-commerce data with nested structure:
        - Customer info (nested)
        - Order items (array of nested objects)
        - Shipping address (nested)
        - Payment details (nested)

        Args:
            count: Number of records to generate
            start_id: Starting ID for order_id (for batch generation)
        """
        records = []

        for i in range(count):
            order_id = start_id + i
            num_items = random.randint(1, 10)
            items = []

            for _ in range(num_items):
                items.append(
                    {
                        "product_id": f"PROD-{random.randint(1, 1000):04d}",
                        "name": fake.catch_phrase(),
                        "category": {
                            "main": random.choice(
                                ["Electronics", "Clothing", "Home", "Books"]
                            ),
                            "sub": random.choice(
                                ["Gadgets", "Accessories", "Furniture", "Fiction"]
                            ),
                            "tags": [fake.word() for _ in range(random.randint(1, 3))],
                        },
                        "price": round(random.uniform(10, 500), 2),
                        "quantity": random.randint(1, 5),
                        "discount": {
                            "applied": random.choice([True, False]),
                            "percentage": round(random.uniform(0, 30), 2),
                            "code": f"DISC{random.randint(100, 999)}"
                            if random.random() > 0.5
                            else None,
                        },
                        "seller": {
                            "id": f"SELL-{random.randint(1, 100)}",
                            "name": fake.company(),
                            "rating": {
                                "score": round(random.uniform(3.0, 5.0), 1),
                                "count": random.randint(10, 10000),
                            },
                        },
                    }
                )

            record = {
                "order_id": f"ORD-{order_id:010d}",
                "timestamp": (
                    datetime.now() - timedelta(days=random.randint(0, 365))
                ).isoformat(),
                "customer": {
                    "id": f"CUST-{random.randint(1, 10000):08d}",
                    "name": fake.name(),
                    "email": fake.email(),
                    "tier": random.choice(["bronze", "silver", "gold", "platinum"]),
                    "lifetime_value": round(random.uniform(100, 50000), 2),
                    "preferences": {
                        "newsletter": random.choice([True, False]),
                        "categories": [
                            fake.word() for _ in range(random.randint(1, 4))
                        ],
                    },
                },
                "items": items,
                "payment": {
                    "method": random.choice(
                        ["credit_card", "paypal", "bank_transfer", "crypto"]
                    ),
                    "status": random.choice(
                        ["pending", "completed", "failed", "refunded"]
                    ),
                    "amount": round(
                        sum(item["price"] * item["quantity"] for item in items), 2
                    ),
                    "currency": "USD",
                    "processor": {
                        "name": random.choice(["Stripe", "PayPal", "Square"]),
                        "fee": round(random.uniform(1, 10), 2),
                    },
                },
                "shipping": {
                    "address": {
                        "street": fake.street_address(),
                        "city": fake.city(),
                        "state": fake.state_abbr(),
                        "country": fake.country_code(),
                        "postal_code": fake.postcode(),
                        "coordinates": {
                            "lat": float(fake.latitude()),
                            "lon": float(fake.longitude()),
                        },
                    },
                    "method": random.choice(["standard", "express", "overnight"]),
                    "cost": round(random.uniform(5, 50), 2),
                    "status": random.choice(
                        ["pending", "shipped", "delivered", "cancelled"]
                    ),
                },
                "metadata": {
                    "device": random.choice(["mobile", "desktop", "tablet"]),
                    "browser": random.choice(["chrome", "firefox", "safari", "edge"]),
                    "source": random.choice(["organic", "paid", "social", "direct"]),
                    "session_id": f"sess_{random.randint(100000, 999999)}",
                },
            }

            records.append(record)

        return records
