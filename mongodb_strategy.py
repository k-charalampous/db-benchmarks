import json
import time

import pymongo

from strategy import Strategy
from timer import BenchmarkTimer


class MongoDBStrategy(Strategy):
    """MongoDB native queries"""

    def __init__(
        self, conn_string: str, db_name: str, collection_name: str, page_size: int = 100
    ):
        self.client = pymongo.MongoClient(conn_string)
        self.db = self.client[db_name]
        self.page_size = page_size
        self.collection_name = collection_name
        self.queries = {
            "simple_where": {"filter": {"customer.tier": "gold"}, "limit": page_size},
            "complex_where": {
                "filter": {
                    "payment.amount": {"$gt": 100},
                    "shipping.status": "delivered",
                    "customer.tier": {"$in": ["gold", "platinum"]},
                    "payment.status": "completed",
                },
                "limit": page_size,
            },
            "pagination_early": {
                "filter": {"customer.tier": "gold"},
                "skip": 100,
                "limit": page_size,
            },
            "pagination_deep": {
                "filter": {"customer.tier": "gold"},
                "skip": 10000,
                "limit": page_size,
            },
            "nested_array_filter": {
                "filter": {"items.product_id": {"$regex": "^PROD-0"}},
                "limit": page_size,
            },
            "simple_nested_agg": [
                {
                    "$group": {
                        "_id": "$customer.tier",
                        "order_count": {"$sum": 1},
                        "avg_amount": {"$avg": "$payment.amount"},
                    }
                }
            ],
            "deep_nested_agg": [
                {"$match": {"items.0.category.main": {"$exists": True}}},
                {
                    "$group": {
                        "_id": {"$arrayElemAt": ["$items.category.main", 0]},
                        "count": {"$sum": 1},
                        "avg_lat": {"$avg": "$shipping.address.coordinates.lat"},
                    }
                },
            ],
            "array_aggregation": [
                {"$unwind": "$items"},
                {
                    "$group": {
                        "_id": "$items.product_id",
                        "times_ordered": {"$sum": 1},
                        "total_quantity": {"$sum": "$items.quantity"},
                        "avg_price": {"$avg": "$items.price"},
                    }
                },
                {"$sort": {"times_ordered": -1}},
                {"$limit": 100},
            ],
            "complex_where_agg": [
                {
                    "$match": {
                        "payment.amount": {"$gt": 100},
                        "shipping.status": "delivered",
                        "customer.tier": {"$in": ["gold", "platinum"]},
                    }
                },
                {
                    "$group": {
                        "_id": {"tier": "$customer.tier", "status": "$payment.status"},
                        "count": {"$sum": 1},
                        "total_amount": {"$sum": "$payment.amount"},
                    }
                },
            ],
            "seller_rating_agg": [
                {"$unwind": "$items"},
                {
                    "$group": {
                        "_id": "$items.seller.name",
                        "avg_rating": {"$avg": "$items.seller.rating.score"},
                        "product_count": {"$sum": 1},
                        "total_sold": {"$sum": "$items.quantity"},
                    }
                },
                {"$match": {"product_count": {"$gt": 5}}},
                {"$sort": {"avg_rating": -1}},
                {"$limit": 50},
            ],
        }

    def close(self):
        self.client.close()

    def setup(self, jsonl_path: str):
        """Setup collection and create indexes. Returns ingestion time in seconds if data was loaded, None otherwise."""
        collection_exists = self.collection_name in self.db.list_collection_names()

        if collection_exists:
            # Check if collection has data
            collection = self.db[self.collection_name]
            doc_count = collection.count_documents({})
            if doc_count > 0:
                print(
                    f"    ✓ Collection {self.collection_name} already exists with {doc_count:,} documents, skipping setup"
                )
                return None

        print(f"    Creating collection {self.collection_name}...")

        # Start timing ingestion
        ingestion_start = time.perf_counter()

        collection = self.db[self.collection_name]

        batch = []
        batch_size = 1000
        count = 0

        print(f"    Loading data from {jsonl_path}...")
        with open(jsonl_path, "r") as f:
            for line in f:
                record = json.loads(line.strip())
                batch.append(record)
                count += 1

                if len(batch) >= batch_size:
                    collection.insert_many(batch, ordered=False)
                    batch = []

            if batch:
                collection.insert_many(batch, ordered=False)

        print(f"    ✓ Loaded {count:,} records")
        print("    Creating indexes...")
        collection.create_index([("customer.tier", 1)])
        collection.create_index([("payment.amount", 1)])
        collection.create_index([("payment.status", 1)])
        collection.create_index([("shipping.status", 1)])
        print("    ✓ Created indexes")

        ingestion_time = time.perf_counter() - ingestion_start
        print(f"    ✓ Ingestion completed in {ingestion_time:.2f}s")

        return ingestion_time

    def execute_query(self, query_type: str):
        query = self.queries.get(query_type)
        collection = self.db[self.collection_name]
        with BenchmarkTimer(query_type) as timer:
            if type(query) is list:
                results = list(collection.aggregate(query))
            else:
                cursor = collection.find(
                    query["filter"],
                    skip=query.get("skip", 0),
                    limit=query["limit"],
                )
                results = list(cursor)
        return timer.elapsed, len(results)
