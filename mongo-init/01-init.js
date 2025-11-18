// MongoDB initialization script
db = db.getSiblingDB("benchmark_db");

// Create collections with validation
db.createCollection("benchmark_flat", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["order_id", "customer_id", "order_date"],
      properties: {
        order_id: { bsonType: "string" },
        customer_id: { bsonType: "string" },
        order_date: { bsonType: "string" },
        status: { bsonType: "string" },
        total_amount: { bsonType: "number" },
      },
    },
  },
});

db.createCollection("benchmark_nested", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["post_id", "author", "timestamp"],
      properties: {
        post_id: { bsonType: "string" },
        author: { bsonType: "object" },
        timestamp: { bsonType: "string" },
      },
    },
  },
});

print("MongoDB benchmark database initialized");
