-- ClickHouse initialization
CREATE DATABASE IF NOT EXISTS benchmark_db;

-- Set some optimal settings for benchmarking
SET
    max_memory_usage = 10000000000;

SET
    max_bytes_before_external_group_by = 20000000000;