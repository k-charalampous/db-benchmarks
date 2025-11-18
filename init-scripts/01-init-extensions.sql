-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Note: pg_duckdb requires manual installation
-- Follow instructions at: https://github.com/duckdb/pg_duckdb
-- After container starts, run:
-- apt-get update && apt-get install -y git build-essential postgresql-server-dev-18
-- cd /tmp && git clone https://github.com/duckdb/pg_duckdb
-- cd pg_duckdb && make install
-- Then: CREATE EXTENSION pg_duckdb;
-- Create helper functions for benchmarking
CREATE
OR REPLACE FUNCTION jsonb_deep_size(jsonb_data JSONB) RETURNS INTEGER AS $ $ BEGIN RETURN length(jsonb_data :: text);

END;

$ $ LANGUAGE plpgsql IMMUTABLE;

-- Create schema for test tables
CREATE SCHEMA IF NOT EXISTS benchmark;

GRANT ALL ON SCHEMA benchmark TO benchmark_user;