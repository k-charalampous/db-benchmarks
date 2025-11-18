-- PostgreSQL 18 initialization script
-- Note: pg_duckdb and TimescaleDB need manual installation
-- Create helper functions for benchmarking
CREATE
OR REPLACE FUNCTION jsonb_deep_size(jsonb_data JSONB) RETURNS INTEGER AS $$ BEGIN RETURN length(jsonb_data :: text);

END;

$$ LANGUAGE plpgsql IMMUTABLE;

-- Create function to check JSON path exists
CREATE
OR REPLACE FUNCTION jsonb_path_exists_any(data JSONB, paths TEXT []) RETURNS BOOLEAN AS $$ DECLARE path TEXT;

BEGIN FOREACH path IN ARRAY paths LOOP IF jsonb_path_exists(data, path :: jsonpath) THEN RETURN TRUE;

END IF;

END LOOP;

RETURN FALSE;

END;

$$ LANGUAGE plpgsql IMMUTABLE;

-- Create schema for test tables
CREATE SCHEMA IF NOT EXISTS benchmark;

GRANT ALL ON SCHEMA benchmark TO benchmark_user;

-- Display PostgreSQL version
DO $$ BEGIN RAISE NOTICE 'PostgreSQL version: %',
version();

END $$;