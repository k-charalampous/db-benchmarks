# Database Benchmarks

Comprehensive benchmarking suite for comparing query and ingestion performance across different database systems (PostgreSQL, ClickHouse, MongoDB, DuckDB).

## Quick Start

### 1. Start Database Services

Start all database containers using Docker Compose:

```bash
docker-compose up -d
```

Wait for all services to be healthy (check with `docker-compose ps`).

### 2. Setup Python Environment

#### Option A: Using Poetry (Recommended)

```bash
poetry install
poetry shell
```

#### Option B: Using virtualenv + requirements.txt

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run Query Benchmarks

**Important:** Run query benchmarks first - they will generate the dataset if it doesn't exist.

```bash
# Create results directory
mkdir -p results

# Run with default dataset size (100,000 records)
mkdir -p results/100000
python main.py 2>&1 | tee results/query_benchmark_100000.log

# Run with custom dataset size (1 million records)
mkdir -p results/1000000
python main.py --dataset-size 1000000 2>&1 | tee results/1000000/query_benchmark.log


# Run with custom dataset size (10 million records)
mkdir -p results/10000000
python main.py --dataset-size 10000000 2>&1 | tee results/10000000/query_benchmark.log
```

### 4. Run Ingestion Benchmarks

After query benchmarks complete, run ingestion benchmarks:

```bash
# Run with default settings
mkdir -p results/100000
python run_ingestion_benchmark.py --dataset-size 100000 --test-size 10000 2>&1 | tee results/100000/ingestion_benchmark.log

# Customize parameters 1M bulk and 50 batches/sec for 100seconds on 10M dataset size
mkdir -p results/10000000
python run_ingestion_benchmark.py \
  --dataset-size 1000000 \
  --test-size 1000000 \
  --batch-size 200 \
  --batches-per-second 50 \
  --duration 100 \
  2>&1 | tee results/10000000/ingestion_benchmark.log
```

## Benchmark Parameters

### Query Benchmarks (`main.py`)

- `--dataset-size`: Number of records to generate and query (default: 100000)

### Ingestion Benchmarks (`run_ingestion_benchmark.py`)

- `--dataset-size`: Size of existing dataset for table name resolution (default: 1000000)
- `--test-size`: Number of records to insert during bulk benchmark (default: 10000)
- `--batch-size`: Records per batch for high-frequency mode (default: 100)
- `--batches-per-second`: Target batch frequency (default: 1)
- `--duration`: Duration of high-frequency test in seconds (default: 60)

## Results

Benchmark results are saved in:

- **Plots**: PNG files in the `results/{dataset_size}` directory
- **Logs**: Text files in `results/` directory
- **Console**: Real-time progress and metrics

## Databases Tested

- **PostgreSQL 17** with pg_duckdb extension
- **PostgreSQL 18**
- **ClickHouse** (flat, nested, materialized views)
- **MongoDB**
- **DuckDB** with Parquet + MinIO

## Cleanup

Stop and remove all containers:

```bash
docker-compose down
```

Remove data volumes (WARNING: deletes all data):

```bash
docker-compose down -v
```
