# Vector Database Benchmarking Suite

A benchmarking suite for evaluating vector databases and search engines on standard ANN (Approximate Nearest Neighbor) datasets.

## Installation

```bash
pip install -r requirements.txt
```

## Available Datasets

The benchmark suite supports the following datasets from [ann-benchmarks.com](http://ann-benchmarks.com):

- `sift-128`: SIFT descriptors (128 dimensions, 1M vectors)
- `fashion-mnist-784`: Fashion-MNIST (784 dimensions, 60K vectors)
- `mnist-784`: MNIST digits (784 dimensions, 60K vectors)
- `gist-960`: GIST descriptors (960 dimensions, 1M vectors)

Datasets are automatically downloaded to `benchmark/data/` if not present.

## Benchmarks

### 1. Database Benchmark (`benchmark/main.py`)

Tests vector databases with client-server architecture.

**Supported databases:**
- Qdrant
- ChromaDB
- Weaviate
- Milvus
- LanceDB
- brinicle

### 2. Embedding Engine Benchmark (`benchmark/embed_bench.py`)

Tests in-process vector search libraries.

**Supported engines:**
- FAISS
- HNSWLib

## Database Setup

Before running benchmarks, start the corresponding database server:

### Qdrant
```bash
docker run -p 6333:6333 qdrant/qdrant
```

### ChromaDB
```bash
docker run -v ./chroma-data:/data -p 8000:8000 chromadb/chroma
```

### Weaviate
Refer to https://github.com/weaviate/weaviate README.md file to get the docker-compose content and store it in a docker-compose.yml file.
Then, run this:
```bash
docker-compose up -d
```

### Milvus
```bash
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh
bash standalone_embed.sh start
```

### brinicle
```bash
git clone https://github.com/bicardinal/brinicle.git
cd brinicle
bash build.sh
make docker-build
make docker-run
```

## Running Benchmarks

### Basic Usage

**Database Benchmark:**
```bash
python benchmark/main.py --db qdrant --dataset sift-128
```

**Engine Benchmark:**
```bash
sudo cpupower -c 2 frequency-set -g performance
taskset -c 2 python benchmark/embed_bench.py --engine faiss --dataset mnist-784
```
To benchmark brinicle library:
```bash
git clone https://github.com/bicardinal/brinicle.git
cd brinicle
bash build.sh
```
Copy brinicle/\_brinicle.cpythons to the .db_bench directory.
Then:
```bash
sudo cpupower -c 2 frequency-set -g performance
taskset -c 2 python -m benchmark.embed_bench --engine brinicle --dataset mnist-784
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--dataset` | Dataset to use: `sift-128`, `fashion-mnist-784`, `mnist-784`, `gist-960` | `sift-128` |
| `--db` / `--engine` | Database or engine to test | `brinicle` / `faiss` |
| `--m` | HNSW M parameter (max connections per layer) | `16` |
| `--efc` | ef_construction (index building quality) | `200` |
| `--efs` | ef_search (search quality/speed tradeoff) | `64` |
| `--max-queries` | Number of queries to run | `10000` |
| `--sample` | Randomly sample queries instead of first N | `False` |
| `--seed` | Random seed for reproducibility | `123` |
| `--data-dir` | Directory for datasets | `./benchmark/data` |

### Example Commands

**Test Qdrant with GIST dataset:**
```bash
python benchmark/main.py --db qdrant --dataset gist-960 --m 32 --efc 400 --efs 100
```

**Test FAISS with Fashion-MNIST:**
```bash
python benchmark/embed_bench.py --engine faiss --dataset fashion-mnist-784 --m 16 --efc 200
```

**Run with limited queries (faster testing):**
```bash
python benchmark/main.py --db chroma --dataset sift-128 --max-queries 1000
```

## Resource-Constrained Benchmarking

To test performance under resource constraints, use Docker's resource limitation flags when starting database containers:

### Memory-Limited Testing

**Qdrant (2GB RAM limit):**
```bash
docker run --memory="2g" -p 6333:6333 qdrant/qdrant
```

**ChromaDB (1GB RAM limit):**
```bash
docker run --memory="1g" -v ./chroma-data:/data -p 8000:8000 chromadb/chroma
```

**Milvus (4GB RAM limit):**
```bash
docker run --memory="4g" -p 19530:19530 milvusdb/milvus:latest
```

### CPU-Limited Testing

**Qdrant (2 CPUs):**
```bash
docker run --cpus="2.0" -p 6333:6333 qdrant/qdrant
```

**ChromaDB (1 CPU):**
```bash
docker run --cpus="1.0" -v ./chroma-data:/data -p 8000:8000 chromadb/chroma
```

### Combined Constraints

**Qdrant (2GB RAM, 2 CPUs):**
```bash
docker run --memory="2g" --cpus="2.0" -p 6333:6333 qdrant/qdrant
```

**ChromaDB (1GB RAM, 1 CPU):**
```bash
docker run --memory="1g" --cpus="1.0" -v ./chroma-data:/data -p 8000:8000 chromadb/chroma
```

Then run your benchmarks normally:
```bash
python benchmark/main.py --db qdrant --dataset sift-128
```

## Output

Benchmarks produce JSON results containing:

```json
{
  "vectors": 1000000,
  "dim": 128,
  "queries": 10000,
  "params": {
    "M": 16,
    "ef_construction": 200,
    "ef_search": 64,
    "seed": 123
  },
  "build_latency": 245.678,
  "search_avg_latency": 0.00234,
  "qps": 427.35,
  "search_wall_time": 23.456,
  "recall@10": 0.9456
}
```

**Metrics explained:**
- `build_latency`: Time to build the index (seconds)
- `search_avg_latency`: Average time per query (seconds)
- `qps`: Queries per second
- `search_wall_time`: Total time for all queries (seconds)
- `recall@10`: Proportion of true neighbors found in top-10 results


## Troubleshooting

**Memory issues during benchmarking:**
- Reduce `--max-queries` parameter
- Use smaller datasets (mnist-784 or fashion-mnist-784)
- Increase Docker memory allocation
