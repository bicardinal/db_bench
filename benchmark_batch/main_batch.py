import os
import sys
import time
import json
import math
import argparse
import re
import threading
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import h5py
import psutil
import requests

from aux.brinicle_client import VectorEngineClient
from aux.memory_inspect import CgroupMemoryMonitor

import pymilvus

import chromadb

import weaviate
import weaviate.classes.config as wvc
from weaviate.util import generate_uuid5

from qdrant_client.models import Distance, VectorParams, HnswConfigDiff, SearchParams
from qdrant_client import QdrantClient, models


DATASETS = {
    "sift-128": {
        "url": "https://ann-benchmarks.com/sift-128-euclidean.hdf5",
        "filename": "sift-128-euclidean.hdf5"
    },
    "fashion-mnist-784": {
        "url": "http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5",
        "filename": "fashion-mnist-784-euclidean.hdf5"
    },
    "mnist-784": {
        "url": "http://ann-benchmarks.com/mnist-784-euclidean.hdf5",
        "filename": "mnist-784-euclidean.hdf5"
    },
    "gist-960": {
        "url": "https://ann-benchmarks.com/gist-960-euclidean.hdf5",
        "filename": "gist-960-euclidean.hdf5"
    }
}

container_names = {
    "milvus": "milvus-standalone",
    "brinicle": "brinicle_container"
}


DEFAULT_DATA_DIR = Path("./data")
K = 10


def download_if_needed(url: str, dst_path: Path):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists():
        print(f"[download] Found existing: {dst_path}")
        return
    print(f"[download] Downloading {url} -> {dst_path}")
    tmp = dst_path.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dst_path)
    print(f"[download] Done: {dst_path}")


def load_arrays(h5_path: Path):
    with h5py.File(h5_path, "r") as src:
        X = np.array(src["train"], dtype=np.float32)
        Q = np.array(src["test"], dtype=np.float32)
        GT = np.array(src["neighbors"], dtype=np.int32)
    return X, Q, GT


def compute_recalls(pred_ids: np.ndarray, gt_top100: np.ndarray, K: int):
    nq = gt_top100.shape[0]
    out = {}
    hits = 0
    for i in range(nq):
        a = pred_ids[i, :K]
        b = gt_top100[i, :K]
        hits += len(set(a.tolist()) & set(b.tolist()))
    out[f"recall@{K}"] = hits / (nq * K)
    return out


def thread_local_factory(make):

    local = threading.local()

    def get():
        client = getattr(local, "client", None)
        if client is None:
            client = make()
            local.client = client
        return client

    return get


def prepare_queries(Q, args):
    nq_total = min(args.max_queries, Q.shape[0])
    rng = np.random.default_rng(args.seed)
    idxs = np.arange(Q.shape[0])
    if args.sample:
        idxs = rng.choice(idxs, size=nq_total, replace=False)
    else:
        idxs = idxs[:nq_total]
    queries = np.ascontiguousarray(Q[idxs], dtype=np.float32)
    return queries, idxs


def chunk_queries(queries, B):
    n = queries.shape[0]
    return [(s, queries[s:s + B]) for s in range(0, n, B)]


def normalize_row(ids, K):
    row = [int(x) for x in ids[:K]]
    if len(row) < K:
        row = row + [-1] * (K - len(row))
    return row


def _warmup(ex, search_batch_fn, sample_chunk, C):

    barrier = threading.Barrier(C)

    def w():
        try:
            barrier.wait(timeout=60)
        except Exception:
            pass
        try:
            search_batch_fn(sample_chunk)
        except Exception:
            pass

    futures = [ex.submit(w) for _ in range(C)]
    for f in futures:
        try:
            f.result(timeout=180)
        except Exception:
            pass


def run_search_harness(search_batch_fn, queries, gt_subset, args, build_latency, N, dim):

    B = max(1, args.batch_size)
    C = max(1, args.concurrency)
    nq_total = queries.shape[0]
    chunks = chunk_queries(queries, B)
    preds = np.full((nq_total, K), -1, dtype=np.int64)

    def worker(start, chunk):
        q0 = time.perf_counter()
        rows = search_batch_fn(chunk)
        q1 = time.perf_counter()
        ids_2d = np.asarray([normalize_row(r, K) for r in rows], dtype=np.int64)
        return start, ids_2d, (q1 - q0)

    per_batch_latencies = []
    collected = []
    with ThreadPoolExecutor(max_workers=C) as ex:
        _warmup(ex, search_batch_fn, chunks[0][1], C)

        t0 = time.perf_counter()
        futures = [ex.submit(worker, s, c) for (s, c) in chunks]
        for fut in as_completed(futures):
            start, ids_2d, lat = fut.result()
            per_batch_latencies.append(lat)
            collected.append((start, ids_2d))
        t1 = time.perf_counter()

    for start, ids_2d in collected:
        preds[start:start + ids_2d.shape[0], :] = ids_2d

    search_wall = t1 - t0
    lat = np.asarray(per_batch_latencies, dtype=np.float64)
    qps = nq_total / search_wall if search_wall > 0 else float("inf")

    recalls = compute_recalls(preds, gt_subset, K)

    results = {
        "vectors": N,
        "dim": dim,
        "queries": int(nq_total),
        "params": {"M": args.m, "ef_construction": args.efc, "ef_search": args.efs, "seed": args.seed},
        "build_latency": build_latency,
        "search_avg_latency": float(np.mean(lat)),
        "search_p50_latency": float(np.percentile(lat, 50)),
        "search_p95_latency": float(np.percentile(lat, 95)),
        "search_p99_latency": float(np.percentile(lat, 99)),
        "qps": qps,
        "search_wall_time": search_wall,
        "batch_size": B,
        "concurrency": C,
        "latency_granularity": "per_query" if B == 1 else "per_request_batch",
    }
    results.update(recalls)

    return results


def wait_qdrant_green(client, name, poll_s=0.5, timeout_s=1800):

    t0 = time.perf_counter()
    prev = None
    while True:
        info = client.get_collection(collection_name=name)
        status = str(getattr(info, "status", "")).lower()
        indexed = getattr(info, "indexed_vectors_count", None) or 0
        green = status.endswith("green")
        if green and prev is not None and indexed == prev:
            return info
        prev = indexed if green else None
        if time.perf_counter() - t0 > timeout_s:
            print(f"[qdrant] wait_green: timeout (status={status}, indexed={indexed})")
            return info
        time.sleep(poll_s)


def run_benchmark_qdrant_build(X, args):
    """
    $ docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
    """
    args.efs = max(K, args.efs)
    storage = "on-disk" if args.qdrant_on_disk else "in-memory"
    print(f"[build] Creating QdrantANN(M={args.m}, efc={args.efc}, efs={args.efs}) [gRPC, float32, {storage}]")
    client = QdrantClient(host="0.0.0.0", grpc_port=6334, prefer_grpc=True)
    if client.collection_exists(collection_name="bench"):
        client.delete_collection(collection_name="bench")

    client.create_collection(
        collection_name="bench",
        vectors_config=VectorParams(
            size=X.shape[1], distance=Distance.EUCLID, on_disk=args.qdrant_on_disk
        ),
        hnsw_config=HnswConfigDiff(m=args.m, ef_construct=args.efc, on_disk=args.qdrant_on_disk),
        optimizers_config=models.OptimizersConfigDiff(
            default_segment_number=max(1, args.qdrant_segments),
        ),
    )
    ids = list(range(X.shape[0]))
    t0 = time.perf_counter()
    client.upload_collection(
        collection_name="bench",
        vectors=X,
        ids=ids,
        parallel=8,
        batch_size=256,
        wait=False,
    )
    wait_qdrant_green(client, "bench")
    build_latency = time.perf_counter() - t0
    print(f"[build] Done in {build_latency:.3f}s (green/indexed).")
    if args.delete:
        to_remove = ids[int(0.9 * len(ids)):]
        client.delete(
            collection_name="bench",
            points_selector=to_remove,
        )
        print(f"[delete] Deleted 10% of data from the index")

    return build_latency


def make_qdrant_search_fn(args, N, dim):
    efs = max(K, args.efs)
    GRPC_CHUNK = 1024

    def make():
        return QdrantClient(host="0.0.0.0", grpc_port=6334, prefer_grpc=True)

    get_client = thread_local_factory(make)
    setup = make()
    params = models.SearchParams(hnsw_ef=efs, exact=False)

    def search_batch_fn(chunk):
        client = get_client()
        reqs = [
            models.QueryRequest(
                query=v.tolist(),
                limit=K,
                params=params,
                with_payload=False,
                with_vector=False,
            )
            for v in chunk
        ]
        out = []
        for i in range(0, len(reqs), GRPC_CHUNK):
            resp = client.query_batch_points(collection_name="bench", requests=reqs[i:i + GRPC_CHUNK])
            for r in resp:
                out.append([int(p.id) for p in r.points])
        return out

    def teardown():
        try:
            setup.close()
        except Exception:
            pass

    return search_batch_fn, teardown


def run_benchmark_brinicle_build(X, args):
    """
    $ git clone https://github.com/bicardinal/brinicle.git
    $ cd brinicle
    $ bash build.sh
    $ make docker-build
    $ make docker-run
    """
    args.efs = max(K, args.efs)
    print(f"[build] Creating Brinicle(M={args.m}, efc={args.efc}, efs={args.efs})")
    idx_name = "brinicle_bench"
    batch_size = 32768
    client = VectorEngineClient()
    client.create_index(
        index_name=idx_name,
        M=args.m,
        dim=X.shape[1],
        ef_construction=args.efc,
        ef_search=args.efs,
        seed=args.seed,
    )
    ids = [str(x) for x in range(X.shape[0])]
    t0 = time.perf_counter()
    client.init(idx_name, mode="build")
    for b in range(0, X.shape[0], batch_size):
        client.ingest_batch_binary(
            index_name=idx_name,
            ids=ids[b:b + batch_size],
            vectors=X[b:b + batch_size],
        ),
        print("batch idx:", b, end='\r')
    client.finalize(idx_name)
    build_latency = time.perf_counter() - t0
    print(f"[build] Done in {build_latency:.3f}s.")
    if args.delete:
        to_remove = ids[int(0.9 * len(ids)):]
        client.delete_items(
            index_name=idx_name,
            external_ids=to_remove,
        )
        print(f"[delete] Deleted 10% of data from the index")
    return build_latency


def make_brinicle_search_fn(args, N, dim):
    efs = max(K, args.efs)
    idx_name = "brinicle_bench"
    n_jobs = max(1, args.brinicle_njobs)

    setup = VectorEngineClient()
    get_client = thread_local_factory(VectorEngineClient)
    state = {"batch_ok": True, "cap": (args.brinicle_max_batch or None)}

    def _per_query(chunk):
        client = get_client()
        out = []
        for v in chunk:
            labels = client.search(idx_name, v, k=K, efs=efs)
            out.append([int(x) for x in labels])
        return out

    def _server_call(client, sub):
        rows = client.search_batch(idx_name, sub, k=K, efs=efs, n_jobs=n_jobs)
        return [[int(x) for x in row] for row in rows]

    def _batch(chunk):
        client = get_client()
        out = []
        i, n = 0, len(chunk)
        while i < n:
            cap = state["cap"]
            size = (n - i) if not cap else min(cap, n - i)
            sub = chunk[i:i + size]
            try:
                out.extend(_server_call(client, sub))
                i += size
            except requests.exceptions.HTTPError as e:
                resp = e.response
                code = getattr(resp, "status_code", None)
                detail = ""
                try:
                    detail = resp.json().get("detail", "")
                except Exception:
                    detail = getattr(resp, "text", "") or ""
                if code == 400 and "Max allowed batch size" in detail:
                    m = re.search(r"Max allowed batch size is\s+(\d+)", detail)
                    newcap = int(m.group(1)) if m else max(1, size // 2)
                    if newcap >= size:
                        newcap = max(1, size // 2)
                    state["cap"] = newcap
                    print(f"[brinicle] server batch cap = {newcap}; sub-chunking")
                    continue  # retry this slice with the smaller cap
                raise
        return out

    def search_batch_fn(chunk):
        if state["batch_ok"]:
            try:
                return _batch(chunk)
            except requests.exceptions.HTTPError as e:
                code = getattr(e.response, "status_code", None)
                if code in (404, 405):
                    print(f"[brinicle] /search/batch.bin unavailable (HTTP {code}); "
                          f"falling back to concurrent single search")
                    state["batch_ok"] = False
                else:
                    raise
            except Exception as e:
                print(f"[brinicle] batch search failed ({e}); falling back to single search")
                state["batch_ok"] = False
        return _per_query(chunk)

    def teardown():
        try:
            setup.close()
        except Exception:
            pass

    return search_batch_fn, teardown



def run_benchmark_chroma_build(X, args):
    """
    $ docker run -v ./chroma-data:/data -p 8000:8000 chromadb/chroma
    """
    args.efs = max(K, args.efs)
    print(f"[build] Creating ChromaDB(M={args.m}, efc={args.efc}, efs={args.efs})")
    client = chromadb.HttpClient(host="localhost", port=8000)
    try:
        client.delete_collection(name="bench")
    except:
        pass
    collection = client.create_collection(
        name="bench",
        metadata={
            "hnsw:space": "l2",
            "hnsw:M": args.m,
            "hnsw:construction_ef": args.efc,
            "hnsw:search_ef": args.efs,
        }
    )
    ids = [str(i) for i in range(X.shape[0])]
    batch_size = 4096
    embeddings = X.tolist()
    t0 = time.perf_counter()

    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i:i + batch_size]
        batch_embeddings = embeddings[i:i + batch_size]
        collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings
        )

    build_latency = time.perf_counter() - t0
    print(f"[build] Done in {build_latency:.3f}s.")
    if args.delete:
        to_remove = ids[int(0.9 * len(ids)):]
        collection.delete(ids=to_remove)
        print(f"[delete] Deleted 10% of data from the index")
    return build_latency


def make_chroma_search_fn(args, N, dim):
    def make():
        client = chromadb.HttpClient(host="localhost", port=8000)
        return client.get_collection("bench")

    get_collection = thread_local_factory(make)

    def search_batch_fn(chunk):
        collection = get_collection()
        res = collection.query(query_embeddings=chunk.tolist(), n_results=K)
        # res['ids'] is a list-of-lists, one row per query; ids are strings.
        return [[int(x) for x in row] for row in res["ids"]]

    def teardown():
        pass

    return search_batch_fn, teardown


def run_benchmark_weaviate_build(X, args):
    """
    docker compose up -d
    """
    args.efs = max(K, args.efs)
    print(f"[build] Creating Weaviate(M={args.m}, efc={args.efc}, efs={args.efs})")
    client = weaviate.connect_to_local()
    n = X.shape[0]
    X = X.tolist()
    try:
        if client.collections.exists("Bench"):
            client.collections.delete("Bench")

        collection = client.collections.create(
            name="Bench",
            vector_index_config=wvc.Configure.VectorIndex.hnsw(
                distance_metric=wvc.VectorDistances.L2_SQUARED,
                max_connections=args.m,
                ef_construction=args.efc,
                ef=args.efs,
            ),
            properties=[
                wvc.Property(name="original_id", data_type=wvc.DataType.INT)
            ]
        )

        t0 = time.perf_counter()

        with collection.batch.fixed_size(batch_size=4096) as batch:
            for i in range(n):
                batch.add_object(
                    properties={"original_id": i},
                    vector=X[i],
                    uuid=generate_uuid5(i)
                )

        build_latency = time.perf_counter() - t0
        print(f"[build] Done in {build_latency:.3f}s.")
        if args.delete:
            delete_start = int(0.9 * n)
            for i in range(delete_start, n):
                collection.data.delete_by_id(uuid=generate_uuid5(i))
            print(f"[delete] Deleted 10% of data from the index")
        return build_latency
    finally:
        client.close()


def make_weaviate_search_fn(args, N, dim):
    client = weaviate.connect_to_local()
    collection = client.collections.get("Bench")

    def search_batch_fn(chunk):
        out = []
        for v in chunk:
            response = collection.query.near_vector(
                near_vector=v.tolist(),
                limit=K,
                return_properties=["original_id"],
            )
            out.append([int(o.properties["original_id"]) for o in response.objects])
        return out

    def teardown():
        try:
            client.close()
        except Exception:
            pass

    return search_batch_fn, teardown


def run_benchmark_milvus_build(X, args):
    """
        $ curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh
        $ bash standalone_embed.sh start
    """
    collection_name = "bench_milvus"
    dim = X.shape[1]
    pymilvus.connections.connect("default", host="localhost", port="19530")
    client = pymilvus.MilvusClient(
        uri="http://localhost:19530",
    )
    if pymilvus.utility.has_collection(collection_name):
        pymilvus.utility.drop_collection(collection_name)
    fields = [
        pymilvus.FieldSchema(name="id", dtype=pymilvus.DataType.INT64, is_primary=True, auto_id=False),
        pymilvus.FieldSchema(name="embedding", dtype=pymilvus.DataType.FLOAT_VECTOR, dim=dim, mmap_enabled=True)
    ]
    schema = pymilvus.CollectionSchema(fields, description="Benchmark collection")
    client.create_collection(collection_name=collection_name, schema=schema, properties={"mmap.enabled": "true"})
    collection = pymilvus.Collection(collection_name)
    print(f"[build] Inserting {X.shape[0]} vectors...")
    t0 = time.perf_counter()
    batch_size = 5000
    for i in range(0, X.shape[0], batch_size):
        end = min(i + batch_size, X.shape[0])
        ids = list(range(i, end))
        embeddings = X[i:end].tolist()
        collection.insert([ids, embeddings])
    print(f"[build] Creating HNSW index (M={args.m}, efc={args.efc})...")
    index_params = {
        "metric_type": "L2",
        "index_type": "HNSW",
        "params": {"M": args.m, "efConstruction": args.efc, "mmap.enabled": "true"}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    collection.load()
    build_latency = time.perf_counter() - t0
    print(f"[build] Done in {build_latency:.3f}s.")

    if args.delete:
        delete_start = int(0.9 * X.shape[0])
        ids_to_remove = list(range(delete_start, X.shape[0]))
        expr = f"id in {ids_to_remove}"
        collection.delete(expr)
        print(f"[delete] Deleted 10% of data from the index")

    return build_latency


def make_milvus_search_fn(args, N, dim):
    efs = max(K, args.efs)
    pymilvus.connections.connect("default", host="localhost", port="19530")
    collection = pymilvus.Collection("bench_milvus")
    collection.load()
    search_params = {"metric_type": "L2", "params": {"ef": efs}}

    def search_batch_fn(chunk):
        # data=[q1..qn] is the native nq batch; res[i] are hits for query i.
        res = collection.search(
            data=chunk.tolist(),
            anns_field="embedding",
            param=search_params,
            limit=K,
        )
        return [[int(h.id) for h in res[i]] for i in range(len(res))]

    def teardown():
        try:
            collection.release()
        except Exception:
            pass

    return search_batch_fn, teardown


ENGINES = {
    "qdrant":   (run_benchmark_qdrant_build,   make_qdrant_search_fn),
    "brinicle": (run_benchmark_brinicle_build, make_brinicle_search_fn),
    "chroma":   (run_benchmark_chroma_build,   make_chroma_search_fn),
    "weaviate": (run_benchmark_weaviate_build, make_weaviate_search_fn),
    "milvus":   (run_benchmark_milvus_build,   make_milvus_search_fn),
}


def start_monitor(container_name):
    try:
        return CgroupMemoryMonitor(container_name=container_name, interval_s=0.01).start()
    except Exception as e:
        print(f"[mem] monitor unavailable for '{container_name}': {e}")
        return None


AVG_KEYS = [
    "search_avg_latency", "search_p50_latency", "search_p95_latency",
    "search_p99_latency", "qps", "search_wall_time", "recall@10",
]


def main():
    process = psutil.Process(os.getpid())
    p = argparse.ArgumentParser(description="Batch/concurrent benchmark of vector databases on ANN datasets")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--dataset", type=str, default="sift-128",
                   choices=list(DATASETS.keys()),
                   help="Dataset to use for benchmarking")
    p.add_argument("--db", type=str, default="brinicle", choices=list(ENGINES.keys()))
    p.add_argument("--m", type=int, default=16)
    p.add_argument("--efc", type=int, default=200, help="ef_construction")
    p.add_argument("--efs", type=int, default=64, help="ef_search")
    p.add_argument("--max-queries", type=int, default=10000)
    p.add_argument("--sample", action="store_true", help="randomly sample queries instead of first N")
    p.add_argument("--delete", action="store_true", help="Intentionally remove 10 percent of the data tail after build")
    p.add_argument("--seed", type=int, default=123)
    # Batch-mode dials
    p.add_argument("--batch-size", type=int, default=1,
                   help="Queries per request. 1 = one query/request; >1 = native multi-query batch where supported")
    p.add_argument("--concurrency", type=int, default=1,
                   help="Parallel client workers (ThreadPoolExecutor) dispatching requests")
    p.add_argument("--qdrant-segments", type=int, default=1, help="Target number of Qdrant segments")
    p.add_argument("--qdrant-on-disk", action="store_true",
                   help="Store Qdrant vectors + HNSW on disk (memmap) for a disk-vs-disk comparison with brinicle")
    p.add_argument("--brinicle-njobs", type=int, default=8,
                   help="brinicle server-side batch parallelism (n_jobs for search_batch)")
    p.add_argument("--brinicle-max-batch", type=int, default=0,
                   help="Max queries per brinicle /search/batch.bin request (0 = auto-discover server cap)")
    p.add_argument("--trials", type=int, default=10, help="Search repetitions to average")
    args = p.parse_args()

    dataset_config = DATASETS[args.dataset]
    data_dir = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    h5_path = data_dir / dataset_config["filename"]
    download_if_needed(dataset_config["url"], h5_path)

    with h5py.File(h5_path, "r") as src:
        print("[hdf5] keys:", list(src.keys()))
        for k in ["train", "test", "neighbors"]:
            ds = src[k]
            print(f"[hdf5] {k}: shape={ds.shape}, dtype={ds.dtype}")

    print("[load] Loading arrays into memory...")
    X, Q, GT = load_arrays(h5_path)
    print(f"[load] X: {X.shape} float32, Q: {Q.shape} float32, GT: {GT.shape} int32")
    print("[db] ", args.db)
    print(f"[mode] batch_size={args.batch_size} concurrency={args.concurrency} trials={args.trials}")

    if args.delete:
        print(f"[delete] Checking the delete scenario")

    build_fn, make_search_fn = ENGINES[args.db]
    container_name = container_names.get(args.db, f"{args.db}_bench")

    mon = start_monitor(container_name)
    build_latency = build_fn(X, args)
    build_peak_mb = None
    build_mem_report = None
    if mon:
        mon.stop()
        build_peak_mb = mon.peak_bytes / (1024 * 1024)
        build_mem_report = mon.peak_report_mb()

    N, dim = X.shape[0], X.shape[1]
    del X

    queries, idxs = prepare_queries(Q, args)
    gt_subset = GT[idxs]

    search_batch_fn, teardown = make_search_fn(args, N, dim)
    batch_results = None
    search_peak_sum = 0.0
    search_peak_count = 0
    search_report_sum = None
    try:
        for trial in range(args.trials):
            mon = start_monitor(container_name)
            results = run_search_harness(search_batch_fn, queries, gt_subset, args, build_latency, N, dim)
            if mon:
                mon.stop()
                search_peak_sum += mon.peak_bytes / (1024 * 1024)
                search_peak_count += 1
                rep = mon.peak_report_mb()
                if search_report_sum is None:
                    search_report_sum = dict(rep)
                else:
                    for rk in search_report_sum:
                        search_report_sum[rk] += rep[rk]

            if batch_results:
                for key in AVG_KEYS:
                    batch_results[key] += results[key]
            else:
                batch_results = results
            print(f"[trial {trial + 1}/{args.trials}] qps={results['qps']:.1f} "
                  f"recall@10={results['recall@10']:.5f} wall={results['search_wall_time']:.3f}s")
    finally:
        teardown()

    # Average the accumulated metrics.
    batch_results["build_mem_peak_mb"] = build_peak_mb
    batch_results["build_mem_report_mb"] = build_mem_report
    batch_results["search_mem_peak_mb_avg"] = (search_peak_sum / search_peak_count) if search_peak_count else None
    batch_results["search_mem_report_mb_avg"] = (
        {rk: v / search_peak_count for rk, v in search_report_sum.items()} if search_report_sum else None
    )
    for key in AVG_KEYS:
        batch_results[key] /= args.trials

    output_dir = Path("benchmark_batch/batch_results" if not args.delete else "benchmark_batch/batch_results_delete")
    output_dir.mkdir(parents=True, exist_ok=True)

    base_filename = (
        f"dbs_{args.db}_{args.dataset}_{args.m}m_{args.efc}efc_{args.efs}efs"
        f"_b{args.batch_size}_c{args.concurrency}"
    )
    json_path = output_dir / f"{base_filename}.json"

    output_data = {
        "database": args.db,
        "dataset": args.dataset,
        "m": args.m,
        "ef_search": args.efs,
        "ef_construction": args.efc,
        "batch_size": args.batch_size,
        "concurrency": args.concurrency,
        "qdrant_on_disk": args.qdrant_on_disk,
        "build_latency": build_latency,
        "build_mem_peak_mb": build_peak_mb,
        "build_mem_report_mb": build_mem_report,
        "results": batch_results
    }

    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=4)
    print(f"\n[save] Results saved to {json_path}")

    print(json.dumps(output_data, indent=4))


if __name__ == "__main__":
    main()
