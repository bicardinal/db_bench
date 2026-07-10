
import os
import time
import json
import argparse
import urllib.request
from pathlib import Path

import numpy as np
import h5py

import hnswlib
import faiss

try:
    from _brinicle import VectorEngine
except ModuleNotFoundError:
    VectorEngine = None
    print("[warning] brinicle lib (_brinicle) is not available for benchmarking. Refer to README.md.")


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


def make_faiss(X, args):
    dim = X.shape[1]
    efs = max(K, args.efs)
    print(f"[build] FAISS HNSW(M={args.m}, efc={args.efc}, efs={efs}) search_threads={args.search_threads}, build_threads={args.build_threads}")
    index = faiss.IndexHNSWFlat(dim, args.m, faiss.METRIC_L2)
    index.hnsw.efConstruction = args.efc
    faiss.omp_set_num_threads(args.build_threads)
    t0 = time.perf_counter()
    index.add(X.astype(np.float32))
    build_latency = time.perf_counter() - t0
    index.hnsw.efSearch = efs
    faiss.omp_set_num_threads(args.search_threads)

    def search_fn(chunk):
        _, labels = index.search(np.ascontiguousarray(chunk, dtype=np.float32), K)
        return labels.astype(np.int64)

    return search_fn, build_latency


def make_hnswlib(X, args):
    n, dim = X.shape
    efs = max(K, args.efs)

    index = hnswlib.Index(space="l2", dim=dim)
    index.init_index(
        max_elements=n,
        ef_construction=args.efc,
        M=args.m,
        random_seed=args.seed,
    )

    ids = np.arange(n, dtype=np.int64)

    t0 = time.perf_counter()
    index.add_items(
        X,
        ids,
        num_threads=args.build_threads,
    )
    build_latency = time.perf_counter() - t0

    index.set_ef(efs)

    def search_fn(chunk):
        labels, _ = index.knn_query(
            chunk,
            k=K,
            num_threads=args.search_threads,
        )
        return labels.astype(np.int64, copy=False)

    return search_fn, build_latency


def make_brinicle(X, args):
    if VectorEngine is None:
        raise RuntimeError("brinicle (_brinicle) is not available. See README.md.")
    dim = X.shape[1]
    efs = max(K, args.efs)
    print(f"[build] brinicle VectorEngine(M={args.m}, efc={args.efc}, efs={efs}) n_jobs={args.search_threads}")
    engine = VectorEngine(
        "batch_embed_index", dim, 0.1,
        M=args.m, ef_construction=args.efc, ef_search=efs, seed=args.seed, build_n_threads=args.build_threads,
    )
    ids = [str(x) for x in range(X.shape[0])]
    t0 = time.perf_counter()
    engine.init(mode="build")
    for b in range(X.shape[0]):
        engine.ingest(external_id=ids[b], vec=X[b])
    engine.finalize()
    build_latency = time.perf_counter() - t0

    def search_fn(chunk):
        rows = engine.search_batch(
            np.ascontiguousarray(chunk, dtype=np.float32), k=K, efs=efs, n_jobs=args.search_threads,
        )
        out = np.full((len(rows), K), -1, dtype=np.int64)
        for i, r in enumerate(rows):
            r = [int(x) for x in r[:K]]
            out[i, :len(r)] = r
        return out

    return search_fn, build_latency


ENGINES = {
    "faiss": make_faiss,
    "hnswlib": make_hnswlib,
    "brinicle": make_brinicle,
}


# ---------------------------------------------------------------------------
# Shared batch-search harness
# ---------------------------------------------------------------------------

def run_batch_search(search_fn, queries, gt_subset, args, build_latency, N, dim):
    nq_total = queries.shape[0]
    B = args.batch_size if args.batch_size and args.batch_size > 0 else nq_total
    preds = np.full((nq_total, K), -1, dtype=np.int64)

    # Untimed warmup (JIT/thread-pool spin-up, page-in).
    search_fn(queries[:min(nq_total, max(8, B))])

    per_batch_latencies = []
    t0 = time.perf_counter()
    for s in range(0, nq_total, B):
        chunk = queries[s:s + B]
        q0 = time.perf_counter()
        ids_2d = search_fn(chunk)
        q1 = time.perf_counter()
        per_batch_latencies.append(q1 - q0)
        preds[s:s + ids_2d.shape[0], :] = ids_2d[:, :K]
    t1 = time.perf_counter()

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
        "search_threads": args.search_threads,
        "build_threads": args.build_threads,
        "latency_granularity": "per_query" if B == 1 else "per_request_batch",
    }
    results.update(recalls)
    return results


AVG_KEYS = [
    "search_avg_latency", "search_p50_latency", "search_p95_latency",
    "search_p99_latency", "qps", "search_wall_time", "recall@10",
]


def main():
    p = argparse.ArgumentParser(description="Batch in-process ANN benchmark (FAISS / hnswlib / brinicle)")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--dataset", type=str, default="sift-128", choices=list(DATASETS.keys()))
    p.add_argument("--engine", type=str, default="brinicle", choices=list(ENGINES.keys()))
    p.add_argument("--m", type=int, default=16)
    p.add_argument("--efc", type=int, default=200, help="ef_construction")
    p.add_argument("--efs", type=int, default=64, help="ef_search")
    p.add_argument("--max-queries", type=int, default=10000)
    p.add_argument("--sample", action="store_true", help="randomly sample queries instead of first N")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--batch-size", type=int, default=1024,
                   help="Queries per native search call (0 = all queries in one call)")
    p.add_argument("--trials", type=int, default=10, help="Search repetitions to average")

    p.add_argument(
        "--build-threads",
        type=int,
        default=16,
        help="CPU threads used for index construction",
    )
    p.add_argument(
        "--search-threads",
        type=int,
        default=8,
        help="CPU threads used for batch search",
    )
    args = p.parse_args()

    dataset_config = DATASETS[args.dataset]
    args.data_dir.mkdir(parents=True, exist_ok=True)
    h5_path = args.data_dir / dataset_config["filename"]
    download_if_needed(dataset_config["url"], h5_path)

    print("[load] Loading arrays into memory...")
    X, Q, GT = load_arrays(h5_path)
    print(f"[load] X: {X.shape} float32, Q: {Q.shape} float32, GT: {GT.shape} int32")
    print(f"[engine] {args.engine}  [mode] batch_size={args.batch_size or 'all'} search_threads={args.search_threads} trials={args.trials}")

    make_engine = ENGINES[args.engine]
    search_fn, build_latency = make_engine(X, args)

    N, dim = X.shape[0], X.shape[1]
    del X

    queries, idxs = prepare_queries(Q, args)
    gt_subset = GT[idxs]

    batch_results = None
    for trial in range(args.trials):
        results = run_batch_search(search_fn, queries, gt_subset, args, build_latency, N, dim)
        if batch_results:
            for key in AVG_KEYS:
                batch_results[key] += results[key]
        else:
            batch_results = results
        print(f"[trial {trial + 1}/{args.trials}] qps={results['qps']:.1f} "
              f"recall@10={results['recall@10']:.5f} wall={results['search_wall_time']:.3f}s")

    for key in AVG_KEYS:
        batch_results[key] /= args.trials

    output_dir = Path("benchmark_batch/batch_embed_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    base_filename = (
        f"engine_{args.engine}_{args.dataset}_{args.m}m_{args.efc}efc_{args.efs}efs"
        f"_b{args.batch_size}_t{args.search_threads}_bt{args.build_threads}"
    )
    json_path = output_dir / f"{base_filename}.json"

    output_data = {
        "database": args.engine,
        "engine": args.engine,
        "dataset": args.dataset,
        "m": args.m,
        "ef_search": args.efs,
        "ef_construction": args.efc,
        "batch_size": args.batch_size,
        "search_threads": args.search_threads,
        "build_threads": args.build_threads,
        "build_latency": build_latency,
        "results": batch_results,
    }

    with open(json_path, "w") as f:
        json.dump(output_data, f, indent=4)
    print(f"\n[save] Results saved to {json_path}")
    print(json.dumps(output_data, indent=4))


if __name__ == "__main__":
    main()