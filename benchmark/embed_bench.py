import os
import sys
import time
import json
import math
import argparse
import urllib.request
from pathlib import Path
import numpy as np
import h5py
import psutil
import requests


import lancedb

import hnswlib

import faiss

try:
	from _brinicle import VectorEngine
except ModuleNotFoundError:
	print("[warning] brinicle lib is not available for benchmarking. Refer to README.md.")


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

DEFAULT_DATA_DIR = Path("./benchmark/data")

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


def run_benchmark_brinicle_build(X, args):
	args.efs = max(K, args.efs)
	print(f"[build] Creating Brinicle(M={args.m}, efc={args.efc}, efs={args.efs})")
	idx_name = "brinicle_bench_sift"
	batch_size = 8192
	client = VectorEngine("sift_index", X.shape[1], 0.1, M=args.m, ef_construction=args.efc, ef_search=args.efs, seed=args.seed)
	ids = [str(x) for x in range(X.shape[0])]
	t0 = time.perf_counter()
	client.init(mode="build")
	for b in range(X.shape[0]):
		client.ingest(
			external_id=ids[b],
			vec=X[b],
		)
		print(b, end='\r')
	client.finalize()
	build_latency = time.perf_counter() - t0
	print(f"[build] Done in {build_latency:.3f}s.")
	return build_latency

def run_benchmark_brinicle_search(Q, GT, args, build_latency, N, dim):
	print(f"[search] K={K}")
	index = "brinicle_bench_sift"
	client = VectorEngine("sift_index", Q.shape[1], 0.1)
	# Queries
	nq_total = min(args.max_queries, Q.shape[0])
	rng = np.random.default_rng(args.seed)
	idxs = np.arange(Q.shape[0])
	if args.sample:
		idxs = rng.choice(idxs, size=nq_total, replace=False)
	else:
		idxs = idxs[:nq_total]
	queries = Q[idxs]

	preds = np.empty((nq_total, K), dtype=np.int64)
	# warm up
	labels = client.search(queries[0], k=K, efs=args.efs)
	print(f"[search] Running {nq_total} queries @ top-{K}...")
	t1 = time.perf_counter()
	query_latencies = []
	for i, v in enumerate(queries, start=0):
		q0 = time.perf_counter()
		labels = client.search(v, k=K, efs=args.efs)  # must return list/iterable of IDs
		q1 = time.perf_counter()
		query_latencies.append(q1 - q0)
		labels = [int(x) for x in labels] # back to int so that we can calculate recall
		if len(labels) < K:
			print("returned results is less than K")
			labels = list(labels) + [-1] * (K - len(labels))
		preds[i, :] = np.asarray(labels[:K], dtype=np.int64)
		print(i, end='\r')
	t2 = time.perf_counter()

	search_wall = t2 - t1
	query_latencies = np.array(query_latencies)
	avg_latency = np.mean(query_latencies)
	p50_latency = np.percentile(query_latencies, 50)
	p95_latency = np.percentile(query_latencies, 95)
	p99_latency = np.percentile(query_latencies, 99)
	total_q_time = np.sum(query_latencies)
	qps = nq_total / total_q_time if total_q_time > 0 else float("inf")

	recalls = compute_recalls(preds, GT[idxs], K)

	results = {
		"vectors": N,
		"dim": dim,
		"queries": int(nq_total),
		"params": {"M": args.m, "ef_construction": args.efc, "ef_search": args.efs, "seed": args.seed},
		"build_latency": build_latency,
		"search_avg_latency": avg_latency,
		"search_p50_latency": p50_latency,
		"search_p95_latency": p95_latency,
		"search_p99_latency": p99_latency,
		"qps": qps,                                   # queries per second (measured on per-call time)
		"search_wall_time": search_wall,              # total wall time (s) around search loop
	}
	results.update(recalls)

	return results


def run_benchmark_hnswlib_build(X, args):
	args.efs = max(K, args.efs)
	n = X.shape[0]
	dim = X.shape[1]
	print(f"[build] Creating HNSW(M={args.m}, efc={args.efc}, efs={args.efs})")
	hnsw_index = hnswlib.Index(space='l2', dim=dim)
	hnsw_index.init_index(max_elements=n, ef_construction=args.efc, M=args.m, random_seed=args.seed)
	ids = np.arange(n)
	t0 = time.perf_counter()
	hnsw_index.add_items(X, ids)
	build_latency = time.perf_counter() - t0
	hnsw_index.set_ef(args.efs)
	print(f"[build] Done in {build_latency:.3f}s.")
	return build_latency, hnsw_index


def run_benchmark_hnswlib_search(hnsw_index, Q, GT, args, build_latency, N, dim):
	print(f"[search] K={K}")
	# Queries
	nq_total = min(args.max_queries, Q.shape[0])
	rng = np.random.default_rng(args.seed)
	idxs = np.arange(Q.shape[0])
	if args.sample:
		idxs = rng.choice(idxs, size=nq_total, replace=False)
	else:
		idxs = idxs[:nq_total]
	queries = Q[idxs].tolist()

	preds = np.empty((nq_total, K), dtype=np.int64)
	# warm up

	hnsw_index.knn_query(queries[0], k=K)
	print(f"[search] Running {nq_total} queries @ top-{K}...")

	t1 = time.perf_counter()
	query_latencies = []
	for i, v in enumerate(queries, start=0):
		q0 = time.perf_counter()
		labels, _ = hnsw_index.knn_query(v, k=K)
		q1 = time.perf_counter()
		labels = labels[0]
		query_latencies.append(q1 - q0)
		if len(labels) < K:
			print("returned results is less than K")
			labels = list(labels) + [-1] * (K - len(labels))
		preds[i, :] = labels[:K]
		print(i, end='\r')
	t2 = time.perf_counter()

	search_wall = t2 - t1
	query_latencies = np.array(query_latencies)
	avg_latency = np.mean(query_latencies)
	p50_latency = np.percentile(query_latencies, 50)
	p95_latency = np.percentile(query_latencies, 95)
	p99_latency = np.percentile(query_latencies, 99)
	total_q_time = np.sum(query_latencies)
	qps = nq_total / total_q_time if total_q_time > 0 else float("inf")


	# Recalls
	recalls = compute_recalls(preds, GT[idxs], K)

	# Aggregate results
	results = {
		"vectors": N,
		"dim": dim,
		"queries": int(nq_total),
		"params": {"M": args.m, "ef_construction": args.efc, "ef_search": args.efs, "seed": args.seed},
		"build_latency": build_latency,
		"search_avg_latency": avg_latency,
		"search_p50_latency": p50_latency,
		"search_p95_latency": p95_latency,
		"search_p99_latency": p99_latency,
		"qps": qps,                                   # queries per second (measured on per-call time)
		"search_wall_time": search_wall,              # total wall time (s) around search loop
	}
	results.update(recalls)

	return results

def run_benchmark_faiss_build(X, args):
	args.efs = max(K, args.efs)
	n = X.shape[0]
	dim = X.shape[1]

	print(f"[build] Creating FAISS HNSW(M={args.m}, efc={args.efc}, efs={args.efs})")

	index = faiss.IndexHNSWFlat(dim, args.m)
	index.hnsw.efConstruction = args.efc

	t0 = time.perf_counter()
	index.add(X.astype(np.float32))
	build_latency = time.perf_counter() - t0

	index.hnsw.efSearch = args.efs

	print(f"[build] Done in {build_latency:.3f}s.")
	return index, build_latency


def run_benchmark_faiss_search(index, Q, GT, args, build_latency, N, dim):
	print(f"[search] K={K}")

	nq_total = min(args.max_queries, Q.shape[0])
	rng = np.random.default_rng(args.seed)
	idxs = np.arange(Q.shape[0])

	if args.sample:
		idxs = rng.choice(idxs, size=nq_total, replace=False)
	else:
		idxs = idxs[:nq_total]

	queries = Q[idxs].astype(np.float32)
	preds = np.empty((nq_total, K), dtype=np.int64)

	index.search(queries[0:1], K)

	print(f"[search] Running {nq_total} queries @ top-{K}...")
	t1 = time.perf_counter()
	query_latencies = []
	for i in range(nq_total):
		q0 = time.perf_counter()
		_, labels = index.search(queries[i:i+1], K)
		q1 = time.perf_counter()
		query_latencies.append(q1 - q0)
		labels = labels[0]
		if len(labels) < K:
			print("returned results is less than K")
			labels = list(labels) + [-1] * (K - len(labels))

		preds[i, :] = labels[:K]
		print(i, end='\r')

	t2 = time.perf_counter()

	search_wall = t2 - t1
	query_latencies = np.array(query_latencies)
	avg_latency = np.mean(query_latencies)
	p50_latency = np.percentile(query_latencies, 50)
	p95_latency = np.percentile(query_latencies, 95)
	p99_latency = np.percentile(query_latencies, 99)
	total_q_time = np.sum(query_latencies)
	qps = nq_total / total_q_time if total_q_time > 0 else float("inf")

	recalls = compute_recalls(preds, GT[idxs], K)

	results = {
		"vectors": N,
		"dim": dim,
		"queries": int(nq_total),
		"params": {"M": args.m, "ef_construction": args.efc, "ef_search": args.efs, "seed": args.seed},
		"build_latency": build_latency,
		"search_avg_latency": avg_latency,
		"search_p50_latency": p50_latency,
		"search_p95_latency": p95_latency,
		"search_p99_latency": p99_latency,
		"qps": qps,
		"search_wall_time": search_wall,
	}
	results.update(recalls)

	return results

def run_benchmark_lancedb_build(X, args):
	args.efs = max(K, args.efs)
	print(f"[build] Creating lance index(dim=128, M={args.m}, seed={args.seed}, efc={args.efc}, efs={args.efs})")
	db = lancedb.connect("./benchamrk/lancedb")
	t0 = time.perf_counter()
	data = [
		{
			"id": i,
			"text": f"Document {i}",
			"vector": X[i].tolist()
		}
		for i in range(X.shape[0])
	]
	table = db.create_table("lance_bench", data=data, mode="overwrite")
	table.create_index(
		metric="l2", num_partitions=int(math.sqrt(X.shape[0])),
		num_sub_vectors=1,
		index_type="IVF_HNSW_SQ",
	)
	build_latency = time.perf_counter() - t0
	print(f"[build] Done in {build_latency:.3f}s.")
	return build_latency


def run_benchmark_lancedb_search(Q, GT, args, build_latency, N, dim):
	print(f"[search] K={K}")
	db = lancedb.connect("./benchamrk/lancedb")
	table = db.open_table("lance_bench")
	# Queries
	nq_total = min(args.max_queries, Q.shape[0])
	rng = np.random.default_rng(args.seed)
	idxs = np.arange(Q.shape[0])
	if args.sample:
		idxs = rng.choice(idxs, size=nq_total, replace=False)
	else:
		idxs = idxs[:nq_total]
	queries = Q[idxs].tolist()

	preds = np.empty((nq_total, K), dtype=np.int64)

	res = table.search([queries[0], queries[1]]).limit(K).to_list()
	print(f"[search] Running {nq_total} queries @ top-{K}...")
	t1 = time.perf_counter()
	query_latencies = []
	for i, v in enumerate(queries, start=0):
		q0 = time.perf_counter()
		labels = table.search(v).nprobes(args.m).limit(K).to_list()  # must return list/iterable of IDs
		q1 = time.perf_counter()
		query_latencies.append(q1 - q0)
		labels = [x["id"] for x in labels]
		if len(labels) < K:
			labels = list(labels) + [-1] * (K - len(labels))
		preds[i, :] = np.asarray(labels[:K], dtype=np.int64)
		print(i, end='\r')
	t2 = time.perf_counter()

	search_wall = t2 - t1
	query_latencies = np.array(query_latencies)
	avg_latency = np.mean(query_latencies)
	p50_latency = np.percentile(query_latencies, 50)
	p95_latency = np.percentile(query_latencies, 95)
	p99_latency = np.percentile(query_latencies, 99)
	total_q_time = np.sum(query_latencies)
	qps = nq_total / total_q_time if total_q_time > 0 else float("inf")

	recalls = compute_recalls(preds, GT[idxs], K)

	results = {
		"vectors": N,
		"dim": dim,
		"queries": int(nq_total),
		"params": {
			"M": args.m, 
			"ef_construction": args.efc, 
			"ef_search": args.efs, 
			"seed": args.seed
		},
		"build_latency": build_latency,
		"search_avg_latency": avg_latency,
		"search_p50_latency": p50_latency,
		"search_p95_latency": p95_latency,
		"search_p99_latency": p99_latency,
		"qps": qps,
		"search_wall_time": search_wall,
	}
	results.update(recalls)

	return results


def main():
	process = psutil.Process(os.getpid())

	p = argparse.ArgumentParser(description="Benchmark vector databases on ANN datasets")
	p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
	p.add_argument("--dataset", type=str, default="sift-128", 
				   choices=list(DATASETS.keys()),
				   help="Dataset to use for benchmarking")
	p.add_argument("--engine", type=str, default="brinicle")
	p.add_argument("--m", type=int, default=16)
	p.add_argument("--efc", type=int, default=200, help="ef_construction")
	p.add_argument("--efs", type=int, default=64, help="ef_search")
	p.add_argument("--max-queries", type=int, default=10000)
	p.add_argument("--sample", action="store_true", help="randomly sample queries instead of first N")
	p.add_argument("--seed", type=int, default=123)
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
	print("[engine] ", args.engine)

	if args.engine == "hnswlib":
		build_latency, hnsw_index = run_benchmark_hnswlib_build(X, args)
	elif args.engine == "brinicle":
		build_latency = run_benchmark_brinicle_build(X, args)
	elif args.engine == "faiss":
		faiss_index, build_latency = run_benchmark_faiss_build(X, args)
	elif args.engine == "lancedb":
		build_latency = run_benchmark_lancedb_build(X, args)
	else:
		raise Exception("engine name does not exist")
	N, dim = X.shape[0], X.shape[1]
	del X

	batch_results = None
	try_size = 10
	for _ in range(try_size):
		if args.engine == "hnswlib":
			results = run_benchmark_hnswlib_search(hnsw_index, Q, GT, args, build_latency, N, dim)
		elif args.engine == "brinicle":
			results = run_benchmark_brinicle_search(Q, GT, args, build_latency, N, dim)
		elif args.engine == "faiss":
			results = run_benchmark_faiss_search(faiss_index, Q, GT, args, build_latency, N, dim)
		elif args.engine == "lancedb":
			results = run_benchmark_lancedb_search(Q, GT, args, build_latency, N, dim)
		else:
			raise Exception("engine name does not exist")

		if batch_results:
			batch_results["search_avg_latency"] += results["search_avg_latency"]
			batch_results["qps"] += results["qps"]
			batch_results["search_wall_time"] += results["search_wall_time"]
		else:
			batch_results = results

	batch_results["search_avg_latency"] /= try_size
	batch_results["qps"] /= try_size
	batch_results["search_wall_time"] /= try_size

	print(json.dumps(batch_results, indent=2))



if __name__ == "__main__":
	main()
