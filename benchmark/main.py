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

from aux.brinicle_client import VectorEngineClient

import lancedb

import pymilvus

import chromadb

import weaviate
import weaviate.classes.config as wvc
from weaviate.util import generate_uuid5

from qdrant_client.models import Distance, VectorParams, HnswConfigDiff, SearchParams
from qdrant_client import QdrantClient


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
	"""
	$ git clone https://github.com/bicardinal/brinicle.git
	$ cd brinicle
	$ bash build.sh
	$ make docker-build
	$ make docker-run
	"""
	args.efs = max(K, args.efs)
	print(f"[build] Creating Brinicle(M={args.m}, efc={args.efc}, efs={args.efs}, k_base={args.k_base})")
	idx_name = "brinicle_bench"
	batch_size = 8192

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
			ids=ids[b:b+batch_size],
			vectors=X[b:b+batch_size],
		)
		print("batch idx:", b, end='\r')
	client.finalize(idx_name)
	build_latency = time.perf_counter() - t0
	print(f"[build] Done in {build_latency:.3f}s.")
	return build_latency


def run_benchmark_brinicle_search(Q, GT, args, build_latency, N, dim):
	print(f"[search] K={K}")
	idx_name = "brinicle_bench"
	client = VectorEngineClient()
	client.load_index(idx_name)
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
	labels = client.search(idx_name, queries[0], k=K, efs=args.efs)
	print(f"[search] Running {nq_total} queries @ top-{K}...")
	t1 = time.perf_counter()
	total_q_time = 0.0
	for i, v in enumerate(queries, start=0):
		q0 = time.perf_counter()
		labels = client.search(idx_name, v, k=K, efs=args.efs)  # must return list/iterable of IDs
		q1 = time.perf_counter()
		total_q_time += (q1 - q0)
		labels = [int(x) for x in labels] # back to int so that we can calculate recall
		if len(labels) < K:
			print("returned results is less than K")
			labels = list(labels) + [-1] * (K - len(labels))
		preds[i, :] = np.asarray(labels[:K], dtype=np.int64)
		print(i, end='\r')
	t2 = time.perf_counter()

	search_wall = t2 - t1  # end-to-end search window
	avg_latency = total_q_time / nq_total
	qps = nq_total / total_q_time if total_q_time > 0 else float("inf")


	recalls = compute_recalls(preds, GT[idxs], K)

	results = {
		"vectors": N,
		"dim": dim,
		"queries": int(nq_total),
		"params": {"M": args.m, "ef_construction": args.efc, "ef_search": args.efs, "seed": args.seed, "k_base": args.k_base},
		"build_latency": build_latency,
		"search_avg_latency": avg_latency,            # seconds per query
		"qps": qps,                                   # queries per second (measured on per-call time)
		"search_wall_time": search_wall,              # total wall time (s) around search loop
	}
	results.update(recalls)

	return results


def run_benchmark_qdrant_build(X, args):
	"""
	$ docker run -p 6333:6333 qdrant/qdrant
	"""
	args.efs = max(K, args.efs)
	print(f"[build] Creating QdrantANN(M={args.m}, efc={args.efc}, efs={args.efs}, k_base={args.k_base})")
	client = QdrantClient(host="0.0.0.0", port="6333", grpc_port="6334", prefer_grpc=False)
	if client.collection_exists(collection_name="bench"):
		client.delete_collection(collection_name="bench")

	client.create_collection(
		collection_name="bench",
		vectors_config=VectorParams(size=X.shape[1], distance=Distance.EUCLID),
		hnsw_config=HnswConfigDiff(m=args.m, ef_construct=args.efc),
	)
	t0 = time.perf_counter()
	client.upload_collection(
		collection_name="bench",
		vectors=X,
		ids=list(range(X.shape[0])),
		parallel=4,
		batch_size=8196,
		wait=True,
	)
	build_latency = time.perf_counter() - t0
	print(f"[build] Done in {build_latency:.3f}s.")
	return build_latency


def run_benchmark_qdrant_search(Q, GT, args, build_latency, N, dim):
	print(f"[search] K={K}")
	client = QdrantClient(host="0.0.0.0", port="6333", grpc_port="6334", prefer_grpc=False)
	nq_total = min(args.max_queries, Q.shape[0])
	rng = np.random.default_rng(args.seed)
	idxs = np.arange(Q.shape[0])
	if args.sample:
		idxs = rng.choice(idxs, size=nq_total, replace=False)
	else:
		idxs = idxs[:nq_total]
	queries = Q[idxs].tolist()

	preds = np.empty((nq_total, K), dtype=np.int64)
	client.query_points(
		collection_name="bench",
		query=queries[0],
		search_params=SearchParams(hnsw_ef=args.efs, exact=False),
		limit=K,
	)
	print(f"[search] Running {nq_total} queries @ top-{K}...")

	t1 = time.perf_counter()
	total_q_time = 0.0
	for i, v in enumerate(queries, start=0):
		q0 = time.perf_counter()
		labels = client.query_points(
			collection_name="bench",
			query=v,
			search_params=SearchParams(hnsw_ef=args.efs, exact=False),
			limit=K,
		).points

		q1 = time.perf_counter()
		labels = np.asarray(
			[int(p.id) for p in labels],
			dtype=np.int64
		)
		total_q_time += (q1 - q0)
		if len(labels) < K:
			print("returned results is less than K")
			labels = list(labels) + [-1] * (K - len(labels))
		preds[i, :] = np.asarray(labels[:K], dtype=np.int64)
		print(i, end='\r')
	t2 = time.perf_counter()

	search_wall = t2 - t1  # end-to-end search window
	avg_latency = total_q_time / nq_total
	qps = nq_total / total_q_time if total_q_time > 0 else float("inf")


	recalls = compute_recalls(preds, GT[idxs], K)

	results = {
		"vectors": N,
		"dim": dim,
		"queries": int(nq_total),
		"params": {"M": args.m, "ef_construction": args.efc, "ef_search": args.efs, "seed": args.seed, "k_base": args.k_base},
		"build_latency": build_latency,
		"search_avg_latency": avg_latency,            # seconds per query
		"qps": qps,                                   # queries per second (measured on per-call time)
		"search_wall_time": search_wall,              # total wall time (s) around search loop
	}
	results.update(recalls)

	return results


def run_benchmark_chroma_build(X, args):
	"""
	$ docker run -v ./chroma-data:/data -p 8000:8000 chromadb/chroma
	"""
	args.efs = max(K, args.efs)
	print(f"[build] Creating ChromaDB(M={args.m}, efc={args.efc}, efs={args.efs}, k_base={args.k_base})")
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
	t0 = time.perf_counter()
	ids = [str(i) for i in range(X.shape[0])]
	embeddings = X.tolist()

	batch_size = 4096
	for i in range(0, len(ids), batch_size):
		batch_ids = ids[i:i + batch_size]
		batch_embeddings = embeddings[i:i + batch_size]
		collection.add(
			ids=batch_ids,
			embeddings=batch_embeddings
		)
	
	build_latency = time.perf_counter() - t0
	print(f"[build] Done in {build_latency:.3f}s.")
	return build_latency

def run_benchmark_chroma_search(Q, GT, args, build_latency, N, dim):
	print(f"[search] K={K}")
	client = chromadb.HttpClient(host="localhost", port=8000)
	collection = client.get_collection("bench")
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
	collection.query(
		query_embeddings=[queries[0]],
		n_results=K
	)
	print(f"[search] Running {nq_total} queries @ top-{K}...")
	
	t1 = time.perf_counter()
	total_q_time = 0.0
	for i, v in enumerate(queries, start=0):
		q0 = time.perf_counter()
		results = collection.query(
			query_embeddings=[v],
			n_results=K
		)
		q1 = time.perf_counter()

		# extract IDs from results
		labels = np.asarray(
			[int(id_str) for id_str in results['ids'][0]],
			dtype=np.int64
		)
		
		total_q_time += (q1 - q0)
		
		if len(labels) < K:
			print("returned results is less than K")
			labels = list(labels) + [-1] * (K - len(labels))
		
		preds[i, :] = np.asarray(labels[:K], dtype=np.int64)
		print(i, end='\r')
	
	t2 = time.perf_counter()
	
	search_wall = t2 - t1
	avg_latency = total_q_time / nq_total
	qps = nq_total / total_q_time if total_q_time > 0 else float("inf")
	
	recalls = compute_recalls(preds, GT[idxs], K)
	
	results = {
		"vectors": N,
		"dim": dim,
		"queries": int(nq_total),
		"params": {"M": args.m, "ef_construction": args.efc, "ef_search": args.efs, "seed": args.seed, "k_base": args.k_base},
		"build_latency": build_latency,
		"search_avg_latency": avg_latency,            # seconds per query
		"qps": qps,                                   # queries per second (measured on per-call time)
		"search_wall_time": search_wall,              # total wall time (s) around search loop
	}
	results.update(recalls)
	return results


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
		return build_latency
	finally:
		client.close()

def run_benchmark_weaviate_search(Q, GT, args, build_latency, N, dim):
	client = weaviate.connect_to_local()
	try:
		collection = client.collections.get("Bench")
		nq_total = min(args.max_queries, Q.shape[0])
		rng = np.random.default_rng(args.seed)
		idxs = np.arange(Q.shape[0])
		if args.sample:
			idxs = rng.choice(idxs, size=nq_total, replace=False)
		else:
			idxs = idxs[:nq_total]

		queries = Q[idxs].tolist()
		preds = np.empty((nq_total, K), dtype=np.int64)

		collection.query.near_vector(
			near_vector=queries[0],
			limit=K
		)

		print(f"[search] Running {nq_total} queries @ top-{K}...")

		t1 = time.perf_counter()
		total_q_time = 0.0

		for i, v in enumerate(queries):
			q0 = time.perf_counter()
			response = collection.query.near_vector(
				near_vector=v,
				limit=K,
				return_properties=["original_id"]
			)
			q1 = time.perf_counter()

			# Extract IDs from Weaviate objects
			labels = [obj.properties["original_id"] for obj in response.objects]

			total_q_time += (q1 - q0)
			if len(labels) < K:
				labels = list(labels) + [-1] * (K - len(labels))

			preds[i, :] = np.asarray(labels[:K], dtype=np.int64)
			print(f"{i}", end='\r')

		t2 = time.perf_counter()

		search_wall = t2 - t1
		avg_latency = total_q_time / nq_total
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
			"qps": qps,
			"search_wall_time": search_wall,
		}
		results.update(recalls)
		return results
	finally:
		client.close()


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
		# token=TOKEN
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
	return build_latency


def run_benchmark_milvus_search(Q, GT, args, build_latency, N, dim):
	pymilvus.connections.connect("default", host="localhost", port="19530")
	collection = pymilvus.Collection("bench_milvus")
	collection.load()
	nq_total = min(args.max_queries, Q.shape[0])
	rng = np.random.default_rng(args.seed)
	idxs = np.arange(Q.shape[0])
	if args.sample:
		idxs = rng.choice(idxs, size=nq_total, replace=False)
	else:
		idxs = idxs[:nq_total]

	queries = Q[idxs].tolist()
	preds = np.empty((nq_total, K), dtype=np.int64)
	
	search_params = {"metric_type": "L2", "params": {"ef": args.efs}}

	collection.search(
		data=[queries[0]], 
		anns_field="embedding", 
		param=search_params, 
		limit=K
	)

	print(f"[search] Running {nq_total} queries @ top-{K}...")
	t1 = time.perf_counter()
	total_q_time = 0.0
	
	for i, v in enumerate(queries):
		q0 = time.perf_counter()
		res = collection.search(
			data=[v], 
			anns_field="embedding", 
			param=search_params, 
			limit=K
		)
		q1 = time.perf_counter()
		
		total_q_time += (q1 - q0)
		
		labels = [hit.id for hit in res[0]]
		
		if len(labels) < K:
			labels = labels + [-1] * (K - len(labels))
		
		preds[i, :] = np.asarray(labels[:K], dtype=np.int64)
		print(f"Query {i+1}/{nq_total}", end='\r')

	t2 = time.perf_counter()

	search_wall = t2 - t1
	avg_latency = total_q_time / nq_total
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
		"qps": qps,
		"search_wall_time": search_wall,
	}
	results.update(recalls)
	collection.release()
	return results


def run_benchmark_lancedb_build(X, args):
	args.efs = max(K, args.efs)
	print(f"[build] Creating lance index(dim=128, M={args.m}, seed={args.seed}, efc={args.efc}, efs={args.efs})")
	db = lancedb.connect("./benchmark/lancedb")
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
	db = lancedb.connect("./benchmark/lancedb")
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

	table.search([queries[0], queries[1]]).limit(K).to_list()
	print(f"[search] Running {nq_total} queries @ top-{K}...")
	t1 = time.perf_counter()
	total_q_time = 0.0
	for i, v in enumerate(queries, start=0):
		q0 = time.perf_counter()
		labels = table.search(v).nprobes(args.m).limit(K).to_list()  # must return list/iterable of IDs
		q1 = time.perf_counter()
		total_q_time += (q1 - q0)
		labels = [x["id"] for x in labels]
		if len(labels) < K:
			labels = list(labels) + [-1] * (K - len(labels))
		preds[i, :] = np.asarray(labels[:K], dtype=np.int64)
		print(i, end='\r')
	t2 = time.perf_counter()

	search_wall = t2 - t1  # end-to-end search window
	avg_latency = total_q_time / nq_total
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
	p.add_argument("--db", type=str, default="brinicle")
	p.add_argument("--m", type=int, default=16)
	p.add_argument("--efc", type=int, default=200, help="ef_construction")
	p.add_argument("--efs", type=int, default=64, help="ef_search")
	p.add_argument("--k-base", type=int, default=-1, help="k base")
	p.add_argument("--max-queries", type=int, default=10000)
	p.add_argument("--sample", action="store_true", help="randomly sample queries instead of first N")
	p.add_argument("--insert-test", action="store_true", help="test insertion effect")
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
	print("[db] ", args.db)
	direct_test = not args.insert_test
	if args.db == "qdrant":
		build_latency = run_benchmark_qdrant_build(X, args)
	elif args.db == "brinicle":
		build_latency = run_benchmark_brinicle_build(X, args)
	elif args.db == "chroma":
		build_latency = run_benchmark_chroma_build(X, args)
	elif args.db == "weaviate":
		build_latency = run_benchmark_weaviate_build(X, args)
	elif args.db == "milvus":
		build_latency = run_benchmark_milvus_build(X, args)
	elif args.db == "lancedb":
		build_latency = run_benchmark_lancedb_build(X, args)
	else:
		raise Exception("db name does not exist")
	N, dim = X.shape[0], X.shape[1]
	del X

	batch_results = None
	try_size = 10
	for _ in range(try_size):
		if args.db == "qdrant":
			results = run_benchmark_qdrant_search(Q, GT, args, build_latency, N, dim)
		elif args.db == "brinicle":
			results = run_benchmark_brinicle_search(Q, GT, args, build_latency, N, dim)
		elif args.db == "chroma":
			results = run_benchmark_chroma_search(Q, GT, args, build_latency, N, dim)
		elif args.db == "weaviate":
			results = run_benchmark_weaviate_search(Q, GT, args, build_latency, N, dim)
		elif args.db == "milvus":
			results = run_benchmark_milvus_search(Q, GT, args, build_latency, N, dim)
		elif args.db == "lancedb":
			results = run_benchmark_lancedb_search(Q, GT, args, build_latency, N, dim)
		else:
			raise Exception("db name does not exist")

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
