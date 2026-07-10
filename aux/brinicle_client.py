import requests
import numpy as np
import orjson
from typing import List, Optional, Dict, Any, Tuple
import httpx

class VectorEngineClient:
	def __init__(self, base_url: str = "http://localhost:1984"):
		self.base_url = base_url.rstrip("/")
		self.session = requests.Session()

	def create_index(
		self,
		index_name: str,
		dim: int,
		delta_ratio: float = 0.10,
		ef_construction: int = 200,
		ef_search: int = 64,
		M: int = 16,
		build_n_threads: int = 2,
		seed: int = 0,
	) -> Dict[str, Any]:
		payload = {
			"index_name": index_name,
			"dim": dim,
			"delta_ratio": delta_ratio
		}
		payload["params"] = {
			"ef_construction": ef_construction,
			"ef_search": ef_search,
			"M": M,
			"rng_seed": seed,
			"build_n_threads": build_n_threads,
		}

		response = self.session.post(f"{self.base_url}/indexes", json=payload)
		response.raise_for_status()
		return response.json()

	def list_indexes(self) -> Dict[str, Any]:
		response = self.session.get(f"{self.base_url}/indexes")
		response.raise_for_status()
		return response.json()

	def delete_index(self, index_name: str, destroy: bool = False) -> Dict[str, Any]:
		response = self.session.delete(
			f"{self.base_url}/indexes/{index_name}",
			params={"destroy": destroy}
		)
		response.raise_for_status()
		return response.json()

	def load_index(self, index_name: str) -> Dict[str, Any]:
		response = self.session.post(
			f"{self.base_url}/indexes/load",
			json={"index_name": index_name}
		)
		response.raise_for_status()
		return response.json()

	def get_status(self, index_name: str) -> Dict[str, Any]:
		response = self.session.get(f"{self.base_url}/indexes/{index_name}/status")
		response.raise_for_status()
		return response.json()

	def init(self, index_name: str, mode: str) -> Dict[str, Any]:
		payload = {"index_name": index_name, "mode": mode}
		response = self.session.post(f"{self.base_url}/init", json=payload)
		response.raise_for_status()
		return response.json()

	def ingest(self, index_name: str, external_id: str, vector: np.ndarray) -> Dict[str, Any]:
		if isinstance(vector, np.ndarray):
			vector = vector.tolist()

		payload = orjson.dumps({
			"index_name": index_name,
			"external_id": external_id,
			"vector": vector
		})

		response = self.session.post(
			f"{self.base_url}/ingest",
			data=payload,
			headers={"Content-Type": "application/json"}
		)
		response.raise_for_status()
		return response.json()

	def ingest_batch_binary(
		self,
		index_name: str,
		ids: List[str],
		vectors: np.ndarray,
	) -> Dict[str, Any]:
		vectors = np.asarray(vectors, dtype="<f4", order="C")

		if vectors.ndim != 2:
			raise ValueError("vectors must be a 2D array")

		n, dim = vectors.shape

		if len(ids) != n:
			raise ValueError(
				f"Expected {n} ids for {n} vectors, got {len(ids)}"
			)

		dtype = np.dtype([
			("external_id", "S32"),
			("vector", "<f4", (dim,)),
		])

		records = np.empty(n, dtype=dtype)
		records["external_id"] = np.asarray(ids, dtype="S32")
		records["vector"] = vectors

		response = self.session.post(
			f"{self.base_url}/ingest/batch",
			params={"index_name": index_name},
			data=records.tobytes(),
			headers={"Content-Type": "application/octet-stream"},
		)
		response.raise_for_status()
		return orjson.loads(response.content)

	def finalize(
		self,
		index_name: str,
		build_params: Optional[Dict[str, Any]] = None,
		optimize: bool = False
	) -> Dict[str, Any]:
		payload = {
			"index_name": index_name,
			"optimize": optimize
		}
		if build_params:
			payload["build_params"] = build_params

		response = self.session.post(f"{self.base_url}/finalize", json=payload)
		response.raise_for_status()
		return response.json()

	def delete_items(
		self,
		index_name: str,
		external_ids: List[str],
		return_not_found: bool = False
	) -> Dict[str, Any]:
		payload = {
			"index_name": index_name,
			"external_ids": external_ids,
			"return_not_found": return_not_found
		}
		response = self.session.post(f"{self.base_url}/delete", json=payload)
		response.raise_for_status()
		return response.json()

	def rebuild(
		self,
		index_name: str,
		build_params: Optional[Dict[str, Any]] = None
	) -> Dict[str, Any]:
		payload = {"index_name": index_name}
		if build_params:
			payload["build_params"] = build_params

		response = self.session.post(f"{self.base_url}/rebuild", json=payload)
		response.raise_for_status()
		return response.json()

	def search(
		self,
		index_name: str,
		query: np.ndarray,
		k: int = 10,
		efs: int = 64
	) -> List[str]:
		r = self.session.post(
			f"{self.base_url}/search.bin",
			params={"index_name": index_name, "k": k, "efs": efs},
			data=query.tobytes(),
			headers={"Content-Type": "application/octet-stream"},
		)
		neighbors = r.json()
		return neighbors

	def search_batch(
		self,
		index_name: str,
		queries: np.ndarray,
		k: int = 10,
		efs: int = 64,
		n_jobs: int = 1,
	) -> List[List[str]]:

		if not isinstance(queries, np.ndarray):
			queries = np.asarray(queries, dtype=np.float32)
		if queries.dtype != np.float32:
			queries = queries.astype(np.float32)
		queries = np.ascontiguousarray(queries)
		r = self.session.post(
			f"{self.base_url}/search/batch.bin",
			params={"index_name": index_name, "k": k, "efs": efs, "n_jobs": n_jobs},
			data=queries.tobytes(),
			headers={"Content-Type": "application/octet-stream"},
		)
		r.raise_for_status()
		return r.json()

	def optimize(self, index_name: str) -> Dict[str, Any]:
		payload = orjson.dumps({"index_name": index_name})
		response = self.session.post(
			f"{self.base_url}/optimize",
			data=payload,
			headers={"Content-Type": "application/json"}
		)
		response.raise_for_status()
		return response.json()

	def close(self):
		self.session.close()

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		self.close()


if __name__ == "__main__":
	with VectorEngineClient() as client:
		client.create_index("test_index", dim=128)

		client.init("test_index", "build")

		vec = np.random.randn(128).astype(np.float32)
		client.ingest("test_index", "vec001", vec)

		vectors = [
			("vec002", np.random.randn(128).astype(np.float32)),
			("vec003", np.random.randn(128).astype(np.float32)),
		]
		client.ingest_batch_binary("test_index", vectors)

		client.finalize("test_index", optimize=True)

		query = np.random.randn(128).astype(np.float32)
		results = client.search("test_index", query, k=5)
		print(f"Search results: {results}")

		status = client.get_status("test_index")
		print(f"Status: {status}")
