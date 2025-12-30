import pathlib
import subprocess
import threading
import time

def _docker_inspect_pid(container_name: str) -> int:
	out = subprocess.check_output(
		["docker", "inspect", "-f", "{{.State.Pid}}", container_name],
		text=True
	).strip()
	return int(out)

def _cgroup_v2_dir_for_pid(pid: int) -> pathlib.Path:
	with open(f"/proc/{pid}/cgroup", "r") as f:
		for line in f:
			parts = line.strip().split(":")
			if len(parts) == 3 and parts[0] == "0" and parts[1] == "":
				rel = parts[2]
				return pathlib.Path("/sys/fs/cgroup") / rel.lstrip("/")
	raise RuntimeError(f"Could not find cgroup v2 path for pid={pid}")

def _read_int(p: pathlib.Path) -> int:
	return int(p.read_text().strip())

class CgroupMemoryMonitor:
	"""
		samples memory.current to compute a peak for *this phase*.
	"""
	def __init__(self, container_name: str, interval_s: float = 0.05):
		self.container_name = container_name
		self.interval_s = interval_s
		self._stop = threading.Event()
		self.thread = None
		self.peak_bytes = 0
		self.last_bytes = 0
		self.cgdir = None

	def start(self):
		pid = _docker_inspect_pid(self.container_name)
		self.cgdir = _cgroup_v2_dir_for_pid(pid)
		mem_current = self.cgdir / "memory.current"
		if not mem_current.exists():
			raise RuntimeError(f"{mem_current} not found (are you sure cgroup v2 is enabled?)")

		def run():
			while not self._stop.is_set():
				try:
					cur = _read_int(mem_current)
					self.last_bytes = cur
					if cur > self.peak_bytes:
						self.peak_bytes = cur
				except FileNotFoundError:
					# container exited
					break
				time.sleep(self.interval_s)

		self.thread = threading.Thread(target=run, daemon=True)
		self.thread.start()
		return self

	def stop(self):
		self._stop.set()
		if self.thread:
			self.thread.join(timeout=2.0)

	def read_memory_stat(self) -> dict:
		stat_path = self.cgdir / "memory.stat"
		out = {}
		for line in stat_path.read_text().splitlines():
			k, v = line.split()
			out[k] = int(v)
		return out
