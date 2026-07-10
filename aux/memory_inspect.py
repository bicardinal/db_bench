import pathlib
import subprocess
import threading
import time
from dataclasses import dataclass


def _docker_inspect_pid(container_name: str) -> int:
	out = subprocess.check_output(
		[
			"docker",
			"container",
			"inspect",
			"-f",
			"{{.State.Pid}}",
			container_name,
		],
		text=True,
		stderr=subprocess.STDOUT,
	).strip()

	pid = int(out)

	if pid <= 0:
		raise RuntimeError(
			f"Container {container_name!r} is not running or has no host PID. "
			f"docker inspect returned pid={pid}."
		)

	return pid


def _cgroup_v2_dir_for_pid(pid: int) -> pathlib.Path:
	with open(f"/proc/{pid}/cgroup", "r") as f:
		for line in f:
			parts = line.strip().split(":")
			if len(parts) == 3 and parts[0] == "0" and parts[1] == "":
				rel = parts[2]
				return pathlib.Path("/sys/fs/cgroup") / rel.lstrip("/")

	raise RuntimeError(f"Could not find cgroup v2 path for pid={pid}")


def _read_int(path: pathlib.Path) -> int:
	return int(path.read_text().strip())


def _read_memory_stat(path: pathlib.Path) -> dict[str, int]:
	out = {}
	for line in path.read_text().splitlines():
		key, value = line.split()
		out[key] = int(value)
	return out


@dataclass
class MemorySample:
	raw_bytes: int = 0
	working_set_bytes: int = 0
	anon_bytes: int = 0
	file_bytes: int = 0
	active_file_bytes: int = 0
	inactive_file_bytes: int = 0
	kernel_bytes: int = 0
	slab_bytes: int = 0


class CgroupMemoryMonitor:

	def __init__(self, container_name: str, interval_s: float = 0.05):
		self.container_name = container_name
		self.interval_s = interval_s

		self._stop = threading.Event()
		self.thread = None
		self.cgdir: pathlib.Path | None = None

		self.last = MemorySample()
		self.peak = MemorySample()

		self.last_bytes = 0
		self.peak_bytes = 0

	def start(self):
		self._stop.clear()

		self.last = MemorySample()
		self.peak = MemorySample()
		self.last_bytes = 0
		self.peak_bytes = 0

		pid = _docker_inspect_pid(self.container_name)
		self.cgdir = _cgroup_v2_dir_for_pid(pid)

		mem_current = self.cgdir / "memory.current"
		mem_stat = self.cgdir / "memory.stat"

		if not mem_current.exists():
			raise RuntimeError(
				f"{mem_current} not found. Are you sure cgroup v2 is enabled?"
			)

		if not mem_stat.exists():
			raise RuntimeError(f"{mem_stat} not found.")

		def sample_once() -> MemorySample:
			raw = _read_int(mem_current)
			stat = _read_memory_stat(mem_stat)

			inactive_file = stat.get("inactive_file", 0)
			working_set = max(raw - inactive_file, 0)

			return MemorySample(
				raw_bytes=raw,
				working_set_bytes=working_set,
				anon_bytes=stat.get("anon", 0),
				file_bytes=stat.get("file", 0),
				active_file_bytes=stat.get("active_file", 0),
				inactive_file_bytes=inactive_file,
				kernel_bytes=stat.get("kernel", 0),
				slab_bytes=stat.get("slab", 0),
			)

		def update_peak(sample: MemorySample):
			self.last = sample

			self.last_bytes = sample.raw_bytes
			self.peak_bytes = max(self.peak_bytes, sample.raw_bytes)

			self.peak.raw_bytes = max(self.peak.raw_bytes, sample.raw_bytes)
			self.peak.working_set_bytes = max(
				self.peak.working_set_bytes,
				sample.working_set_bytes,
			)
			self.peak.anon_bytes = max(self.peak.anon_bytes, sample.anon_bytes)
			self.peak.file_bytes = max(self.peak.file_bytes, sample.file_bytes)
			self.peak.active_file_bytes = max(
				self.peak.active_file_bytes,
				sample.active_file_bytes,
			)
			self.peak.inactive_file_bytes = max(
				self.peak.inactive_file_bytes,
				sample.inactive_file_bytes,
			)
			self.peak.kernel_bytes = max(self.peak.kernel_bytes, sample.kernel_bytes)
			self.peak.slab_bytes = max(self.peak.slab_bytes, sample.slab_bytes)

		def run():
			while not self._stop.is_set():
				try:
					update_peak(sample_once())
				except FileNotFoundError:
					break
				time.sleep(self.interval_s)

		update_peak(sample_once())

		self.thread = threading.Thread(target=run, daemon=True)
		self.thread.start()
		return self

	def stop(self):
		self._stop.set()
		if self.thread:
			self.thread.join(timeout=2.0)

	@staticmethod
	def _mb(value: int) -> float:
		return value / (1024 * 1024)

	def last_report_mb(self) -> dict[str, float]:
		return {
			"raw_mb": self._mb(self.last.raw_bytes),
			"working_set_mb": self._mb(self.last.working_set_bytes),
			"anon_mb": self._mb(self.last.anon_bytes),
			"file_mb": self._mb(self.last.file_bytes),
			"active_file_mb": self._mb(self.last.active_file_bytes),
			"inactive_file_mb": self._mb(self.last.inactive_file_bytes),
			"kernel_mb": self._mb(self.last.kernel_bytes),
			"slab_mb": self._mb(self.last.slab_bytes),
		}

	def peak_report_mb(self) -> dict[str, float]:
		return {
			"raw_peak_mb": self._mb(self.peak.raw_bytes),
			"working_set_peak_mb": self._mb(self.peak.working_set_bytes),
			"anon_peak_mb": self._mb(self.peak.anon_bytes),
			"file_peak_mb": self._mb(self.peak.file_bytes),
			"active_file_peak_mb": self._mb(self.peak.active_file_bytes),
			"inactive_file_peak_mb": self._mb(self.peak.inactive_file_bytes),
			"kernel_peak_mb": self._mb(self.peak.kernel_bytes),
			"slab_peak_mb": self._mb(self.peak.slab_bytes),
		}

	def read_memory_stat(self) -> dict[str, int]:
		if self.cgdir is None:
			raise RuntimeError("Monitor has not been started yet.")

		return _read_memory_stat(self.cgdir / "memory.stat")