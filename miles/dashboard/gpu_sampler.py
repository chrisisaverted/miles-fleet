"""Per-node GPU utilization sampler feeding the dashboard timeline.

One instance runs per GPU node (the collector spawns it as a Ray actor with
``NodeAffinitySchedulingStrategy``; the class itself is plain Python and
unit-testable). A daemon thread samples every SMI-visible device on the node at
``interval`` seconds — physical device order, independent of CUDA/HIP process
visibility variables — buffers locally, and hands batches to the injected
``push(node, batch)`` callable (the collector wraps its own Ray handle) every
``FLUSH_INTERVAL_SECONDS``, so there is roughly one RPC per node per flush
rather than one per sample.

Degradation: NVML is preferred and AMD SMI is the fallback; neither being
usable (no pynvml/amdsmi, driver mismatch) disables the sampler with a single
warning — the timeline just lacks the util band. A device that fails mid-run
(e.g. during a GPU reset) is skipped for that tick with rate-limited warnings;
the other devices keep reporting.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import ClassVar, Protocol

from miles.dashboard.logging_utils import RateLimitedWarner
from miles.dashboard.store import GpuProcessSample, GpuSample

logger = logging.getLogger(__name__)


class _GpuProvider(Protocol):
    """Vendor SMI adapter. Returns values in the dashboard's units (util %,
    VRAM MiB, power W) so the sampler stays vendor-agnostic."""

    name: str

    def initialize(self) -> tuple[list, list[str]]: ...

    def read_device(self, handle) -> tuple[int, int, int]: ...

    def read_processes(self, handle) -> list[tuple[int, str, int]]: ...


def _text(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


class _NvmlProvider:
    name = "NVML"

    def __init__(self, api):
        self._api = api

    def initialize(self) -> tuple[list, list[str]]:
        """Returns (handles, uuids); raises when NVML is unusable so
        auto-detection can fall through to the next backend."""
        self._api.nvmlInit()
        count = self._api.nvmlDeviceGetCount()
        if count == 0:
            raise RuntimeError("no NVML devices")
        handles = [self._api.nvmlDeviceGetHandleByIndex(i) for i in range(count)]
        return handles, [_text(self._api.nvmlDeviceGetUUID(handle)) for handle in handles]

    def read_device(self, handle) -> tuple[int, int, int]:
        util = int(self._api.nvmlDeviceGetUtilizationRates(handle).gpu)
        mem_mb = int(self._api.nvmlDeviceGetMemoryInfo(handle).used) >> 20
        power_w = int(self._api.nvmlDeviceGetPowerUsage(handle)) // 1000
        return util, mem_mb, power_w

    def read_processes(self, handle) -> list[tuple[int, str, int]]:
        processes = []
        for proc in self._api.nvmlDeviceGetComputeRunningProcesses(handle):
            pid = int(proc.pid)
            processes.append((pid, self._process_name(pid), int(proc.usedGpuMemory or 0) >> 20))
        return processes

    def _process_name(self, pid: int) -> str:
        try:
            return _text(self._api.nvmlSystemGetProcessName(pid))
        except Exception:
            return f"pid {pid}"  # process exited between enumeration and lookup, or name unavailable


class _AmdSmiProvider:
    name = "AMD SMI"

    def __init__(self, api):
        self._api = api

    def initialize(self) -> tuple[list, list[str]]:
        self._api.amdsmi_init()
        handles = self._api.amdsmi_get_processor_handles()
        if not handles:
            raise RuntimeError("no AMD SMI devices")
        # Keep the SMI slot as the dashboard lane id. Ray numbers the GPU
        # resources visible to its node in the same sequential space; do not
        # re-apply HIP/CUDA visibility variables or re-index by KFD id. Note:
        # partitioned MI300+ devices share one UUID per physical GPU, so the
        # uuids are not necessarily unique per lane.
        return list(handles), [_text(self._api.amdsmi_get_gpu_device_uuid(handle)) for handle in handles]

    def read_device(self, handle) -> tuple[int, int, int]:
        util = int(self._api.amdsmi_get_gpu_activity(handle)["gfx_activity"])
        # Unlike NVML, AMD SMI reports VRAM usage in MiB already.
        mem_mb = int(self._api.amdsmi_get_gpu_vram_usage(handle)["vram_used"])
        try:
            power_w = _amd_socket_power(self._api.amdsmi_get_power_info(handle))
        except Exception:
            # Power sensors can be unexposed (VM guests, some partition modes)
            # while activity/VRAM read fine; report 0 W rather than losing the sample.
            power_w = 0
        return util, mem_mb, power_w

    def read_processes(self, handle) -> list[tuple[int, str, int]]:
        processes = []
        for proc in self._api.amdsmi_get_gpu_process_list(handle):
            mem_mb = int((proc.get("memory_usage") or {}).get("vram_mem") or 0) >> 20
            if mem_mb == 0:
                continue  # KFD lists bystander pids holding no VRAM; NVML reports only compute processes
            pid = int(proc["pid"])
            name = proc.get("name")
            processes.append((pid, str(name) if name and name != "N/A" else f"pid {pid}", mem_mb))
        return processes


def _amd_socket_power(power: dict) -> int:
    # socket_power selects current power on MI300+ and average power on older
    # cards; older wrappers lack the field entirely. Unavailable sensors show
    # up as the string "N/A" (ROCm >= 6.1) or a raw integer sentinel (uint16
    # 0xFFFF on ROCm 6.0, UINT32_MAX from newer C structs).
    for field in ("socket_power", "current_socket_power", "average_socket_power"):
        value = power.get(field)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value < 0xFFFF:
            return int(value)
    raise ValueError(f"AMD SMI socket power unavailable: {power!r}")


def _import_nvml():
    import pynvml

    return pynvml


def _import_amdsmi():
    import amdsmi

    return amdsmi


class GpuSampler:
    FLUSH_INTERVAL_SECONDS: ClassVar[float] = 5.0
    # per-process memory breakdown is a coarser, heavier SMI call (enumerates
    # every process) than the plain util/mem read, so it samples on its own,
    # slower cadence rather than every `interval` tick
    PROCESS_SAMPLE_INTERVAL_SECONDS: ClassVar[float] = 5.0

    def __init__(
        self,
        push: Callable[[str, list[GpuSample]], None],
        *,
        node: str,
        interval: float = 1.0,
        nvml=None,
        push_processes: Callable[[str, list[GpuProcessSample]], None] | None = None,
        amdsmi=None,
    ):
        assert interval > 0, f"{interval=}"
        assert nvml is None or amdsmi is None, "inject only one GPU telemetry backend"
        self._push = push
        self._push_processes = push_processes
        self.node = node
        self.interval = interval
        self._provider: _GpuProvider | None = None
        self._handles: list = []
        self._uuids: list[str] = []
        self._buffer: list[GpuSample] = []
        self._process_buffer: list[GpuProcessSample] = []
        self._buffer_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._warner = RateLimitedWarner(logger)
        self.available = self._init_provider(nvml=nvml, amdsmi=amdsmi)

    def _init_provider(self, *, nvml, amdsmi) -> bool:
        if nvml is not None:
            candidates = [(_NvmlProvider, lambda: nvml)]
        elif amdsmi is not None:
            candidates = [(_AmdSmiProvider, lambda: amdsmi)]
        else:
            candidates = [(_NvmlProvider, _import_nvml), (_AmdSmiProvider, _import_amdsmi)]

        failures = []
        for cls, import_api in candidates:
            try:
                provider = cls(import_api())
                self._handles, self._uuids = provider.initialize()
            except Exception as error:
                failures.append((cls.name, str(error)))
                logger.debug("%s unavailable on %s", cls.name, self.node, exc_info=True)
                continue
            self._provider = provider
            return True

        if len(failures) == 1:
            name, error = failures[0]
            logger.warning("%s unavailable on %s (%s); GPU utilization will not be collected", name, self.node, error)
        else:
            detail = "; ".join(f"{name}: {error}" for name, error in failures)
            logger.warning(
                "GPU telemetry unavailable on %s (%s); GPU utilization will not be collected", self.node, detail
            )
        return False

    # ------------------------------ lifecycle -------------------------------

    def gpu_uuids(self) -> list[str]:
        return list(self._uuids)

    def start(self) -> bool:
        """Begin sampling; returns False (and stays inert) when no SMI backend is usable."""
        if not self.available:
            return False
        assert self._thread is None, "sampler already started"
        self._thread = threading.Thread(target=self._run, name="dashboard-gpu-sampler", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval + self.FLUSH_INTERVAL_SECONDS)
        self.flush()

    def _run(self) -> None:
        next_flush = time.monotonic() + self.FLUSH_INTERVAL_SECONDS
        next_process_sample = time.monotonic() + self.PROCESS_SAMPLE_INTERVAL_SECONDS
        while not self._stop_event.is_set():
            self.sample_once(time.time())
            if self._push_processes is not None and time.monotonic() >= next_process_sample:
                self.sample_processes_once(time.time())
                next_process_sample = time.monotonic() + self.PROCESS_SAMPLE_INTERVAL_SECONDS
            if time.monotonic() >= next_flush:
                self.flush()
                next_flush = time.monotonic() + self.FLUSH_INTERVAL_SECONDS
            self._stop_event.wait(self.interval)

    # -------------------------------- sampling ------------------------------

    def sample_once(self, ts: float) -> int:
        """Sample every device once into the buffer. Returns the sample count."""
        if not self.available:
            return 0
        count = 0
        for gpu, handle in enumerate(self._handles):
            try:
                util, mem_mb, power_w = self._provider.read_device(handle)
            except Exception:
                self._warner.warn(
                    f"{self._provider.name} read failed for gpu {gpu} on {self.node}; skipping this tick"
                )
                continue
            with self._buffer_lock:
                self._buffer.append(
                    GpuSample(ts=ts, node=self.node, gpu=gpu, util=util, mem_mb=mem_mb, power_w=power_w)
                )
            count += 1
        return count

    def sample_processes_once(self, ts: float) -> int:
        """Per-process VRAM breakdown once per GPU: who is actually holding
        the memory, not just the per-GPU aggregate ``sample_once`` reports."""
        if not self.available:
            return 0
        count = 0
        for gpu, handle in enumerate(self._handles):
            try:
                processes = self._provider.read_processes(handle)
            except Exception:
                self._warner.warn(
                    f"{self._provider.name} process query failed for gpu {gpu} on {self.node}; skipping this tick"
                )
                continue
            for pid, name, mem_mb in processes:
                with self._buffer_lock:
                    self._process_buffer.append(
                        GpuProcessSample(ts=ts, node=self.node, gpu=gpu, pid=pid, name=name, mem_mb=mem_mb)
                    )
                count += 1
        return count

    def flush(self) -> None:
        with self._buffer_lock:
            batch, self._buffer = self._buffer, []
            process_batch, self._process_buffer = self._process_buffer, []
        if batch:
            self._push(self.node, batch)
        if process_batch and self._push_processes is not None:
            self._push_processes(self.node, process_batch)
