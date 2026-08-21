"""Per-node GPU utilization sampler feeding the dashboard timeline.

One instance runs per GPU node (the collector spawns it as a Ray actor with
``NodeAffinitySchedulingStrategy``; the class itself is plain Python and
unit-testable). A daemon thread samples every SMI-visible device on the node at
``interval`` seconds — physical device order, independent of CUDA/HIP process
visibility variables — buffers locally, and hands batches to the injected
``push(node, batch)`` callable (the collector wraps its own Ray handle) every
``FLUSH_INTERVAL_SECONDS``, so there is roughly one RPC per node per flush.

NVML is preferred and AMD SMI is the fallback. Missing libraries or driver
mismatches disable the sampler with a single warning — the timeline just lacks
the util band. A device that fails mid-run (e.g. during a GPU reset) is skipped
for that tick with rate-limited warnings; the other devices keep reporting.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar, Protocol

from miles.dashboard.logging_utils import RateLimitedWarner
from miles.dashboard.store import GpuProcessSample, GpuSample

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _GpuDevice:
    index: int
    handle: object
    uuid: str


@dataclass(frozen=True)
class _GpuProcess:
    pid: int
    name: str
    memory_bytes: int


class _GpuProvider(Protocol):
    name: str

    def initialize(self) -> list[_GpuDevice]: ...

    def read_device(self, handle: object) -> tuple[int, int, int]: ...

    def read_processes(self, handle: object) -> list[_GpuProcess]: ...

    def shutdown(self) -> None: ...


def _text(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


class _NvmlProvider:
    name = "NVML"

    def __init__(self, api):
        self._api = api
        self._initialized = False

    def initialize(self) -> list[_GpuDevice]:
        try:
            self._api.nvmlInit()
            self._initialized = True
            count = self._api.nvmlDeviceGetCount()
            if count == 0:
                raise RuntimeError("no NVML devices")
            devices = []
            for index in range(count):
                handle = self._api.nvmlDeviceGetHandleByIndex(index)
                devices.append(_GpuDevice(index=index, handle=handle, uuid=_text(self._api.nvmlDeviceGetUUID(handle))))
            return devices
        except Exception:
            self._shutdown_after_init_failure()
            raise

    def read_device(self, handle: object) -> tuple[int, int, int]:
        util = int(self._api.nvmlDeviceGetUtilizationRates(handle).gpu)
        mem_mb = int(self._api.nvmlDeviceGetMemoryInfo(handle).used) >> 20
        power_w = int(self._api.nvmlDeviceGetPowerUsage(handle)) // 1000
        return util, mem_mb, power_w

    def read_processes(self, handle: object) -> list[_GpuProcess]:
        processes = []
        for process in self._api.nvmlDeviceGetComputeRunningProcesses(handle):
            pid = int(process.pid)
            try:
                name = _text(self._api.nvmlSystemGetProcessName(pid))
            except Exception:
                name = f"pid {pid}"
            processes.append(_GpuProcess(pid=pid, name=name, memory_bytes=int(process.usedGpuMemory or 0)))
        return processes

    def shutdown(self) -> None:
        if not self._initialized:
            return
        self._initialized = False
        self._api.nvmlShutdown()

    def _shutdown_after_init_failure(self) -> None:
        try:
            self.shutdown()
        except Exception:
            logger.debug("NVML shutdown after initialization failure failed", exc_info=True)


class _AmdSmiProvider:
    name = "AMD SMI"

    def __init__(self, api):
        self._api = api
        self._initialized = False

    def initialize(self) -> list[_GpuDevice]:
        try:
            self._api.amdsmi_init()
            self._initialized = True
            handles = self._api.amdsmi_get_processor_handles()
            if not handles:
                raise RuntimeError("no AMD SMI devices")
            # Keep the SMI slot as the dashboard lane id. Ray numbers the GPU
            # resources visible to its node in the same sequential space; do
            # not re-apply HIP/CUDA visibility variables or re-index by KFD id.
            return [
                _GpuDevice(
                    index=index,
                    handle=handle,
                    uuid=_text(self._api.amdsmi_get_gpu_device_uuid(handle)),
                )
                for index, handle in enumerate(handles)
            ]
        except Exception:
            self._shutdown_after_init_failure()
            raise

    def read_device(self, handle: object) -> tuple[int, int, int]:
        activity = self._api.amdsmi_get_gpu_activity(handle)
        util = _bounded_int(activity["gfx_activity"], field="gfx_activity", upper=100)

        # Unlike NVML, this AMD SMI API reports both values in MiB already.
        vram = self._api.amdsmi_get_gpu_vram_usage(handle)
        mem_mb = _bounded_int(vram["vram_used"], field="vram_used")

        power = self._api.amdsmi_get_power_info(handle)
        power_w = _amd_socket_power(power)
        return util, mem_mb, power_w

    def read_processes(self, handle: object) -> list[_GpuProcess]:
        processes = []
        for process in self._api.amdsmi_get_gpu_process_list(handle):
            pid = int(process["pid"])
            raw_name = process.get("name")
            name = str(raw_name) if raw_name and raw_name != "N/A" else f"pid {pid}"
            memory = process.get("memory_usage") or {}
            processes.append(_GpuProcess(pid=pid, name=name, memory_bytes=int(memory.get("vram_mem") or 0)))
        return processes

    def shutdown(self) -> None:
        if not self._initialized:
            return
        self._initialized = False
        self._api.amdsmi_shut_down()

    def _shutdown_after_init_failure(self) -> None:
        try:
            self.shutdown()
        except Exception:
            logger.debug("AMD SMI shutdown after initialization failure failed", exc_info=True)


def _bounded_int(value, *, field: str, upper: int | None = None) -> int:
    if value is None or isinstance(value, (str, bool)):
        raise ValueError(f"unsupported {field}: {value!r}")
    result = int(value)
    if result < 0 or (upper is not None and result > upper):
        raise ValueError(f"invalid {field}: {value!r}")
    return result


def _amd_socket_power(power: dict) -> int:
    # socket_power selects current power on MI300+ and average power on older
    # cards. The fallbacks tolerate older Python wrappers with that field absent.
    for field in ("socket_power", "current_socket_power", "average_socket_power"):
        try:
            value = power.get(field)
            if value == 0xFFFF:  # AMD SMI's raw uint16 "not available" sentinel
                raise ValueError(f"unsupported {field}: {value!r}")
            return _bounded_int(value, field=field, upper=100_000)
        except ValueError:
            continue
    raise ValueError(f"AMD SMI socket power unavailable: {power!r}")


def _import_nvml():
    import pynvml

    return pynvml


def _import_amdsmi():
    import amdsmi

    return amdsmi


class GpuSampler:
    FLUSH_INTERVAL_SECONDS: ClassVar[float] = 5.0
    # Per-process memory breakdown is a coarser, heavier SMI call (enumerates
    # every process) than the plain util/mem read, so it samples on its own,
    # slower cadence rather than every `interval` tick.
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
        self._devices: list[_GpuDevice] = []
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
            candidates = [("NVML", lambda: _NvmlProvider(nvml))]
        elif amdsmi is not None:
            candidates = [("AMD SMI", lambda: _AmdSmiProvider(amdsmi))]
        else:
            candidates = [
                ("NVML", lambda: _NvmlProvider(_import_nvml())),
                ("AMD SMI", lambda: _AmdSmiProvider(_import_amdsmi())),
            ]

        failures = []
        for name, make_provider in candidates:
            try:
                provider = make_provider()
                devices = provider.initialize()
            except Exception as error:
                failures.append(f"{name}: {error}")
                logger.debug("%s unavailable on %s", name, self.node, exc_info=True)
                continue
            self._provider = provider
            self._devices = devices
            self._uuids = [device.uuid for device in devices]
            return True

        detail = "; ".join(failures)
        if len(candidates) == 1:
            logger.warning(
                "%s unavailable on %s (%s); GPU utilization will not be collected",
                candidates[0][0],
                self.node,
                detail,
            )
        else:
            logger.warning(
                "GPU telemetry unavailable on %s (%s); GPU utilization will not be collected", self.node, detail
            )
        return False

    # ------------------------------ lifecycle -------------------------------

    def gpu_uuids(self) -> list[str]:
        return list(self._uuids)

    def start(self) -> bool:
        """Begin sampling; returns False (and stays inert) without a telemetry provider."""
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
            if self._thread.is_alive():
                # A vendor call can block during a driver reset. Keep the
                # provider initialized until that call returns rather than
                # racing its native library teardown from this thread.
                self.available = False
                self.flush()
                logger.warning(
                    "%s sampler thread did not stop on %s; skipping provider shutdown", self._provider.name, self.node
                )
                return
        try:
            self.flush()
        finally:
            self._shutdown_provider()

    def _shutdown_provider(self) -> None:
        provider, self._provider = self._provider, None
        self.available = False
        if provider is None:
            return
        try:
            provider.shutdown()
        except Exception:
            logger.warning("%s shutdown failed on %s", provider.name, self.node, exc_info=True)

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
        assert self._provider is not None
        count = 0
        for device in self._devices:
            try:
                util, mem_mb, power_w = self._provider.read_device(device.handle)
            except Exception:
                self._warner.warn(
                    f"{self._provider.name} read failed for gpu {device.index} on {self.node}; skipping this tick"
                )
                continue
            with self._buffer_lock:
                self._buffer.append(
                    GpuSample(
                        ts=ts,
                        node=self.node,
                        gpu=device.index,
                        util=util,
                        mem_mb=mem_mb,
                        power_w=power_w,
                    )
                )
            count += 1
        return count

    def sample_processes_once(self, ts: float) -> int:
        """Per-process VRAM breakdown once per GPU: who is actually holding
        the memory, not just the per-GPU aggregate ``sample_once`` reports."""
        if not self.available:
            return 0
        assert self._provider is not None
        count = 0
        for device in self._devices:
            try:
                processes = self._provider.read_processes(device.handle)
            except Exception:
                self._warner.warn(
                    f"{self._provider.name} process query failed for gpu {device.index} on {self.node}; "
                    "skipping this tick"
                )
                continue
            for process in processes:
                with self._buffer_lock:
                    self._process_buffer.append(
                        GpuProcessSample(
                            ts=ts,
                            node=self.node,
                            gpu=device.index,
                            pid=process.pid,
                            name=process.name,
                            mem_mb=process.memory_bytes >> 20,
                        )
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
