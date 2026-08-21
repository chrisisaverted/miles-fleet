import logging
import time

import pytest

from miles.dashboard import gpu_sampler as gpu_sampler_module
from miles.dashboard.gpu_sampler import GpuSampler
from miles.dashboard.store import GpuProcessSample, GpuSample


class FakeNvml:
    """Just enough of pynvml for the sampler: handle == device index."""

    def __init__(self, count=2, fail_init=False, fail_count=False, failing_devices=()):
        self.count = count
        self.fail_init = fail_init
        self.fail_count = fail_count
        self.failing_devices = set(failing_devices)
        self.init_calls = 0
        self.shutdown_calls = 0

    def nvmlInit(self):
        self.init_calls += 1
        if self.fail_init:
            raise RuntimeError("driver/library version mismatch")

    def nvmlShutdown(self):
        self.shutdown_calls += 1

    def nvmlDeviceGetCount(self):
        if self.fail_count:
            raise RuntimeError("device enumeration failed")
        return self.count

    def nvmlDeviceGetHandleByIndex(self, index):
        return index

    def nvmlDeviceGetUUID(self, handle):
        return f"GPU-fake-{handle}"

    def nvmlDeviceGetUtilizationRates(self, handle):
        if handle in self.failing_devices:
            raise RuntimeError("GPU is lost")
        return type("Util", (), {"gpu": 40 + handle})()

    def nvmlDeviceGetMemoryInfo(self, handle):
        return type("Mem", (), {"used": (handle + 1) * 1024 * 1024 * 1024})()  # GiB in bytes

    def nvmlDeviceGetPowerUsage(self, handle):
        return 600_000 + handle  # milliwatts

    def nvmlDeviceGetComputeRunningProcesses(self, handle):
        if handle in self.failing_devices:
            raise RuntimeError("GPU is lost")
        return [type("Proc", (), {"pid": 1000 + handle, "usedGpuMemory": (handle + 1) * 512 * 1024 * 1024})()]

    def nvmlSystemGetProcessName(self, pid):
        return f"proc-{pid}"


class FakeAmdSmi:
    """AMD SMI API shapes used by ROCm 7.0 and 7.2; handle == device index."""

    def __init__(
        self,
        count=2,
        *,
        fail_init=False,
        fail_enumeration=False,
        fail_uuid_devices=(),
        failing_devices=(),
    ):
        self.count = count
        self.fail_init = fail_init
        self.fail_enumeration = fail_enumeration
        self.fail_uuid_devices = set(fail_uuid_devices)
        self.failing_devices = set(failing_devices)
        self.init_calls = 0
        self.shutdown_calls = 0

    def amdsmi_init(self):
        self.init_calls += 1
        if self.fail_init:
            raise RuntimeError("AMD SMI initialization failed")

    def amdsmi_shut_down(self):
        self.shutdown_calls += 1

    def amdsmi_get_processor_handles(self):
        if self.fail_enumeration:
            raise RuntimeError("device enumeration failed")
        return list(range(self.count))

    def amdsmi_get_gpu_device_uuid(self, handle):
        if handle in self.fail_uuid_devices:
            raise RuntimeError("UUID unavailable")
        return f"GPU-amd-{handle}"

    def amdsmi_get_gpu_activity(self, handle):
        if handle in self.failing_devices:
            raise RuntimeError("GPU is lost")
        return {"gfx_activity": 70 + handle, "umc_activity": 10, "mm_activity": 0}

    def amdsmi_get_gpu_vram_usage(self, handle):
        # AMD SMI reports vram_used in MiB, not bytes.
        return {"vram_total": 192 * 1024, "vram_used": (handle + 1) * 2048}

    def amdsmi_get_power_info(self, handle):
        # MI300+ exposes current socket power through socket_power. Keep the
        # legacy field different so the test catches selecting the wrong key.
        return {
            "socket_power": 500 + handle,
            "current_socket_power": 500 + handle,
            "average_socket_power": 250 + handle,
        }

    def amdsmi_get_gpu_process_list(self, handle):
        if handle in self.failing_devices:
            raise RuntimeError("GPU is lost")
        return [
            {
                "pid": 2000 + handle,
                "name": f"amd-proc-{handle}" if handle == 0 else "N/A",
                "memory_usage": {"vram_mem": (handle + 1) * 768 * 1024 * 1024},
            }
        ]


class PushSpy:
    def __init__(self):
        self.calls: list[tuple[str, list[GpuSample]]] = []

    def __call__(self, node, batch):
        self.calls.append((node, batch))


class ProcessPushSpy:
    def __init__(self):
        self.calls: list[tuple[str, list[GpuProcessSample]]] = []

    def __call__(self, node, batch):
        self.calls.append((node, batch))


def test_sample_once_converts_units():
    push = PushSpy()
    sampler = GpuSampler(push, node="10.0.0.1", nvml=FakeNvml(count=2))
    assert sampler.available
    assert sampler.gpu_uuids() == ["GPU-fake-0", "GPU-fake-1"]

    assert sampler.sample_once(ts=10.0) == 2
    sampler.flush()
    [(node, batch)] = push.calls
    assert node == "10.0.0.1"
    assert batch == [
        GpuSample(ts=10.0, node="10.0.0.1", gpu=0, util=40, mem_mb=1024, power_w=600),
        GpuSample(ts=10.0, node="10.0.0.1", gpu=1, util=41, mem_mb=2048, power_w=600),
    ]


def test_amd_sample_once_preserves_native_units_and_uuids():
    push = PushSpy()
    amdsmi = FakeAmdSmi(count=2)
    sampler = GpuSampler(push, node="amd-node", amdsmi=amdsmi)
    assert sampler.available
    assert sampler.gpu_uuids() == ["GPU-amd-0", "GPU-amd-1"]

    assert sampler.sample_once(ts=11.0) == 2
    sampler.flush()
    [(node, batch)] = push.calls
    assert node == "amd-node"
    assert batch == [
        GpuSample(ts=11.0, node="amd-node", gpu=0, util=70, mem_mb=2048, power_w=500),
        GpuSample(ts=11.0, node="amd-node", gpu=1, util=71, mem_mb=4096, power_w=501),
    ]

    sampler.stop()
    assert amdsmi.shutdown_calls == 1


def test_amd_socket_power_falls_back_from_unavailable_unified_field():
    push = PushSpy()
    amdsmi = FakeAmdSmi(count=1)
    amdsmi.amdsmi_get_power_info = lambda handle: {
        "socket_power": 0xFFFF,
        "current_socket_power": 475,
        "average_socket_power": 200,
    }
    sampler = GpuSampler(push, node="n", amdsmi=amdsmi)

    assert sampler.sample_once(ts=1.0) == 1
    sampler.flush()
    assert push.calls[0][1][0].power_w == 475
    sampler.stop()


def test_flush_clears_buffer_and_skips_empty():
    push = PushSpy()
    sampler = GpuSampler(push, node="n", nvml=FakeNvml(count=1))
    sampler.flush()  # empty: no call
    assert push.calls == []

    sampler.sample_once(ts=1.0)
    sampler.flush()
    sampler.flush()  # cleared: no duplicate push
    assert len(push.calls) == 1


def test_nvml_init_failure_disables_sampler(caplog):
    push = PushSpy()
    with caplog.at_level(logging.WARNING):
        sampler = GpuSampler(push, node="n", nvml=FakeNvml(fail_init=True))
    assert not sampler.available
    assert sampler.start() is False
    assert sampler.sample_once(ts=1.0) == 0
    assert push.calls == []
    assert any("NVML unavailable" in r.message for r in caplog.records)


@pytest.mark.parametrize("provider_name", ["nvml", "amdsmi"])
def test_partial_initialization_failure_shuts_down_provider(provider_name):
    if provider_name == "nvml":
        provider = FakeNvml(fail_count=True)
    else:
        provider = FakeAmdSmi(fail_enumeration=True)

    sampler = GpuSampler(PushSpy(), node="n", **{provider_name: provider})

    assert not sampler.available
    assert provider.init_calls == 1
    assert provider.shutdown_calls == 1
    sampler.stop()
    assert provider.shutdown_calls == 1


def test_amd_uuid_failure_after_init_shuts_down_provider():
    amdsmi = FakeAmdSmi(count=2, fail_uuid_devices={1})

    sampler = GpuSampler(PushSpy(), node="n", amdsmi=amdsmi)

    assert not sampler.available
    assert sampler.gpu_uuids() == []
    assert amdsmi.shutdown_calls == 1


def test_production_auto_detection_falls_back_from_nvml_to_amdsmi(monkeypatch):
    nvml = FakeNvml(fail_init=True)
    amdsmi = FakeAmdSmi(count=1)
    monkeypatch.setattr(gpu_sampler_module, "_import_nvml", lambda: nvml)
    monkeypatch.setattr(gpu_sampler_module, "_import_amdsmi", lambda: amdsmi)

    sampler = GpuSampler(PushSpy(), node="n")

    assert sampler.available
    assert sampler.gpu_uuids() == ["GPU-amd-0"]
    assert nvml.init_calls == 1
    assert amdsmi.init_calls == 1
    sampler.stop()
    assert amdsmi.shutdown_calls == 1


def test_explicit_injection_does_not_probe_the_other_backend(monkeypatch):
    monkeypatch.setattr(
        gpu_sampler_module,
        "_import_amdsmi",
        lambda: pytest.fail("explicit NVML injection must not probe AMD SMI"),
    )

    sampler = GpuSampler(PushSpy(), node="n", nvml=FakeNvml(fail_init=True))

    assert not sampler.available


def test_missing_optional_backends_disable_sampler(monkeypatch, caplog):
    def missing_nvml():
        raise ImportError("pynvml is not installed")

    def missing_amdsmi():
        raise ImportError("amdsmi is not installed")

    monkeypatch.setattr(gpu_sampler_module, "_import_nvml", missing_nvml)
    monkeypatch.setattr(gpu_sampler_module, "_import_amdsmi", missing_amdsmi)
    with caplog.at_level(logging.WARNING):
        sampler = GpuSampler(PushSpy(), node="n")

    assert not sampler.available
    assert sampler.start() is False
    assert any("GPU telemetry unavailable" in record.message for record in caplog.records)


def test_only_one_backend_may_be_injected():
    with pytest.raises(AssertionError, match="inject only one"):
        GpuSampler(PushSpy(), node="n", nvml=FakeNvml(), amdsmi=FakeAmdSmi())


def test_failing_device_is_skipped_others_report(caplog):
    push = PushSpy()
    sampler = GpuSampler(push, node="n", nvml=FakeNvml(count=3, failing_devices={1}))
    with caplog.at_level(logging.WARNING):
        assert sampler.sample_once(ts=1.0) == 2
    sampler.flush()
    [(_, batch)] = push.calls
    assert [s.gpu for s in batch] == [0, 2]
    assert any("skipping this tick" in r.message for r in caplog.records)


def test_amd_failing_device_is_skipped_while_others_report(caplog):
    push = PushSpy()
    sampler = GpuSampler(push, node="n", amdsmi=FakeAmdSmi(count=3, failing_devices={1}))
    with caplog.at_level(logging.WARNING):
        assert sampler.sample_once(ts=1.0) == 2
    sampler.flush()
    [(_, batch)] = push.calls
    assert [sample.gpu for sample in batch] == [0, 2]
    assert any("AMD SMI read failed for gpu 1" in record.message for record in caplog.records)
    sampler.stop()


def test_amd_uses_smi_visible_order_without_refiltering_process_env(monkeypatch):
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "2")
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "2")
    push = PushSpy()
    sampler = GpuSampler(push, node="n", amdsmi=FakeAmdSmi(count=3))

    assert sampler.sample_once(ts=1.0) == 3
    sampler.flush()
    assert [sample.gpu for sample in push.calls[0][1]] == [0, 1, 2]
    sampler.stop()


def test_thread_lifecycle_flushes_on_stop():
    push = PushSpy()
    sampler = GpuSampler(push, node="n", interval=0.01, nvml=FakeNvml(count=1))
    assert sampler.start() is True
    time.sleep(0.08)
    sampler.stop()
    assert push.calls, "stop() must flush buffered samples"
    total = sum(len(batch) for _, batch in push.calls)
    assert total >= 3  # ~8 ticks at 10ms; generous margin against scheduler jitter


def test_stop_is_idempotent_and_flushes_once():
    push = PushSpy()
    amdsmi = FakeAmdSmi(count=1)
    sampler = GpuSampler(push, node="n", amdsmi=amdsmi)
    sampler.sample_once(ts=1.0)

    sampler.stop()
    sampler.stop()

    assert len(push.calls) == 1
    assert amdsmi.shutdown_calls == 1
    assert not sampler.available
    assert sampler.sample_once(ts=2.0) == 0


def test_sample_processes_once_converts_units():
    push = PushSpy()
    push_processes = ProcessPushSpy()
    sampler = GpuSampler(push, node="n", nvml=FakeNvml(count=2), push_processes=push_processes)
    assert sampler.sample_processes_once(ts=5.0) == 2
    sampler.flush()
    [(node, batch)] = push_processes.calls
    assert node == "n"
    assert batch == [
        GpuProcessSample(ts=5.0, node="n", gpu=0, pid=1000, name="proc-1000", mem_mb=512),
        GpuProcessSample(ts=5.0, node="n", gpu=1, pid=1001, name="proc-1001", mem_mb=1024),
    ]


def test_amd_processes_use_nested_vram_bytes_and_name_fallback():
    push_processes = ProcessPushSpy()
    sampler = GpuSampler(
        PushSpy(),
        node="n",
        amdsmi=FakeAmdSmi(count=2),
        push_processes=push_processes,
    )

    assert sampler.sample_processes_once(ts=5.0) == 2
    sampler.flush()
    [(node, batch)] = push_processes.calls
    assert node == "n"
    assert batch == [
        GpuProcessSample(ts=5.0, node="n", gpu=0, pid=2000, name="amd-proc-0", mem_mb=768),
        GpuProcessSample(ts=5.0, node="n", gpu=1, pid=2001, name="pid 2001", mem_mb=1536),
    ]
    sampler.stop()


def test_failing_device_skipped_for_process_sampling(caplog):
    push = PushSpy()
    push_processes = ProcessPushSpy()
    sampler = GpuSampler(push, node="n", nvml=FakeNvml(count=3, failing_devices={1}), push_processes=push_processes)
    with caplog.at_level(logging.WARNING):
        assert sampler.sample_processes_once(ts=1.0) == 2
    sampler.flush()
    [(_, batch)] = push_processes.calls
    assert [s.gpu for s in batch] == [0, 2]
    assert any("skipping this tick" in r.message for r in caplog.records)


def test_amd_failing_device_is_skipped_for_process_sampling(caplog):
    push_processes = ProcessPushSpy()
    sampler = GpuSampler(
        PushSpy(),
        node="n",
        amdsmi=FakeAmdSmi(count=3, failing_devices={1}),
        push_processes=push_processes,
    )
    with caplog.at_level(logging.WARNING):
        assert sampler.sample_processes_once(ts=1.0) == 2
    sampler.flush()
    [(_, batch)] = push_processes.calls
    assert [sample.gpu for sample in batch] == [0, 2]
    assert any("AMD SMI process query failed for gpu 1" in record.message for record in caplog.records)
    sampler.stop()


def test_process_batch_dropped_silently_without_push_processes():
    # sampling still works if called directly, but flush() must not crash or
    # invent a push destination — the buffer is just discarded (design: the
    # feature is a no-op end-to-end when the collector never wires it up)
    push = PushSpy()
    sampler = GpuSampler(push, node="n", nvml=FakeNvml(count=1))
    assert sampler.sample_processes_once(ts=1.0) == 1
    sampler.flush()
    assert push.calls == []


def test_interval_must_be_positive():
    with pytest.raises(AssertionError):
        GpuSampler(lambda n, b: None, node="n", interval=0, nvml=FakeNvml())


def test_real_nvml_when_gpus_present():
    # Guards the FakeNvml against drifting from the real pynvml API surface;
    # runs wherever a GPU + driver exist (devbox/CI-GPU), skips elsewhere.
    pynvml = pytest.importorskip("pynvml")
    try:
        pynvml.nvmlInit()
        pynvml.nvmlShutdown()
    except Exception:
        pytest.skip("no usable NVML device")

    push = PushSpy()
    push_processes = ProcessPushSpy()
    sampler = GpuSampler(push, node="local", nvml=pynvml, push_processes=push_processes)
    assert sampler.available
    assert sampler.sample_once(ts=1.0) >= 1
    # idle test GPUs may have zero compute processes — asserting >= 0 just
    # guards that the real nvmlDeviceGetComputeRunningProcesses call doesn't raise
    assert sampler.sample_processes_once(ts=1.0) >= 0
    sampler.flush()
    [(_, batch)] = push.calls
    sample = batch[0]
    assert 0 <= sample.util <= 100
    assert sample.mem_mb >= 0 and sample.power_w >= 0
    assert sampler.gpu_uuids()[0].startswith("GPU-")
    if push_processes.calls:
        proc_sample = push_processes.calls[0][1][0]
        assert proc_sample.pid > 0 and proc_sample.mem_mb >= 0 and proc_sample.name
    sampler.stop()


def test_real_amdsmi_when_gpus_present():
    # Guards the fake against the API installed by the ROCm dashboard image;
    # skips on the ordinary CPU/NVIDIA fast-test workers.
    amdsmi = pytest.importorskip("amdsmi")
    push = PushSpy()
    push_processes = ProcessPushSpy()
    sampler = GpuSampler(push, node="local", amdsmi=amdsmi, push_processes=push_processes)
    if not sampler.available:
        pytest.skip("no usable AMD SMI device")

    try:
        assert sampler.sample_once(ts=1.0) >= 1
        assert sampler.sample_processes_once(ts=1.0) >= 0
        sampler.flush()
        [(_, batch)] = push.calls
        sample = batch[0]
        assert 0 <= sample.util <= 100
        assert sample.mem_mb >= 0 and sample.power_w >= 0
        assert sampler.gpu_uuids()[0]
        if push_processes.calls:
            proc_sample = push_processes.calls[0][1][0]
            assert proc_sample.pid > 0 and proc_sample.mem_mb >= 0 and proc_sample.name
    finally:
        sampler.stop()
