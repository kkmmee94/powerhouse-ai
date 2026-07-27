import itertools
import threading

import powerhouse_ai.gpu
from powerhouse_ai.gpu import PeakVRAMTracker, parse_nvidia_smi_row
from powerhouse_ai.schemas import GPUStatus


def test_parse_nvidia_smi_row() -> None:
    status = parse_nvidia_smi_row("NVIDIA GeForce RTX 5060 Ti, 16311, 1200, 15111, 42, 55")
    assert status.available is True
    assert status.total_mb == 16311
    assert status.free_mb == 15111
    assert status.temperature_c == 55


def test_peak_vram_tracker_samples_periodically(monkeypatch) -> None:
    samples = itertools.count(100)
    periodic_sample_observed = threading.Event()

    def fake_status() -> GPUStatus:
        used_mb = next(samples)
        if used_mb >= 102:
            periodic_sample_observed.set()
        return GPUStatus(available=True, used_mb=used_mb)

    monkeypatch.setattr(powerhouse_ai.gpu, "get_gpu_status", fake_status)
    with PeakVRAMTracker(sample_interval_seconds=0.001) as tracker:
        assert periodic_sample_observed.wait(timeout=1)

    assert tracker.peak_mb is not None
    assert tracker.peak_mb >= 102
