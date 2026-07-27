from __future__ import annotations

import csv
import io
import shutil
import subprocess
import threading
from types import TracebackType

from powerhouse_ai.schemas import GPUStatus

NVIDIA_QUERY = "name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu"


def parse_nvidia_smi_row(row: str) -> GPUStatus:
    values = next(csv.reader(io.StringIO(row), skipinitialspace=True))
    if len(values) != 6:
        raise ValueError(f"Expected 6 NVIDIA fields, received {len(values)}.")
    return GPUStatus(
        available=True,
        name=values[0].strip(),
        total_mb=int(values[1].strip()),
        used_mb=int(values[2].strip()),
        free_mb=int(values[3].strip()),
        utilization_percent=int(values[4].strip()),
        temperature_c=int(values[5].strip()),
    )


def get_gpu_status() -> GPUStatus:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return GPUStatus(available=False, error="nvidia-smi was not found.")
    try:
        result = subprocess.run(
            [
                executable,
                f"--query-gpu={NVIDIA_QUERY}",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        first_row = result.stdout.strip().splitlines()[0]
        return parse_nvidia_smi_row(first_row)
    except (subprocess.SubprocessError, ValueError, IndexError) as exc:
        return GPUStatus(available=False, error=str(exc))


class PeakVRAMTracker:
    def __init__(self, sample_interval_seconds: float = 0.1) -> None:
        if sample_interval_seconds <= 0:
            raise ValueError("VRAM sample interval must be positive.")
        self.sample_interval_seconds = sample_interval_seconds
        self.peak_mb: int | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def sample(self) -> GPUStatus:
        status = get_gpu_status()
        if status.used_mb is not None:
            with self._lock:
                self.peak_mb = max(self.peak_mb or 0, status.used_mb)
        return status

    def _run(self) -> None:
        while not self._stop.wait(self.sample_interval_seconds):
            self.sample()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self.sample()
        self._thread = threading.Thread(
            target=self._run,
            name="powerhouse-vram-sampler",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self.sample()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=6)
            self._thread = None

    def __enter__(self) -> PeakVRAMTracker:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.stop()
