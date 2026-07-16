"""Tee pipeline stdout/stderr to a log file under the run output root."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, TextIO


class _Tee(TextIO):
    def __init__(self, stream: TextIO, log_file: TextIO) -> None:
        self._stream = stream
        self._log_file = log_file

    def write(self, data: str) -> int:
        self._stream.write(data)
        self._log_file.write(data)
        return len(data)

    def flush(self) -> None:
        self._stream.flush()
        self._log_file.flush()

    def fileno(self) -> int:
        return self._stream.fileno()

    def isatty(self) -> bool:
        return self._stream.isatty()


@contextmanager
def pipeline_run_log(output_root: Path, *, filename: str = "pipeline.log") -> Iterator[Path]:
    """Mirror stdout/stderr to ``{output_root}/{filename}`` for the run duration."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / filename
    started = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n{'=' * 72}\n# pipeline run started {started}\n{'=' * 72}\n")
        log_file.flush()
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = _Tee(old_out, log_file)
        sys.stderr = _Tee(old_err, log_file)
        try:
            yield log_path
        finally:
            sys.stdout = old_out
            sys.stderr = old_err
            ended = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            log_file.write(f"\n# pipeline run finished {ended}\n")
            log_file.flush()
