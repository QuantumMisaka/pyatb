from __future__ import annotations

from pathlib import Path

from pyatb.easy_use.input_models import CoreInputData
from pyatb.easy_use.sources.abacus import AbacusReader
from pyatb.easy_use.sources.hamgnn import HamGNNReader


READERS = {
    "abacus": AbacusReader,
    "hamgnn": HamGNNReader,
}


def load_core_data(path: str, source_type: str | None = None) -> CoreInputData:
    resolved = Path(path).resolve()
    if source_type:
        return READERS[source_type].load(resolved)
    for reader in READERS.values():
        if reader.detect(resolved):
            return reader.load(resolved)
    raise ValueError(f"Cannot detect a supported source type from {resolved}")

