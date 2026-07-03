from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from pyatb.easy_use.input_models import CoreInputData


class SourceReader(ABC):
    source_name: str

    @classmethod
    @abstractmethod
    def detect(cls, path: Path) -> bool:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> CoreInputData:
        raise NotImplementedError

