from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
from ase import Atoms


@dataclass
class CoreInputData:
    source_type: str
    root_dir: Path
    out_dir: Optional[Path]
    stru_file: Optional[Path]

    ase_stru: Atoms
    lattice_vectors: np.ndarray
    lattice_constant: float

    nspin: int
    e_fermi: Optional[float]
    n_occu: Optional[int]
    n_bands: Optional[int]
    n_elec: Optional[int]

    hr_route: Optional[str]
    sr_route: Optional[str]
    rr_route: Optional[str]
    hr_unit: Optional[str]
    rr_unit: Optional[str]
    package_name: str = "ABACUS"

    pp_dir: Optional[Path] = None
    orb_dir: Optional[Path] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationRequest:
    modules: list[str]

    dim: str = "3"
    kmode: str = "line"
    kline_density: float = 0.01
    knum: int = 0
    kpath: Optional[str] = None
    tolerance: float = 1e-3

    mp_density: float = 0.10
    max_kpoint_num: int = 800

    energy_range: Optional[list[float]] = None
    omega_range: Optional[list[float]] = None
    band_range: Optional[list[int]] = None

    occu_switch: int = 0
    method: int = 0
    m_matrix: str = "1 0 0  0 1 0  0 0 1"
    valence: str = "auto"
