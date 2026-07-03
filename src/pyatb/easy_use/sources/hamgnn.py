from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from pyatb.easy_use.input_models import CoreInputData
from pyatb.easy_use.stru_analyzer import read_abacus_stru
from pyatb.easy_use.sources.base import SourceReader


def _build_hamgnn_routes(out_dir: Path, nspin: int) -> tuple[str, str, str, str, str]:
    if nspin == 2:
        hr_route = f"{out_dir / 'H0.csr'}, {out_dir / 'H1.csr'}"
    else:
        hr_route = str(out_dir / "H.csr")
    return hr_route, str(out_dir / "S.csr"), str(out_dir / "R.csr"), "eV", "Bohr"


def extract_hamgnn_properties(json_file: str | Path):
    with open(json_file, "r") as f:
        data = json.load(f)
    max_val = data.get("max_val", None)
    e_fermi = data.get("fermi_energy", None)
    band_gap = data.get("band_gap", None)
    n_elec = int(data.get("num_electrons", 0))
    n_occu = math.floor(n_elec / 2)
    return max_val, e_fermi, band_gap, n_elec, n_occu


class HamGNNReader(SourceReader):
    source_name = "hamgnn"

    @classmethod
    def detect(cls, path: Path) -> bool:
        if path.is_file():
            return path.name == "properties.json"
        return (path / "properties.json").exists()

    @classmethod
    def load(cls, path: Path) -> CoreInputData:
        path = path.resolve()
        root_dir = path.parent if path.is_file() else path
        properties_file = path if path.is_file() else root_dir / "properties.json"

        max_val, e_fermi, band_gap, n_elec, n_occu = extract_hamgnn_properties(properties_file)
        nspin = 1
        n_bands = int(n_elec)

        stru_file = root_dir / "STRU"
        with open(stru_file, "r") as f_stru:
            ase_stru = read_abacus_stru(f_stru, None, True)

        hr_route, sr_route, rr_route, hr_unit, rr_unit = _build_hamgnn_routes(root_dir, nspin)
        return CoreInputData(
            source_type="hamgnn",
            root_dir=root_dir,
            out_dir=root_dir,
            stru_file=stru_file,
            ase_stru=ase_stru,
            lattice_vectors=np.array(ase_stru.get_cell()),
            lattice_constant=1.0,
            nspin=nspin,
            e_fermi=e_fermi,
            n_occu=n_occu,
            n_bands=n_bands,
            n_elec=n_elec,
            hr_route=hr_route,
            sr_route=sr_route,
            rr_route=rr_route,
            hr_unit=hr_unit,
            rr_unit=rr_unit,
            package_name="ABACUS",
            metadata={"max_val": max_val, "band_gap": band_gap},
        )
