from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
from ase import Atoms


Z_VALENCE_PATTERNS = [
    re.compile(r'z[_\s]?valence\s*=\s*"?\s*([0-9]+(?:\.[0-9]*)?)', re.IGNORECASE),
    re.compile(r'<\s*z[_\s]?valence\s*>\s*([0-9]+(?:\.[0-9]*)?)\s*<\s*/\s*z[_\s]?valence\s*>', re.IGNORECASE),
    re.compile(r'^\s*z[_\s]?valence\s+([0-9]+(?:\.[0-9]*)?)\s*$', re.IGNORECASE | re.MULTILINE),
    re.compile(r'valence\s+charge[^0-9]*([0-9]+(?:\.[0-9]*)?)', re.IGNORECASE),
]


@lru_cache(maxsize=None)
def parse_upf_valence(upf_path: str | os.PathLike[str]) -> float:
    with open(upf_path, "r", encoding="utf-8", errors="ignore") as f:
        head = f.read(200000)
    for pattern in Z_VALENCE_PATTERNS:
        match = pattern.search(head)
        if match:
            return float(match.group(1))
    raise ValueError(f"unable to parse z_valence from UPF: {upf_path}")


def first_occurrence_species(ase_stru):
    order, seen = [], set()
    for symbol in ase_stru.get_chemical_symbols():
        if symbol not in seen:
            order.append(symbol)
            seen.add(symbol)
    return order

def kpath_generator(ase_stru: Atoms, kline_density=0.01, dim="3", tolerance=5e-4, knum=0, kpath=None):
    if dim == "2":
        bandpath = ase_stru.cell.bandpath(path=kpath, density=kline_density, eps=tolerance, pbc=[1, 1, 0])
    elif dim == "3":
        bandpath = ase_stru.cell.bandpath(path=kpath, density=kline_density, eps=tolerance)
    elif dim == "x":
        bandpath = ase_stru.cell.bandpath(path=kpath, density=kline_density, eps=tolerance, pbc=[1, 0, 0])
    elif dim == "y":
        bandpath = ase_stru.cell.bandpath(path=kpath, density=kline_density, eps=tolerance, pbc=[0, 1, 0])
    else:
        if " " in dim:
            pbc = [int(value) for value in dim.split()]
            bandpath = ase_stru.cell.bandpath(path=kpath, density=kline_density, eps=tolerance, pbc=pbc)
        else:
            raise ValueError(f"unsupported dim for kpath generation: {dim}")

    path_label = bandpath.path
    pattern = re.findall(r"[A-Z][0-9]*", path_label)
    path_label_array = list(pattern)
    cleaned_labels = [label for label in pattern if label != ","]

    special_points = {
        key: value if np.all((value >= -1) & (value <= 1)) else np.mod(value, 1)
        for key, value in bandpath.special_points.items()
    }

    kpt_output = []
    kpoint_label = []
    kpoint_num_in_line = []

    rec_lat_matrix = ase_stru.cell.reciprocal()[:]
    for idx, label in enumerate(path_label_array):
        label_next = path_label_array[idx + 1] if idx + 1 < len(path_label_array) else None
        coordinates = special_points.get(label)
        coordinates_next = special_points.get(label_next) if label_next else None
        if coordinates is not None and coordinates_next is not None:
            if knum == 0:
                k_real = coordinates @ rec_lat_matrix
                k_real_next = coordinates_next @ rec_lat_matrix
                distance = np.linalg.norm(k_real - k_real_next)
                density = max(int(distance * (2 * np.pi) / kline_density), 3)
                kpt_output.append(f"{'  '.join([f'{coord: .10f}' for coord in coordinates])}  {format(density, '<4')}   # {label}")
                kpoint_label.append(f"{label}   ")
                kpoint_num_in_line.append(f"{density}  ")
            else:
                kpt_output.append(f"{'  '.join([f'{coord: .10f}' for coord in coordinates])}  {format(knum, '<4')}   # {label}")
                kpoint_label.append(f"{label}   ")
                kpoint_num_in_line.append(f"{knum}  ")
        elif coordinates is None:
            continue
        else:
            kpt_output.append(f"{'  '.join([f'{coord: .10f}' for coord in coordinates])}  {format('1', '<4')}   # {label}")
            kpoint_label.append(f"{label}   ")
            kpoint_num_in_line.append(format("1", "<4"))

    kpt_output.append(f" kpoint_label{' ' * 14}{','.join(cleaned_labels)}")
    return kpt_output, kpoint_label, kpoint_num_in_line


def get_k_mesh_from_dim(lattice_vectors, dim, mesh_density):
    adaptive_density_ratio = 10
    if dim == "3":
        pbc = [1, 1, 1]
    elif dim == "2":
        pbc = [1, 1, 0]
    elif dim == "x":
        pbc = [1, 0, 0]
    elif dim == "y":
        pbc = [0, 1, 0]
    elif len(dim) == 3 and all(x in ["0", "1"] for x in dim):
        pbc = [int(x) for x in dim]
    else:
        raise ValueError("Invalid value for dim. Please provide 3/2/x/y or a 3-char string like 101.")

    reciprocal_lattice_vectors = 2 * np.pi * np.linalg.inv(lattice_vectors).T
    lengths = np.linalg.norm(reciprocal_lattice_vectors, axis=1)
    k_mesh = [max(1, int(length / mesh_density)) if p == 1 else 1 for length, p in zip(lengths, pbc)]

    integrate_grid = [k if p == 1 else 1 for k, p in zip(k_mesh, pbc)]
    adaptive_grid = [max(1, int(k / adaptive_density_ratio)) if p == 1 else 1 for k, p in zip(k_mesh, pbc)]
    adaptive_grid = [min(max(k, 5), 20) if p == 1 else 1 for k, p in zip(adaptive_grid, pbc)]
    return integrate_grid, adaptive_grid, pbc

def parse_energy_range(erange_value: Optional[str]) -> Optional[list[float]]:
    if not erange_value:
        return None
    value = erange_value.strip()
    if " " in value:
        return [float(v) for v in value.split()]
    num = float(value)
    return [-num, num]


def parse_omega_range(orange: Optional[list[float]]) -> Optional[list[float]]:
    if not orange:
        return None
    if len(orange) == 1:
        return [0.0, orange[0]]
    if len(orange) == 2:
        return [orange[0], orange[1]]
    raise ValueError("omega range should contain one or two numbers")


def parse_band_range(
    bandrange: Optional[str],
    n_occu: Optional[int],
    n_bands: Optional[int],
    default_window: int = 100,
) -> Optional[list[int]]:
    if n_occu is None or n_bands is None:
        return None
    if bandrange:
        if " " in bandrange:
            return [int(band) for band in bandrange.split()]
        if len(bandrange.split()) == 1:
            window = int(bandrange.split()[0])
            return [max(1, n_occu - window), min(n_bands, n_occu + window)]
    return [max(1, n_occu - default_window), min(n_bands, n_occu + default_window)]


def resolve_output_directory(root_dir: Path, output_arg: Optional[str]) -> Path:
    current_directory = Path.cwd().resolve()
    root_dir = root_dir.resolve()
    if output_arg is not None:
        output_dir = root_dir / output_arg
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    try:
        if current_directory != root_dir:
            current_directory.relative_to(root_dir)
            return current_directory
    except ValueError:
        pass
    output_dir = root_dir / "pyatb"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
