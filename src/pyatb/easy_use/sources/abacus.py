from __future__ import annotations

import re
from os import PathLike
from pathlib import Path

import numpy as np
from ase import Atoms

from pyatb.easy_use.input_models import CoreInputData
from pyatb.easy_use.stru_analyzer import read_abacus_stru
from pyatb.easy_use.sources.base import SourceReader


def parse_abacus_input(input_text: str) -> dict[str, str]:
    variables = {}
    for line in input_text.splitlines():
        line_content = line.split("#", 1)[0].strip()
        if not line_content:
            continue
        parts = line_content.split(maxsplit=1)
        if len(parts) == 2:
            key, value = parts
            variables[key] = value
    return variables


def load_abacus_input(path_like: str | PathLike[str]) -> dict[str, str]:
    path = Path(path_like)
    try:
        return parse_abacus_input(path.read_text())
    except FileNotFoundError:
        print(f"文件 {path} 未找到。")
        return {}


def extract_abacus_data_from_log(out_suffix_path: str | PathLike[str]):
    log_file_path = Path(out_suffix_path) / "running_scf.log"
    if not log_file_path.is_file():
        print(f"错误：未找到日志文件 {log_file_path}")
        return

    e_tot = None
    e_fermi = None
    n_occu = None
    n_bands = None
    n_elec = None
    post_occu = None
    post_bands = None
    post_elec = None
    after_nelec_delta = False

    re_etot = re.compile(r"!FINAL_ETOT_IS\s+([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)")
    re_fermi = re.compile(r"EFERMI\s*=\s*([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)")
    re_occu = re.compile(r"(?:occupied\s+bands|occupied\s+electronic\s+states)\s*=\s*(\d+)", re.IGNORECASE)
    re_bands = re.compile(r"NBANDS[^\d]*([0-9]+)", re.IGNORECASE)
    re_nelec_now = re.compile(r"nelec\s+now\s*[:=]\s*=?\s*([+-]?\d+(?:\.\d+)?)", re.IGNORECASE)
    re_autoset_ele = re.compile(r"AUTOSET\s+(?:the\s+)?number\s+of\s+electrons\s*[:=]?\s*=?\s*([+-]?\d+(?:\.\d+)?)", re.IGNORECASE)
    re_num_elec = re.compile(r"number\s+of\s+electrons\s*[:=]?\s*=?\s*([+-]?\d+(?:\.\d+)?)", re.IGNORECASE)
    re_nelec_delta_warn = re.compile(r"nelec_delta\s+is\s+NOT\s+zero", re.IGNORECASE)
    re_element_elec = re.compile(r"total\s+electron\s+number\s+of\s+element\s+\S+\s*=\s*\d+", re.IGNORECASE)

    e_fermi_from_EFermi_lines = None

    try:
        with log_file_path.open("r", encoding="utf-8", errors="ignore") as log_file:
            for raw_line in log_file:
                line = raw_line.strip()

                m = re_etot.search(line)
                if m:
                    e_tot = float(m.group(1))
                    continue

                m = re_fermi.search(line)
                if m:
                    e_fermi = float(m.group(1))
                    continue

                if "E_Fermi" in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            e_fermi_from_EFermi_lines = float(parts[-1])
                        except ValueError:
                            pass

                if re_nelec_delta_warn.search(line):
                    after_nelec_delta = True
                    continue

                if after_nelec_delta:
                    m = re_occu.search(line)
                    if m:
                        post_occu = int(m.group(1))
                        continue
                    m = re_bands.search(line)
                    if m:
                        post_bands = int(m.group(1))
                        continue
                    m = re_nelec_now.search(line)
                    if m:
                        post_elec = int(float(m.group(1)))
                        continue
                    m = re_autoset_ele.search(line)
                    if m:
                        if post_elec is None:
                            post_elec = int(float(m.group(1)))
                        continue
                    m = re_num_elec.search(line)
                    if m:
                        if not re_element_elec.search(line) and post_elec is None:
                            try:
                                post_elec = int(float(m.group(1)))
                            except ValueError:
                                pass
                        continue
                else:
                    m = re_occu.search(line)
                    if m:
                        n_occu = int(m.group(1))
                        continue
                    m = re_bands.search(line)
                    if m:
                        n_bands = int(m.group(1))
                        continue
                    m = re_nelec_now.search(line)
                    if m:
                        n_elec = int(float(m.group(1)))
                        continue
                    m = re_autoset_ele.search(line)
                    if m:
                        if n_elec is None:
                            n_elec = int(float(m.group(1)))
                        continue
                    m = re_num_elec.search(line)
                    if m:
                        if not re_element_elec.search(line) and n_elec is None:
                            try:
                                n_elec = int(float(m.group(1)))
                            except ValueError:
                                pass
                        continue

        if after_nelec_delta:
            if post_occu is not None:
                n_occu = post_occu
            if post_bands is not None:
                n_bands = post_bands
            if post_elec is not None:
                n_elec = post_elec

        if e_fermi is None and e_fermi_from_EFermi_lines is not None:
            e_fermi = e_fermi_from_EFermi_lines

        if e_tot is None or e_fermi is None:
            print("错误：无法从 running_scf.log 中提取 e_tot 或 e_fermi")
            return

        print(f"{log_file_path} 提取完成。")
        print(f"E_TOTAL (eV)   =  {e_tot}")
        print(f"E_FERMI (eV)   =  {e_fermi}")
        print(f"OCCUPIED_BANDS =  {n_occu}")
        print(f"NBANDS         =  {n_bands}")
        print(f"N_ELEC         =  {n_elec}")
        return e_tot, e_fermi, n_occu, n_bands, n_elec
    except Exception as exc:
        print(f"读取/解析日志时发生异常：{exc}")
        return


def rewrite_abacus_stru_paths(pseudo_dir, orbital_dir, f_stru, ase_stru: Atoms, f_out=None):
    pseudo_dir = Path(pseudo_dir)
    orbital_dir = Path(orbital_dir)
    f_stru = Path(f_stru)
    f_out = f_stru if f_out is None else Path(f_out)

    with f_stru.open("r") as file:
        content = file.read()

    atoms_pp = ase_stru.info["pp"]
    atoms_orb = ase_stru.info["basis"]
    atoms_all = ase_stru.get_chemical_symbols()
    atoms_dict = {}
    for idx, atom_name in enumerate(atoms_all):
        if atom_name not in atoms_dict:
            atoms_dict[atom_name] = [ase_stru.get_masses()[idx], atoms_pp[atom_name], atoms_orb[atom_name]]

    for key, values in atoms_dict.items():
        element = key
        mass, pp, orb = values
        pp_filename = pseudo_dir / pp
        orb_filename = orbital_dir / orb
        content = re.sub(rf"{element}\s+\d+\.\d+\s+.*\.upf", f"{element} {mass} {pp_filename}", content)
        content = re.sub(rf"{element}_[^/\s]+\.orb", f"{orb_filename}", content)

    with f_out.open("w") as file:
        file.write(content)


def _resolve_abacus_paths(path: Path) -> tuple[Path, Path, Path]:
    if path.is_dir():
        if path.name.startswith("OUT."):
            out_dir = path
            return out_dir / "INPUT", out_dir.parent, out_dir
        input_file = path / "INPUT"
        root_dir = path
        suffix = load_abacus_input(input_file).get("suffix", "ABACUS")
        return input_file, root_dir, root_dir / f"OUT.{suffix}"
    if path.is_file():
        if path.parent.name.startswith("OUT.") and path.name == "INPUT":
            out_dir = path.parent
            return path, out_dir.parent, out_dir
        root_dir = path.parent
        suffix = load_abacus_input(path).get("suffix", "ABACUS")
        return path, root_dir, root_dir / f"OUT.{suffix}"
    raise FileNotFoundError(f"ABACUS path not found: {path}")


def _resolve_optional_dir(root_dir: Path, path_value: str | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    return path if path.is_absolute() else root_dir / path


def _resolve_input_file(root_dir: Path, path_value: str | None, default_name: str) -> Path:
    path = Path(default_name if not path_value else path_value)
    return path if path.is_absolute() else root_dir / path


def _build_abacus_routes(out_dir: Path, nspin: int) -> tuple[str, str, str, str, str]:
    old_hr_spin0 = out_dir / "data-HR-sparse_SPIN0.csr"
    new_hr1 = out_dir / "hrs1_nao.csr"
    if new_hr1.exists():
        if nspin == 2:
            hr_route = f"{out_dir / 'hrs1_nao.csr'}, {out_dir / 'hrs2_nao.csr'}"
        else:
            hr_route = str(out_dir / "hrs1_nao.csr")
        return hr_route, str(out_dir / "srs1_nao.csr"), str(out_dir / "rr.csr"), "Ry", "Bohr"
    if old_hr_spin0.exists():
        if nspin == 2:
            hr_route = f"{out_dir / 'data-HR-sparse_SPIN0.csr'}, {out_dir / 'data-HR-sparse_SPIN1.csr'}"
        else:
            hr_route = str(out_dir / "data-HR-sparse_SPIN0.csr")
        return hr_route, str(out_dir / "data-SR-sparse_SPIN0.csr"), str(out_dir / "data-rR-sparse.csr"), "Ry", "Bohr"
    raise FileNotFoundError(f"Cannot find ABACUS HR files in {out_dir}")


class AbacusReader(SourceReader):
    source_name = "abacus"

    @classmethod
    def detect(cls, path: Path) -> bool:
        if path.is_file():
            return path.name == "INPUT"
        return (path / "INPUT").exists()

    @classmethod
    def load(cls, path: Path) -> CoreInputData:
        input_file, root_dir, out_dir = _resolve_abacus_paths(path.resolve())
        variables_dict_full = load_abacus_input(out_dir / "INPUT")
        latname = variables_dict_full["latname"]
        nspin = int(variables_dict_full["nspin"])
        pp_dir = _resolve_optional_dir(root_dir, variables_dict_full.get("pseudo_dir"))
        orb_dir = _resolve_optional_dir(root_dir, variables_dict_full.get("orbital_dir"))

        e_tot, e_fermi, n_occu, n_bands, n_elec = extract_abacus_data_from_log(out_dir)

        stru_file = _resolve_input_file(root_dir, variables_dict_full.get("stru_file"), "STRU")
        with stru_file.open("r") as f_stru:
            i_latname = None if latname == "none" else latname
            ase_stru = read_abacus_stru(f_stru, i_latname, True)

        hr_route, sr_route, rr_route, hr_unit, rr_unit = _build_abacus_routes(out_dir, nspin)
        
        return CoreInputData(
            source_type="abacus",
            root_dir=root_dir,
            out_dir=out_dir,
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
            pp_dir=pp_dir,
            orb_dir=orb_dir,
            metadata={"e_tot": e_tot, "latname": latname},
        )
