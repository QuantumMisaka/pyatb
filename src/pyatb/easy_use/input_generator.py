from __future__ import annotations

import shutil
from pathlib import Path

from pyatb.easy_use.input_blocks import MODULE_BUILDERS, build_base_input
from pyatb.easy_use.input_parsers import build_generation_request, build_parser, normalize_source_type
from pyatb.easy_use.input_utils import resolve_output_directory
from pyatb.easy_use.sources.abacus import rewrite_abacus_stru_paths
from pyatb.easy_use.sources import load_core_data


def prepare_output_directory(output_dir: Path, core) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if core.stru_file is None:
        return
    stru_target = output_dir / "STRU"
    if core.stru_file.resolve() == stru_target.resolve():
        return
    if core.pp_dir is None and core.orb_dir is None:
        shutil.copy2(core.stru_file, stru_target)
        return
    rewrite_abacus_stru_paths(str(core.pp_dir), str(core.orb_dir), f_stru=str(core.stru_file), ase_stru=core.ase_stru, f_out=str(stru_target))


def write_energy_summary(output_dir: Path, core) -> None:
    if core.source_type == "hamgnn":
        summary_file = output_dir / "get_Energy_HamGNN.out"
        lines = [
            "E_TOTAL (eV)   =  None_yet",
            f"E_FERMI (eV)   =  {core.e_fermi}",
            f"BAND_GAP       =  {core.metadata.get('band_gap')}",
            "NBANDS         =  None_yet ",
            f"NELEC          =  {core.n_elec}",
            f"Occupied bands =  {core.n_occu} # estimated by NELEC/2 ",
        ]
    else:
        summary_file = output_dir / "get_Energy.out"
        lines = [
            f"E_TOTAL (eV)   =  {core.metadata.get('e_tot')}",
            f"E_FERMI (eV)   =  {core.e_fermi}",
            f"Occupied bands =  {core.n_occu}",
            f"NBANDS         =  {core.n_bands}",
            f"NELEC          =  {core.n_elec}",
        ]
    summary_file.write_text("\n".join(lines) + "\n")


def generate_input_text(core, req) -> str:
    input_text = build_base_input(core, req)
    for module in req.modules:
        builder = MODULE_BUILDERS[module]
        input_text = builder(input_text, core, req)
    return input_text


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    source_type = normalize_source_type(args)
    core = load_core_data(args.input, source_type=source_type)
    req = build_generation_request(args, core)
    output_dir = resolve_output_directory(core.root_dir, args.output)

    prepare_output_directory(output_dir, core)
    write_energy_summary(output_dir, core)

    input_text = generate_input_text(core, req)
    (output_dir / "Input").write_text(input_text)

    print(f"[PYATB with {core.source_type.upper()}] Input and STRU files for PyATB is generated in: {output_dir}")


if __name__ == "__main__":
    main()
