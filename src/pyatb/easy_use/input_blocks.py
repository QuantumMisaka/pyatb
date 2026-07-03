from __future__ import annotations

import os
import warnings

from pyatb.easy_use.input_models import CoreInputData, GenerationRequest
from pyatb.easy_use.input_utils import (
    first_occurrence_species,
    get_k_mesh_from_dim,
    kpath_generator,
    parse_upf_valence,
)


def _base_input_text(core: CoreInputData, req: GenerationRequest) -> str:
    lines = [
        "INPUT_PARAMETERS",
        "{",
        f"    nspin                           {core.nspin}",
        f"    package                         {core.package_name}",
        f"    fermi_energy                    {core.e_fermi}",
        "    fermi_energy_unit               eV",
        f"    HR_route                        {core.hr_route}",
        f"    SR_route                        {core.sr_route}",
        f"    rR_route                        {core.rr_route}",
        f"    HR_unit                         {core.hr_unit}",
        f"    rR_unit                         {core.rr_unit}",
        f"    max_kpoint_num                  {req.max_kpoint_num}",
        "}",
        "",
        "LATTICE",
        "{",
        f"    lattice_constant                {_format_float13(core.lattice_constant)}",
        "    lattice_constant_unit           Angstrom",
        "    lattice_vector",
        f"    {_format_float_vector13(core.lattice_vectors[0])}",
        f"    {_format_float_vector13(core.lattice_vectors[1])}",
        f"    {_format_float_vector13(core.lattice_vectors[2])}",
        "}",
    ]
    return "\n".join(lines) + "\n"


def build_base_input(core: CoreInputData, req: GenerationRequest) -> str:
    return _base_input_text(core, req)


def _append_block(input_text: str, block_name: str, lines: list[str]) -> str:
    block = [f"\n{block_name}", "{"] + [f"    {line}" for line in lines] + ["}"]
    return input_text + "\n".join(block) + "\n"


def _format_vector(values) -> str:
    return " ".join(map(str, values))


def _format_float13(value) -> str:
    return f"{value:.13f}"


def _format_float_vector13(values) -> str:
    return "".join(f"{value:19.13f}" for value in values)


def _mp_grid(core: CoreInputData, req: GenerationRequest, density: float):
    integrate_grid, adaptive_grid, pbc = get_k_mesh_from_dim(core.lattice_vectors, req.dim, density)
    return integrate_grid, adaptive_grid, pbc


def _default_energy_range(core: CoreInputData, req: GenerationRequest, fallback=(-1.0, 1.0)) -> list[float]:
    if req.energy_range:
        return req.energy_range
    return [fallback[0], fallback[1]]


def _default_omega_range(req: GenerationRequest, fallback=(0.0, 5.0)) -> list[float]:
    if req.omega_range:
        return req.omega_range
    return [fallback[0], fallback[1]]


def _default_band_range(core: CoreInputData, req: GenerationRequest) -> list[int]:
    if req.band_range:
        return req.band_range
    if core.n_occu is not None and core.n_bands is not None:
        return [max(1, core.n_occu - 100), min(core.n_bands, core.n_occu + 100)]
    return [-1, -1]


def _occ_band(core: CoreInputData, req: GenerationRequest, allow_unset: bool = False) -> int | None:
    if req.occu_switch > 0:
        return req.occu_switch
    if allow_unset and core.n_occu is None:
        return None
    return core.n_occu


def _stru_file(core: CoreInputData) -> str:
    return "STRU"


def _append_multiline_block(input_text: str, block_name: str, lines: list[str], extra_lines: list[str] | None = None) -> str:
    block = [f"\n{block_name}", "{"] + [f"    {line}" for line in lines]
    if extra_lines:
        block.extend(extra_lines)
    block.append("}")
    return input_text + "\n".join(block) + "\n"


def _line_mode_lines(core: CoreInputData, req: GenerationRequest) -> tuple[list[str], list[str]]:
    band_line, _, _ = kpath_generator(
        core.ase_stru,
        req.kline_density,
        req.dim,
        tolerance=req.tolerance,
        knum=req.knum,
        kpath=req.kpath,
    )
    num_lines = len(band_line) - 1
    lines = [
        f"kpoint_mode               {req.kmode}",
        f"kpoint_num                {num_lines}",
        "high_symmetry_kpoint",
    ]
    extra_lines = [f"   {line}" for line in band_line]
    return lines, extra_lines


def _kpoint_lines(core: CoreInputData, req: GenerationRequest, mp_density: float) -> tuple[list[str], list[str] | None]:
    if req.kmode in ["mp", "mesh"]:
        integrate_grid, _, _ = _mp_grid(core, req, mp_density)
        return [
            "kpoint_mode               mp",
            f"mp_grid                   {_format_vector(integrate_grid)}",
        ], None
    return _line_mode_lines(core, req)


def _grid_lines(label: str, core: CoreInputData, req: GenerationRequest, density: float) -> list[str]:
    integrate_grid, _, _ = _mp_grid(core, req, density)
    return [f"{label}                        {_format_vector(integrate_grid)}"]


def _integrate_and_adaptive_lines(core: CoreInputData, req: GenerationRequest, density: float, threshold: int | float) -> list[str]:
    integrate_grid, adaptive_grid, _ = _mp_grid(core, req, density)
    return [
        "integrate_mode            Grid",
        f"integrate_grid            {_format_vector(integrate_grid)}",
        f"adaptive_grid             {_format_vector(adaptive_grid)}",
        f"adaptive_grid_threshold   {threshold}",
    ]


def _parse_valence_line(core: CoreInputData, req: GenerationRequest) -> tuple[int, str]:
    placeholder = "#  number of valence electrons for the elements; it should match the Pseudopotentials, for example: 12 6"
    if isinstance(req.valence, str) and req.valence.strip().lower() != "auto":
        tokens = req.valence.strip().split()
        if tokens:
            try:
                valences = [int(round(float(token))) for token in tokens]
                return len(valences), " ".join(map(str, valences))
            except ValueError:
                warnings.warn(f"valence string parse failed: {req.valence}; using placeholder.")
        species_order = first_occurrence_species(core.ase_stru)
        return len(species_order), placeholder

    species_order = first_occurrence_species(core.ase_stru)
    pp_map = getattr(core.ase_stru, "info", {}).get("pp", None)
    if not isinstance(pp_map, dict) or not pp_map:
        warnings.warn("ase_stru.info['pp'] is missing; using placeholder valence line.")
        return len(species_order), placeholder

    base_dir = os.path.dirname(os.path.abspath(str(core.stru_file))) if core.stru_file else os.getcwd()
    valences = []
    try:
        for elem in species_order:
            pp_name_or_path = pp_map[elem]
            upf_path = pp_name_or_path if os.path.isabs(pp_name_or_path) else os.path.normpath(os.path.join(base_dir, pp_name_or_path))
            valences.append(int(round(parse_upf_valence(upf_path))))
    except Exception as exc:
        warnings.warn(f"failed to parse valence from pseudopotentials: {exc}; using placeholder.")
        return len(species_order), placeholder
    return len(species_order), " ".join(map(str, valences))


def build_band_block(input_text: str, core: CoreInputData, req: GenerationRequest) -> str:
    lines = ["wf_collect                0"]
    k_lines, extra_lines = _kpoint_lines(core, req, 0.03)
    lines.extend(k_lines)
    return _append_multiline_block(input_text, "BAND_STRUCTURE", lines, extra_lines)


def build_bandunfolding_block(input_text: str, core: CoreInputData, req: GenerationRequest) -> str:
    band_range = _default_band_range(core, req)
    lines = [
        f"band_range                {band_range[0]}  {band_range[1]}",
        "ecut                      10",
        f"stru_file                 {_stru_file(core)}",
        f"m_matrix                  {req.m_matrix}",
    ]
    k_lines, extra_lines = _kpoint_lines(core, req, 0.012)
    lines.extend(k_lines)
    return _append_multiline_block(input_text, "BANDUNFOLDING", lines, extra_lines)


def build_bandunfolding_spin_texture_block(input_text: str, core: CoreInputData, req: GenerationRequest) -> str:
    band_range = _default_band_range(core, req)
    lines = [
        f"stru_file                 {_stru_file(core)}",
        "ecut                      10",
        f"band_range                {band_range[0]}  {band_range[1]}",
        f"m_matrix                  {req.m_matrix}",
    ]
    k_lines, extra_lines = _kpoint_lines(core, req, 0.012)
    lines.extend(k_lines)
    return _append_multiline_block(
        input_text,
        "BANDUNFOLDING_SPIN_TEXTURE",
        lines,
        extra_lines,
    )


def build_fatband_block(input_text: str, core: CoreInputData, req: GenerationRequest) -> str:
    band_range = _default_band_range(core, req)
    lines = [
        f"band_range                {band_range[0]}  {band_range[1]}",
        f"stru_file                 {_stru_file(core)}",
    ]
    k_lines, extra_lines = _kpoint_lines(core, req, 0.015)
    lines.extend(k_lines)
    return _append_multiline_block(input_text, "FAT_BAND", lines, extra_lines)


def build_cohp_block(input_text: str, core: CoreInputData, req: GenerationRequest) -> str:
    erange = _default_energy_range(core, req, fallback=(-10.0, 10.0))
    e_low, e_high = core.e_fermi + erange[0], core.e_fermi + erange[1]
    integrate_grid, _, _ = _mp_grid(core, req, req.mp_density)
    lines = [
        f"stru_file                 {_stru_file(core)}",
        "atom_i_index              -1",
        "atom_j_index              -1",
        "atom_i_orbs               all",
        "atom_j_orbs               all",
        "method                    COHP",
        "spin                      sum",
        f"e_range                   {e_low} {e_high}",
        "de                        0.05",
        "sigma                     0.15",
        "invert                    1",
        "shift_to_efermi           1",
        "output_prefix             COHP",
        "kpoint_mode               mp",
        f"mp_grid                   {_format_vector(integrate_grid)}",
    ]
    return _append_block(input_text, "COHP", lines)


def build_fermi_energy_block(input_text: str, core: CoreInputData, req: GenerationRequest) -> str:
    integrate_grid, _, _ = _mp_grid(core, req, req.mp_density)
    lines = [
        "temperature                 0.0",
        f"electron_num                {core.n_elec}",
        f"grid                        {_format_vector(integrate_grid)}",
        "epsilon                     0.001",
    ]
    return _append_block(input_text, "FERMI_ENERGY", lines)


def build_fermi_surface_block(input_text: str, core: CoreInputData, req: GenerationRequest) -> str:
    integrate_grid, _, _ = _mp_grid(core, req, 0.03)
    band_range = _default_band_range(core, req)
    lines = [
        "bar                         0.001",
        f"nbands                      {band_range[0]} {band_range[1]}",
        "kpoint_mode                 mp",
        f"mp_grid                     {_format_vector(integrate_grid)}",
    ]
    return _append_block(input_text, "FERMI_SURFACE", lines)


def build_findnodes_block(input_text: str, core: CoreInputData, req: GenerationRequest) -> str:
    e_low, e_high = core.e_fermi + _default_energy_range(core, req, fallback=(0.0, 0.0))[0], core.e_fermi + _default_energy_range(core, req, fallback=(0.0, 0.0))[1]
    integrate_grid, adaptive_grid, _ = _mp_grid(core, req, 0.06)
    lines = [
        f"energy_range              {e_low} {e_high}",
        "k_start                   0.0  0.0 -0.2",
        "k_vect1                   0.0  0.0  0.0",
        "k_vect2                   0.0  0.0  0.0",
        "k_vect3                   0.0  0.0  0.4",
        f"initial_grid              {_format_vector(integrate_grid)}",
        "initial_threshold         0.01",
        f"adaptive_grid             {_format_vector(adaptive_grid)}",
        "adaptive_threshold        0.001",
    ]
    return _append_block(input_text, "FIND_NODES", lines)


def build_pdos_block(input_text: str, core: CoreInputData, req: GenerationRequest) -> str:
    erange = _default_energy_range(core, req, fallback=(-8.0, 8.0))
    e_low, e_high = core.e_fermi + erange[0], core.e_fermi + erange[1]
    integrate_grid, _, _ = _mp_grid(core, req, 0.10)
    lines = [
        f"stru_file                 {_stru_file(core)}",
        f"e_range                   {e_low} {e_high}",
        "de                        0.01",
        "sigma                     0.10",
        "kpoint_mode               mp",
        f"mp_grid                   {_format_vector(integrate_grid)}",
    ]
    return _append_block(input_text, "PDOS", lines)


def build_spintexture_block(input_text: str, core: CoreInputData, req: GenerationRequest) -> str:
    band_range = _default_band_range(core, req)
    lines = [
        f"band_range                {band_range[0]}  {band_range[1]}",
    ]
    k_lines, extra_lines = _kpoint_lines(core, req, 0.012)
    lines.extend(k_lines)
    return _append_multiline_block(input_text, "SPIN_TEXTURE", lines, extra_lines)


def build_surface_state_block(input_text: str, core: CoreInputData, req: GenerationRequest) -> str:
    lines = [
        "cal_surface_method          green_fun",
        "surface_direction           c",
        "energy_windows              -1.0 1.0",
        "de                          0.01",
        "eta                         0.01",
        "coupling_layers             6",
        "calculate_layer             1",
        f"kpoint_mode                 {req.kmode}",
    ]
    return _append_block(input_text, "SURFACE_STATE", lines)


def build_ahc_block(input_text: str, core: CoreInputData, req: GenerationRequest) -> str:
    lines = _integrate_and_adaptive_lines(core, req, req.mp_density, 1000)
    return _append_block(input_text, "AHC", lines)


def build_anc_block(input_text: str, core: CoreInputData, req: GenerationRequest) -> str:
    erange = _default_energy_range(core, req)
    lines = [
        f"method                    {req.method}",
        f"fermi_range               {erange[0]} {erange[1]}",
        "de                        0.01",
        "eta                       0.10",
    ] + _grid_lines("integrate_grid", core, req, req.mp_density)
    return _append_block(input_text, "ANC", lines)


def build_berry_block(input_text: str, core: CoreInputData, req: GenerationRequest) -> str:
    occ_band = _occ_band(core, req)
    lines = [f"method                    {req.method}"]
    if occ_band is not None:
        lines.append(f"occ_band                  {occ_band}")
    if req.kmode in ["mp", "mesh"]:
        integrate_grid, adaptive_grid, _ = _mp_grid(core, req, 0.02)
        lines.extend([
            "kpoint_mode               mp",
            f"mp_grid                   {_format_vector(integrate_grid)}",
            f"adaptive_grid             {_format_vector(adaptive_grid)}",
            "adaptive_grid_threshold   1000",
        ])
        return _append_block(input_text, "BERRY_CURVATURE", lines)
    k_lines, extra_lines = _line_mode_lines(core, req)
    lines.extend(k_lines)
    return _append_multiline_block(input_text, "BERRY_CURVATURE", lines, extra_lines)


def build_chern_block(input_text: str, core: CoreInputData, req: GenerationRequest) -> str:
    occ_band = _occ_band(core, req)
    lines = [f"method                    {req.method}"]
    if occ_band is not None:
        lines.append(f"occ_band                  {occ_band}")
    lines.extend([
        "integrate_mode            Grid",
        f"integrate_grid            {_format_vector(_mp_grid(core, req, req.mp_density)[0])}",
        "k_start                   0 0 0",
        "k_vect1                   1 0 0",
        "k_vect2                   0 1 0",
    ])
    return _append_block(input_text, "CHERN_NUMBER", lines)


def build_chirality_block(input_text: str, core: CoreInputData, req: GenerationRequest) -> str:
    occ_band = _occ_band(core, req)
    lines = [
        f"method                      {req.method}",
        f"occ_band                    {occ_band if occ_band is not None else -1}",
        "k_vect                      0.0 0.0 0.0",
        "radius                      0.01",
        "point_num                   1000",
    ]
    return _append_block(input_text, "CHIRALITY", lines)


def build_orbital_magnetization_block(input_text: str, core: CoreInputData, req: GenerationRequest) -> str:
    integrate_grid, _, _ = _mp_grid(core, req, req.mp_density)
    erange = _default_energy_range(core, req, fallback=(-2.0, 2.0))
    lines = [
        f"fermi_energy                {core.e_fermi}",
        f"fermi_range                 {erange[0]} {erange[1]}",
        "de                          0.05",
        "eta                         0.01",
        f"grid                        {_format_vector(integrate_grid)}",
    ]
    return _append_block(input_text, "ORBITAL_MAGNETIZATION", lines)


def build_polarization_block(input_text: str, core: CoreInputData, req: GenerationRequest) -> str:
    integrate_grid, _, _ = _mp_grid(core, req, req.mp_density)
    atom_type, valence_line = _parse_valence_line(core, req)
    lines = [
        f"occ_band       {core.n_occu}",
        f"nk1            {int(integrate_grid[0])}",
        f"nk2            {int(integrate_grid[1])}",
        f"nk3            {int(integrate_grid[2])}",
        f"atom_type      {atom_type}",
        f"stru_file      {_stru_file(core)}",
        f"valence_e      {valence_line}",
    ]
    return _append_block(input_text, "POLARIZATION", lines)


def build_shc_block(input_text: str, core: CoreInputData, req: GenerationRequest) -> str:
    integrate_grid, _, _ = _mp_grid(core, req, req.mp_density)
    erange = _default_energy_range(core, req)
    lines = [
        "alpha                       x",
        "beta                        y",
        "gamma                       z",
        f"fermi_range                 {erange[0]} {erange[1]}",
        "de                          0.01",
        "eta                         0.01",
        f"integrate_grid              {_format_vector(integrate_grid)}",
    ]
    return _append_block(input_text, "SHC", lines)


def build_wilson_block(input_text: str, core: CoreInputData, req: GenerationRequest) -> str:
    occ_band = _occ_band(core, req)
    lines = []
    if occ_band is not None:
        lines.append(f"occ_band       {occ_band}")
    lines.extend([
        "k_start        0.0  0.0  0.5",
        "k_vect1        1.0  0.0  0.0",
        "k_vect2        0.0  0.5  0.0",
        "nk1            101",
        "nk2            101",
    ])
    return _append_block(input_text, "WILSON_LOOP", lines)


def build_bcd_block(input_text: str, core: CoreInputData, req: GenerationRequest) -> str:
    erange = _default_energy_range(core, req, fallback=(-1.0, 1.0))
    e_low, e_high = core.e_fermi + erange[0], core.e_fermi + erange[1]
    integrate_grid, _, _ = _mp_grid(core, req, 0.012)
    lines = [
        f"omega                      {e_low} {e_high}",
        "domega                     0.001",
        f"grid                       {_format_vector(integrate_grid)}",
    ]
    return _append_block(input_text, "BERRY_CURVATURE_DIPOLE", lines)


def build_cpge_block(input_text: str, core: CoreInputData, req: GenerationRequest) -> str:
    erange = _default_energy_range(core, req, fallback=(-1.0, 1.0))
    e_low, e_high = core.e_fermi + erange[0], core.e_fermi + erange[1]
    lines = [
        f"omega                      {e_low} {e_high}",
        "domega                     0.001",
    ] + _integrate_and_adaptive_lines(core, req, 0.01, 2000)
    return _append_block(input_text, "CPGE", lines)


def build_drude_weight_block(input_text: str, core: CoreInputData, req: GenerationRequest) -> str:
    omega_range = _default_omega_range(req, fallback=(0.0, 5.0))
    lines = [
        f"omega                       {omega_range[0]} {omega_range[1]}",
        "domega                      0.01",
        "integrate_mode              Grid",
    ]
    return _append_block(input_text, "DRUDE_WEIGHT", lines)


def build_jdos_block(input_text: str, core: CoreInputData, req: GenerationRequest) -> str:
    omega_range = _default_omega_range(req, fallback=(0.0, 5.0))
    integrate_grid, _, _ = _mp_grid(core, req, req.mp_density)
    lines = [
        f"occ_band                   {core.n_occu}",
        f"omega                      {omega_range[0]} {omega_range[1]}",
        "domega                     0.01",
        "eta                        0.10",
        f"grid                       {_format_vector(integrate_grid)}",
    ]
    return _append_block(input_text, "JDOS", lines)


def build_optical_block(input_text: str, core: CoreInputData, req: GenerationRequest) -> str:
    omega_range = _default_omega_range(req, fallback=(0.0, 10.0))
    integrate_grid, _, _ = _mp_grid(core, req, 0.05)
    lines = [
        f"occ_band                   {core.n_occu}",
        f"omega                      {omega_range[0]} {omega_range[1]}",
        "domega                     0.01",
        "eta                        0.10",
        f"grid                       {_format_vector(integrate_grid)}",
    ]
    return _append_block(input_text, "OPTICAL_CONDUCTIVITY", lines)


def build_pockels_block(input_text: str, core: CoreInputData, req: GenerationRequest) -> str:
    integrate_grid, _, _ = _mp_grid(core, req, 0.05)
    omega_range = _default_omega_range(req, fallback=(0.0, 5.0))
    lines = [
        "omega1                      0",
        f"omega                       {omega_range[0]} {omega_range[1]}",
        "domega                      0.01",
        f"grid                        {_format_vector(integrate_grid)}",
    ]
    return _append_block(input_text, "POCKELS", lines)


def build_shg_block(input_text: str, core: CoreInputData, req: GenerationRequest) -> str:
    omega_range = _default_omega_range(req, fallback=(0.0, 4.0))
    integrate_grid, _, _ = _mp_grid(core, req, 0.05)
    lines = [
        f"omega                      {omega_range[0]} {omega_range[1]}",
        "domega                     0.01",
        "eta                        0.05",
        f"grid                       {_format_vector(integrate_grid)}",
    ]
    return _append_block(input_text, "SHG", lines)


def build_shift_block(input_text: str, core: CoreInputData, req: GenerationRequest) -> str:
    omega_range = _default_omega_range(req, fallback=(0.0, 5.0))
    integrate_grid, _, _ = _mp_grid(core, req, req.mp_density)
    lines = [
        f"occ_band                   {core.n_occu}",
        f"omega                      {omega_range[0]} {omega_range[1]}",
        "domega                     0.01",
        "eta                        0.10",
        f"grid                       {_format_vector(integrate_grid)}",
    ]
    return _append_block(input_text, "SHIFT_CURRENT", lines)


def build_boltz_transport_block(input_text: str, core: CoreInputData, req: GenerationRequest) -> str:
    integrate_grid, _, _ = _mp_grid(core, req, req.mp_density)
    lines = [
        "transport_coeff_cal         1",
        "effective_mass_cal          0",
        "transport_method            CRTA",
        f"electron_num                {core.n_elec}",
        f"grid                        {_format_vector(integrate_grid)}",
        "delta_mu_range              -5.0 5.0",
        "mu_step                     0.1",
        "temperature_range           300 300",
        "temperature_step            50",
        "eta                         0.1",
        "relax_time                  10",
        "def_pot                     2",
        "young_mod                   240",
    ]
    return _append_block(input_text, "BOLTZ_TRANSPORT", lines)


def build_reduce_basis_block(input_text: str, core: CoreInputData, req: GenerationRequest) -> str:
    erange = _default_energy_range(core, req, fallback=(-2.0, 2.0))
    band_range = _default_band_range(core, req)
    lines = [
        f"e_range                     {erange[0]} {erange[1]}",
        "threshold                   0.02",
        f"band_index_range            {band_range[0]} {band_range[1]}",
        f"kpoint_mode                 {req.kmode}",
    ]
    return _append_block(input_text, "REDUCE_BASIS", lines)


BAND_MODULE_BUILDERS = {
    "band": build_band_block,
    "band_structure": build_band_block,
    "bandunfolding": build_bandunfolding_block,
    "bandunfolding_spin_texture": build_bandunfolding_spin_texture_block,
    "cohp": build_cohp_block,
    "fatband": build_fatband_block,
    "fat_band": build_fatband_block,
    "fermi_energy": build_fermi_energy_block,
    "fermi_surface": build_fermi_surface_block,
    "findnodes": build_findnodes_block,
    "find_nodes": build_findnodes_block,
    "pdos": build_pdos_block,
    "spintexture": build_spintexture_block,
    "spin_texture": build_spintexture_block,
    "surface_state": build_surface_state_block,
}


GEOMETRY_MODULE_BUILDERS = {
    "ahc": build_ahc_block,
    "anc": build_anc_block,
    "berry": build_berry_block,
    "berry_curvature": build_berry_block,
    "chern": build_chern_block,
    "chern_number": build_chern_block,
    "chirality": build_chirality_block,
    "orbital_magnetization": build_orbital_magnetization_block,
    "polar": build_polarization_block,
    "polarization": build_polarization_block,
    "shc": build_shc_block,
    "wilson": build_wilson_block,
    "wilson_loop": build_wilson_block,
}


OPTICAL_MODULE_BUILDERS = {
    "bcd": build_bcd_block,
    "berry_curvature_dipole": build_bcd_block,
    "cpge": build_cpge_block,
    "drude_weight": build_drude_weight_block,
    "jdos": build_jdos_block,
    "optical": build_optical_block,
    "optical_conductivity": build_optical_block,
    "pockels": build_pockels_block,
    "shg": build_shg_block,
    "shift": build_shift_block,
    "shift_current": build_shift_block,
}


TRANSPORT_MODULE_BUILDERS = {
    "boltz_transport": build_boltz_transport_block,
}


UTILITY_MODULE_BUILDERS = {
    "reduce_basis": build_reduce_basis_block,
}


MODULE_BUILDERS = {}
MODULE_BUILDERS.update(BAND_MODULE_BUILDERS)
MODULE_BUILDERS.update(GEOMETRY_MODULE_BUILDERS)
MODULE_BUILDERS.update(OPTICAL_MODULE_BUILDERS)
MODULE_BUILDERS.update(TRANSPORT_MODULE_BUILDERS)
MODULE_BUILDERS.update(UTILITY_MODULE_BUILDERS)
