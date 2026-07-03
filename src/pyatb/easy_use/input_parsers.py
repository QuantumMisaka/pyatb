from __future__ import annotations

import argparse

from pyatb.easy_use.input_blocks import (
    BAND_MODULE_BUILDERS,
    GEOMETRY_MODULE_BUILDERS,
    OPTICAL_MODULE_BUILDERS,
    TRANSPORT_MODULE_BUILDERS,
    UTILITY_MODULE_BUILDERS,
)
from pyatb.easy_use.input_models import CoreInputData, GenerationRequest
from pyatb.easy_use.input_utils import parse_band_range, parse_energy_range, parse_omega_range


MODULE_FLAG_SPECS = {
    # Band modules
    "band": {
        "module": "band",
        "flags": ("--band",),
        "help": "Band Structure calculation",
    },
    "bandunfolding": {
        "module": "bandunfolding",
        "flags": ("--bandunfolding", "--unfolding", "--bandunfold", "--unfold"),
        "help": "Band unfolding",
    },
    "bandunfolding_spin_texture": {
        "module": "bandunfolding_spin_texture",
        "flags": ("--bandunfolding_spin_texture", "--bust", "--unfold_spin"),
        "help": "Band unfolding spin texture",
    },
    "fatband": {
        "module": "fatband",
        "flags": ("--fatband", "--pband", "--projectedband"),
        "help": "Projected band structures",
    },
    "cohp": {
        "module": "cohp",
        "flags": ("--cohp",),
        "help": "COHP/COOP atom-pair population spectrum.",
    },
    "fermi_energy": {
        "module": "fermi_energy",
        "flags": ("--fermi_energy", "--fe"),
        "help": "Solve Fermi energy from electron number on a k-grid.",
    },
    "fermi_surface": {
        "module": "fermi_surface",
        "flags": ("--fs", "--fermisurface"),
        "help": "Fermi surface calculation.",
    },
    "findnodes": {
        "module": "findnodes",
        "flags": ("--findnodes", "--fnodes"),
        "help": "Find Weyl nodes calculation",
    },
    "pdos": {
        "module": "pdos",
        "flags": ("--pdos",),
        "help": "Projected DOS calculation",
    },
    "spintexture": {
        "module": "spintexture",
        "flags": ("--spintexture", "--spin", "--spintex"),
        "help": "Spin texture",
    },
    "surface_state": {
        "module": "surface_state",
        "flags": ("--surface_state", "--ss"),
        "help": "Surface state calculation",
    },
    # Geometry modules
    "ahc": {
        "module": "ahc",
        "flags": ("--ahc",),
        "help": "Anomalous Hall Conductivity",
    },
    "anc": {
        "module": "anc",
        "flags": ("--anc",),
        "help": "Anomalous Nerst Conductivity",
    },
    "berry": {
        "module": "berry",
        "flags": ("--berry", "-b"),
        "help": "Berry Curvature",
    },
    "chern": {
        "module": "chern",
        "flags": ("--chern",),
        "help": "Chern Number",
    },
    "chirality": {
        "module": "chirality",
        "flags": ("--chirality",),
        "help": "Node chirality around a selected k-point sphere.",
    },
    "orbital_magnetization": {
        "module": "orbital_magnetization",
        "flags": ("--orbital_magnetization", "--om"),
        "help": "Orbital magnetization",
    },
    "polar": {
        "module": "polarization",
        "flags": ("--polar", "--polarization"),
        "help": "Polarization",
    },
    "shc": {
        "module": "shc",
        "flags": ("--shc",),
        "help": "Spin Hall conductivity",
    },
    "wilson": {
        "module": "wilson",
        "flags": ("--wilson", "--wl", "--wilson_loop"),
        "help": "Wilson loop",
    },
    # Optical modules
    "bcd": {
        "module": "bcd",
        "flags": ("--bcd", "--berry_curvature_dipole"),
        "help": "Berry curvature dipole",
    },
    "cpge": {
        "module": "cpge",
        "flags": ("--cpge",),
        "help": "CPGE",
    },
    "drude_weight": {
        "module": "drude_weight",
        "flags": ("--drude_weight", "--drude"),
        "help": "Drude weight",
    },
    "jdos": {
        "module": "jdos",
        "flags": ("--jdos",),
        "help": "JDOS",
    },
    "optical": {
        "module": "optical",
        "flags": ("--optical",),
        "help": "Optical conductivity",
    },
    "pockels": {
        "module": "pockels",
        "flags": ("--pockels",),
        "help": "Pockels response",
    },
    "shg": {
        "module": "shg",
        "flags": ("--shg",),
        "help": "Second harmonic generation",
    },
    "shift": {
        "module": "shift",
        "flags": ("--shift", "--shift_current"),
        "help": "Shift current",
    },
    # Transport modules
    "boltz_transport": {
        "module": "boltz_transport",
        "flags": ("--boltz_transport", "--boltz"),
        "help": "Boltzmann transport",
    },
    # Utility modules
    "reduce_basis": {
        "module": "reduce_basis",
        "flags": ("--reduce_basis",),
        "help": "Reduce basis utility",
    },
}


def _ordered_module_flags() -> list[str]:
    ordered = []
    seen = set()
    for builders in (
        BAND_MODULE_BUILDERS,
        GEOMETRY_MODULE_BUILDERS,
        OPTICAL_MODULE_BUILDERS,
        TRANSPORT_MODULE_BUILDERS,
        UTILITY_MODULE_BUILDERS,
    ):
        for module_name in builders:
            for flag_name, spec in MODULE_FLAG_SPECS.items():
                if flag_name in seen:
                    continue
                if module_name == flag_name or module_name == spec["module"]:
                    ordered.append(flag_name)
                    seen.add(flag_name)
    return ordered


MODULE_FLAG_MAP = {
    flag_name: MODULE_FLAG_SPECS[flag_name]["module"]
    for flag_name in _ordered_module_flags()
}


MODULE_GROUPS = (
    ("Band modules", BAND_MODULE_BUILDERS),
    ("Geometry modules", GEOMETRY_MODULE_BUILDERS),
    ("Optical modules", OPTICAL_MODULE_BUILDERS),
    ("Transport modules", TRANSPORT_MODULE_BUILDERS),
    ("Utility modules", UTILITY_MODULE_BUILDERS),
)


def _module_flags_for_builders(builders: dict[str, object]) -> list[str]:
    flags = []
    for flag_name in MODULE_FLAG_MAP:
        module_name = MODULE_FLAG_MAP[flag_name]
        if flag_name in builders or module_name in builders:
            flags.append(flag_name)
    return flags


def _add_module_arguments(parser: argparse.ArgumentParser) -> None:
    for title, builders in MODULE_GROUPS:
        group = parser.add_argument_group(title)
        for flag_name in _module_flags_for_builders(builders):
            spec = MODULE_FLAG_SPECS[flag_name]
            group.add_argument(*spec["flags"], dest=flag_name, action="store_true", help=spec["help"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Input Generator Script.")
    parser.add_argument("-i", "--input", "--in", type=str, default="./", help="Set SCF/result directory.")
    parser.add_argument("-o", "--output", "--out", type=str, default=None, help="Set output directory for PYATB.")
    _add_module_arguments(parser)
    parser.add_argument("--kline", type=float, default=0.01, help="Density of K-path.")
    parser.add_argument("--knum", type=int, default=0, help="Number of kpoints on each line of K-path.")
    parser.add_argument("--kpath", type=str, default=None, help="Manual K-path string.")
    parser.add_argument("--kmode", type=str, default="line", help="Mode of K-path, e.g. mp or line.")
    parser.add_argument("--dim", type=str, default="3", help="Periodicity definition for k-path/k-mesh generation.")
    parser.add_argument("--tolerance", "--tol", type=float, default=1e-3, help="Tolerance for determining Bravais lattice.")
    parser.add_argument("--erange", type=str, default="8", help="Energy range referred to Fermi level.")
    parser.add_argument("--frange", type=float, default=1, help="Reserved Fermi range argument.")
    parser.add_argument("--valence", "--valence_e", type=str, default="auto", help="Valence electron settings for polarization.")
    parser.add_argument("--orange", type=float, nargs="+", default=[0, 10], help="Optical frequency range")
    parser.add_argument("--mp", "--density", type=float, default=0.10, help="MP grid density")
    parser.add_argument("--m_matrix", "--matrix", type=str, default="1 0 0  0 1 0  0 0 1", help="Band unfolding matrix.")
    parser.add_argument("--occu", type=int, default=0, help="Occupied band switch.")
    parser.add_argument("--method", type=int, default=0, help="Method for Berry-like calculations.")
    parser.add_argument("--bandrange", "--brange", type=str, default=None, help="Band range for FAT_BAND or BANDUNFOLDING.")
    parser.add_argument("--max_kpoint_num", "--maxkpt", type=int, default=800, help="Max parallel kpoint number in one iteration.")
    calculator = parser.add_mutually_exclusive_group()
    calculator.add_argument("--abacus", action="store_true", help="Use ABACUS as source reader.")
    calculator.add_argument("--hamgnn", action="store_true", help="Use HamGNN as source reader.")
    return parser


def normalize_source_type(args) -> str:
    if args.hamgnn:
        return "hamgnn"
    return "abacus"


def collect_requested_modules(args) -> list[str]:
    modules = []
    seen = set()
    for flag_name, module_name in MODULE_FLAG_MAP.items():
        if getattr(args, flag_name):
            if module_name not in seen:
                modules.append(module_name)
                seen.add(module_name)
    return modules


def build_generation_request(args, core: CoreInputData) -> GenerationRequest:
    return GenerationRequest(
        modules=collect_requested_modules(args),
        dim=args.dim,
        kmode=args.kmode,
        kline_density=args.kline,
        knum=args.knum,
        kpath=args.kpath,
        tolerance=args.tolerance,
        mp_density=args.mp,
        max_kpoint_num=args.max_kpoint_num,
        energy_range=parse_energy_range(args.erange),
        omega_range=parse_omega_range(args.orange),
        band_range=parse_band_range(args.bandrange, core.n_occu, core.n_bands),
        occu_switch=args.occu,
        method=args.method,
        m_matrix=args.m_matrix,
        valence=args.valence,
    )
