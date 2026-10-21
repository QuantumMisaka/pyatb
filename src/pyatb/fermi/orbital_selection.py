from dataclasses import dataclass
import os
import re


L_MULTIPLICITY = {
    "s": 1,
    "p": 3,
    "d": 5,
    "f": 7,
    "g": 9,
}
SHELL_ORDER = ["s", "p", "d", "f", "g"]
STRU_SECTION_HEADERS = {
    "ABFS_ORBITAL",
    "ATOMIC_POSITIONS",
    "ATOMIC_SPECIES",
    "LATTICE_CONSTANT",
    "LATTICE_CONSTANT_UNIT",
    "LATTICE_VECTORS",
    "NUMERICAL_DESCRIPTOR",
    "NUMERICAL_ORBITAL",
    "PAW_FILES",
}


@dataclass(frozen=True)
class OrbitalAtom:
    index: int
    element: str
    orbital_file: str
    shell_counts: dict
    start: int
    stop: int


@dataclass(frozen=True)
class OrbitalMap:
    atoms: tuple
    total_orbitals: int


@dataclass(frozen=True)
class OrbitalSelection:
    atom_i_orbitals: list
    atom_j_orbitals: list
    orbital_map: OrbitalMap


@dataclass(frozen=True)
class ResolvedOrbitals:
    indices: list


def _strip_comment(line):
    return re.sub(r"#.*$", "", line).strip()


def _section_lines(lines, section_name):
    for index, line in enumerate(lines):
        if _strip_comment(line).upper() == section_name:
            section = []
            for next_line in lines[index + 1:]:
                clean = _strip_comment(next_line)
                if clean and clean.upper() in STRU_SECTION_HEADERS:
                    break
                if clean:
                    section.append(clean)
            return section
    return []


def parse_input_orbital_dir(input_file):
    if not input_file:
        return ""
    input_dir = os.path.dirname(os.path.abspath(input_file))
    with open(input_file) as file_obj:
        tokens = []
        for line in file_obj:
            tokens.extend(_strip_comment(line).split())

    for index, token in enumerate(tokens):
        if token == "orbital_dir" and index + 1 < len(tokens):
            orbital_dir = tokens[index + 1]
            if os.path.isabs(orbital_dir):
                return orbital_dir
            return os.path.abspath(os.path.join(input_dir, orbital_dir))
    return input_dir


def parse_stru_metadata(stru_file):
    with open(stru_file) as file_obj:
        lines = file_obj.readlines()

    species_lines = _section_lines(lines, "ATOMIC_SPECIES")
    numerical_lines = _section_lines(lines, "NUMERICAL_ORBITAL")
    position_lines = _section_lines(lines, "ATOMIC_POSITIONS")

    elements = []
    for line in species_lines:
        tokens = line.split()
        if len(tokens) < 3:
            continue
        elements.append(tokens[0])

    orbital_files = {}
    for element, line in zip(elements, numerical_lines):
        orbital_files[element] = line.split()[0]

    atoms = []
    index = 1
    current_element = None
    expected_count = None
    skipped_header = False
    for line in position_lines[1:]:
        tokens = line.split()
        if not tokens:
            continue
        if tokens[0] in elements:
            current_element = tokens[0]
            expected_count = None
            skipped_header = False
            continue
        if current_element is None:
            continue
        if not skipped_header:
            skipped_header = True
            continue
        if expected_count is None:
            expected_count = int(float(tokens[0]))
            continue
        if expected_count <= 0:
            continue
        atoms.append({"index": index, "element": current_element})
        index += 1
        expected_count -= 1

    return atoms, orbital_files


def _shell_counts_from_orbital_file(orbital_path):
    counts = {}
    if not orbital_path or not os.path.exists(orbital_path):
        return counts
    with open(orbital_path, errors="ignore") as file_obj:
        for line in file_obj:
            match = re.match(r"\s*Number of (\w)-orbital\s*-->\s*(\d+)", line)
            if match:
                shell = match.group(1).lower()
                if shell in L_MULTIPLICITY:
                    counts[shell] = int(match.group(2))
    return counts


def _shell_counts_from_filename(filename):
    basename = os.path.basename(filename)
    counts = {}
    for count, shell in re.findall(r"(\d+)([spdfg])", basename):
        counts[shell] = int(count)
    return counts


def parse_orbital_shells(orbital_path):
    counts = _shell_counts_from_orbital_file(orbital_path)
    if counts:
        return counts
    return _shell_counts_from_filename(orbital_path)


def _resolve_orbital_path(stru_file, orbital_dir, orbital_file):
    candidates = []
    if os.path.isabs(orbital_file):
        candidates.append(orbital_file)
    if orbital_dir:
        candidates.append(os.path.join(orbital_dir, orbital_file))
    candidates.append(os.path.join(os.path.dirname(os.path.abspath(stru_file)), orbital_file))
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def _shell_ranges(shell_counts, start):
    ranges = {}
    cursor = start
    for shell in SHELL_ORDER:
        count = shell_counts.get(shell, 0)
        shell_size = count * L_MULTIPLICITY[shell]
        if shell_size:
            ranges[shell] = list(range(cursor, cursor + shell_size))
        cursor += shell_size
    return ranges, cursor


def build_orbital_map(stru_file, input_file=None, orbital_dir=None):
    if not orbital_dir:
        orbital_dir = parse_input_orbital_dir(input_file) if input_file else None

    atom_meta, orbital_files = parse_stru_metadata(stru_file)
    atoms = []
    cursor = 0
    for atom in atom_meta:
        orbital_file = orbital_files.get(atom["element"])
        if orbital_file is None:
            raise ValueError("missing numerical orbital file for element %s" % atom["element"])
        orbital_path = _resolve_orbital_path(stru_file, orbital_dir, orbital_file)
        shell_counts = parse_orbital_shells(orbital_path)
        if not shell_counts:
            raise ValueError("cannot determine orbital shells from %s" % orbital_path)
        _, stop = _shell_ranges(shell_counts, cursor)
        atoms.append(
            OrbitalAtom(
                index=atom["index"],
                element=atom["element"],
                orbital_file=orbital_path,
                shell_counts=shell_counts,
                start=cursor,
                stop=stop,
            )
        )
        cursor = stop
    return OrbitalMap(atoms=tuple(atoms), total_orbitals=cursor)


def parse_global_orbital_indices(selector):
    if isinstance(selector, (list, tuple)):
        return [int(value) for value in selector]
    values = []
    for token in re.split(r"[\s,]+", str(selector).strip()):
        if token:
            values.append(int(token))
    return values


def _selector_tokens(selector):
    if selector is None:
        return ["all"]
    if isinstance(selector, (list, tuple)):
        raw_tokens = selector
    else:
        raw_tokens = re.split(r"[\s,]+", str(selector).strip())
    return [token.lower() for token in raw_tokens if token]


def _selector_shell(token):
    if token in L_MULTIPLICITY:
        return token
    match = re.match(r"^\d+([spdfg])$", token)
    if match:
        return match.group(1)
    return None


def resolve_atom_orbitals(orbital_map, atom_index, selector="all"):
    atom = None
    for candidate in orbital_map.atoms:
        if candidate.index == int(atom_index):
            atom = candidate
            break
    if atom is None:
        raise ValueError("atom index %s is not present in STRU" % atom_index)

    if str(selector).strip().lower() in ("all", "*"):
        return ResolvedOrbitals(list(range(atom.start, atom.stop)))

    shell_ranges, _ = _shell_ranges(atom.shell_counts, atom.start)
    orbitals = []
    for token in _selector_tokens(selector):
        if token in ("all", "*"):
            orbitals.extend(range(atom.start, atom.stop))
            continue
        shell = _selector_shell(token)
        if shell is not None:
            if shell not in shell_ranges:
                raise ValueError("atom %s has no %s orbitals" % (atom_index, token))
            orbitals.extend(shell_ranges[shell])
            continue
        orbitals.append(int(token))

    result = []
    seen = set()
    for orbital in orbitals:
        if orbital in seen:
            continue
        if orbital < 0 or orbital >= orbital_map.total_orbitals:
            raise ValueError("orbital index %s is out of range" % orbital)
        seen.add(orbital)
        result.append(orbital)
    return ResolvedOrbitals(result)


def resolve_cohp_orbitals(stru_file, atom_i_index, atom_j_index, atom_i_orbs="all",
                          atom_j_orbs="all", input_file=None, orbital_dir=None):
    orbital_map = build_orbital_map(stru_file, input_file=input_file, orbital_dir=orbital_dir)
    return OrbitalSelection(
        atom_i_orbitals=resolve_atom_orbitals(orbital_map, atom_i_index, atom_i_orbs).indices,
        atom_j_orbitals=resolve_atom_orbitals(orbital_map, atom_j_index, atom_j_orbs).indices,
        orbital_map=orbital_map,
    )
