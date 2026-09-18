import numpy as np
import inspect
import types

import pyatb.fermi.cohp as cohp_module
from pyatb.fermi.cohp import COHP, cohp_values_one_k


def test_cohp_values_one_k_matches_standalone_formula():
    matrix = np.array(
        [
            [1.0, 2.0 + 0.5j, 0.0],
            [2.0 - 0.5j, 3.0, 4.0],
            [0.0, 4.0, 5.0],
        ],
        dtype=np.complex128,
    )
    eigenvalues = np.array([-1.0, 0.5], dtype=float)
    eigenvectors = np.array(
        [
            [1.0 + 0.0j, 0.0 + 1.0j],
            [0.5 + 0.5j, 1.0 + 0.0j],
            [0.25 + 0.0j, 0.5 - 0.25j],
        ],
        dtype=np.complex128,
    )

    energies, values = cohp_values_one_k(
        matrix, eigenvalues, eigenvectors, [0, 1], [2], pair_factor=2.0
    )

    expected = []
    for ib in range(eigenvectors.shape[1]):
        total = 0.0
        for iorb in [0, 1]:
            total += (eigenvectors[iorb, ib].conjugate() * matrix[iorb, 2] * eigenvectors[2, ib]).real
        expected.append(2.0 * total)

    np.testing.assert_allclose(energies, eigenvalues)
    np.testing.assert_allclose(values, np.array(expected))


def test_atom_blocks_reconstruct_hamiltonian_expectation_one_k():
    matrix = np.array([[1.0, 0.3 - 0.2j], [0.3 + 0.2j, 2.0]], dtype=np.complex128)
    eigenvectors = np.array([[0.8 + 0.1j], [0.4 - 0.3j]], dtype=np.complex128)
    eigenvectors /= np.linalg.norm(eigenvectors)
    eigenvalues = np.array([0.0])

    onsite_i = cohp_values_one_k(matrix, eigenvalues, eigenvectors, [0], [0])[1][0]
    onsite_j = cohp_values_one_k(matrix, eigenvalues, eigenvectors, [1], [1])[1][0]
    offsite = cohp_values_one_k(
        matrix, eigenvalues, eigenvectors, [0], [1], pair_factor=2.0
    )[1][0]
    expected = (eigenvectors[:, 0].conj() @ matrix @ eigenvectors[:, 0]).real
    np.testing.assert_allclose(onsite_i + onsite_j + offsite, expected)


def test_print_plot_script_only_writes_script(tmp_path):
    cohp = COHP.__new__(COHP)
    cohp.output_path = str(tmp_path)

    cohp.print_plot_script()

    assert (tmp_path / "plot_cohp.py").is_file()
    assert not (tmp_path / "cohp.pdf").exists()
    assert "subprocess.run" not in inspect.getsource(COHP.print_plot_script)


class _SingleKGenerator:
    total_kpoint_num = 1

    def __iter__(self):
        return iter([np.array([[0.0, 0.0, 0.0]], dtype=float)])


class _UnitSolver:
    def diago_H(self, kpoints):
        eigenvectors = np.ones((kpoints.shape[0], 1, 1), dtype=np.complex128)
        eigenvalues = np.zeros((kpoints.shape[0], 1), dtype=float)
        return eigenvectors, eigenvalues

    def get_Hk(self, kpoints):
        return np.ones((kpoints.shape[0], 1, 1), dtype=np.complex128)


def _cohp_with_fake_solver(nspin):
    cohp = COHP.__new__(COHP)
    cohp.nspin = nspin
    cohp._COHP__k_generator = _SingleKGenerator()
    solver = _UnitSolver()
    cohp._COHP__tb_solver = (solver, solver) if nspin == 2 else (solver,)
    return cohp


def _calculate_fake_spectrum(cohp, monkeypatch):
    captured = {}
    selection = types.SimpleNamespace(
        atom_i_orbitals=[0],
        atom_j_orbitals=[0],
        orbital_map=types.SimpleNamespace(total_orbitals=1, atoms=[]),
    )
    monkeypatch.setattr(cohp_module, "resolve_cohp_orbitals", lambda **kwargs: selection)
    monkeypatch.setattr(COHP, "print_data", lambda self, output_prefix, energy, spectrum, *args: captured.update(spectrum=spectrum.copy()))

    cohp.calculate_cohp(
        fermi_energy=0.0,
        stru_file="STRU",
        atom_i_index=1,
        atom_j_index=1,
        method="COHP",
        spin="sum",
        e_range=[-1.0, 1.0],
        de=0.5,
        sigma=0.1,
        invert=0,
    )
    return captured["spectrum"]


def test_nspin1_sum_cohp_includes_spin_degeneracy(monkeypatch):
    nspin1_spectrum = _calculate_fake_spectrum(_cohp_with_fake_solver(nspin=1), monkeypatch)
    nspin2_sum_spectrum = _calculate_fake_spectrum(_cohp_with_fake_solver(nspin=2), monkeypatch)

    np.testing.assert_allclose(nspin1_spectrum, nspin2_sum_spectrum)


class _TwoKGenerator:
    total_kpoint_num = 2

    def __iter__(self):
        return iter([np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=float)])


class _WeightedSolver(_UnitSolver):
    def get_Hk(self, kpoints):
        values = np.where(kpoints[:, 0] == 0.0, 1.0, 3.0)
        return values[:, None, None].astype(np.complex128)


def test_calculate_cohp_uses_explicit_nonuniform_kpoint_weights(monkeypatch):
    cohp = COHP.__new__(COHP)
    cohp.nspin = 2
    cohp._COHP__k_generator = _TwoKGenerator()
    cohp._COHP__tb_solver = (_WeightedSolver(), _WeightedSolver())
    captured = {}
    selection = types.SimpleNamespace(
        atom_i_orbitals=[0],
        atom_j_orbitals=[0],
        orbital_map=types.SimpleNamespace(total_orbitals=1, atoms=[]),
    )
    monkeypatch.setattr(cohp_module, "resolve_cohp_orbitals", lambda **kwargs: selection)
    monkeypatch.setattr(
        COHP,
        "print_data",
        lambda self, output_prefix, energy, spectrum, *args: captured.update(spectrum=spectrum.copy()),
    )

    cohp.calculate_cohp(
        fermi_energy=0.0,
        stru_file="STRU",
        atom_i_index=1,
        atom_j_index=1,
        method="COHP",
        spin="up",
        e_range=[-1.0, 1.0],
        de=0.5,
        sigma=0.1,
        invert=0,
        kpoint_weights=[0.25, 0.75],
    )

    expected = np.zeros(5)
    cohp._accumulate_one_k(
        expected,
        np.array([[2.5]], dtype=np.complex128),
        np.array([0.0]),
        np.array([[1.0]], dtype=np.complex128),
        [0],
        [0],
        -1.0,
        0.5,
        0.1,
        False,
        pair_factor=1.0,
        kpoint_weight=1.0,
    )
    np.testing.assert_allclose(captured["spectrum"], expected)
