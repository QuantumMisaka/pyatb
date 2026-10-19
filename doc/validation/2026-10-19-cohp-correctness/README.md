# COHP correctness and discrete integrals — 2026-10-19

Base: local `sidereus-adam@dad27b5`, including the develop/LTS CSR integration.
The user authorized the fixes and improvements identified in the previous review.
Scope: PyATB COHP/COOP; no Forge operation or IAO algorithm changes.

## Decisions and implemented behavior

- Ruling: identical resolved orbital sets use factor 1; disjoint sets use 2,
  including channels on one atom. Reject partial overlaps and empty sets until
  a different decomposition is explicitly defined. Atom IDs alone cannot decide
  the pair factor. This makes the Hermitian contribution unambiguous.
- Inversion now appears in the spectrum header, JSON and generated plot label.
  File names and two-column data layout stay compatible; default label is -COHP.
- Validate finite Fermi/range values, positive finite sigma/de and an increasing
  range with at least two samples. Include Gaussian tails and boundary states;
  preserve the continuous Gaussian definition and do not rescale truncated plots.
- Ruling: add discrete zero-temperature E<=EF ICOHP/ICOOP to existing JSON output.
  Record raw and displayed values, units, cutoff convention and spin degeneracy.
  No inferred finite-temperature occupation or material bonding conclusion.
- Reuse each k/spin orbital contraction for spectrum and integral; no extra
  diagonalization or dense state/grid array. Keep chunked solving and MPI sums.
- Normalize large finite k weights safely by their maximum before summing.
- Native replay discovered an existing optional-input bug: None is required in
  the generic parser, so omitted kpoint_weights failed. Use the existing empty
  list default convention and map it to uniform weights in the COHP adapter.
  Explicit numeric-list keywords with no values raise ValueError; omitted
  weights and malformed explicit weights must not be conflated.

## Evidence

- Test-first kernel/output regressions: baseline 24 failed / 8 passed; failures
  include wrong same-atom factors, missing sign/integral output and bad grids.
- Omitted-weight parser regression: baseline 1 failed / 5 passed.
- Independent review caught explicit empty weights silently defaulting; its
  regression failed before correction (1 failed / 6 passed), then 7 passed.
- Final complete suite: **77 passed in 32.94s**, no skips. Native replay outputs are retained alongside this
  report; see `tests.txt`, `replay.txt`, `runtime-binding.json` and
  `si-summary.json`. Native optical/static handedness regression is included.
- Six real Si replays consume retained ABACUS matrices without another SCF run:
  develop/LTS COHP and COOP, two MPI ranks with uneven chunks, and a narrow plot
  window. Both tracks agree; MPI/chunk spectra agree; occupied ICOHP is unchanged
  when the plot shrinks from [-15,25] to [0,1] eV.
- For the same-atom s/p query, raw ICOHP is 0.03990074886873071 eV in all four
  relevant replays. This is an operational/numerical invariant, not a validated
  material bond-energy interpretation or parity with the standalone WFC route.
- Analytic tests cover both spin channels, nspin=1 degeneracy, explicit nonuniform
  k weights, chunk offsets, COHP/COOP, sign, exact EF cutoff and window exclusion.

## Reproduction and limitations

Run `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q tests` in the
abacus-env environment. Initialize the pinned Eigen submodule for the native
handedness test. `replay_si.py --help` documents retained-input and carrier paths;
raw process files remain under `/home/james/scratch/pyatb-cohp-validation-20261019/si-v3`.
The source/native binding identifies all staged Python files and reused libraries.

The carrier uses candidate Python with the existing installed native extension;
it is not a newly built wheel. Real replay is Si/nspin=1; spin-polarized behavior
has analytic regression coverage. Full candidate wheel build/install, real
spin-polarized material coverage, matched-input standalone parity and performance
benchmarking remain separate follow-up. No quantitative speedup is claimed.
Old standalone processed curves remain withdrawn and are not golden data.
No remote push or package publication is performed by this task.

Independent code review found no remaining blockers after correcting the explicit
empty-weight fallback. Review covered numerical conventions, MPI/chunk handling,
input compatibility and the scope of real-run evidence.
