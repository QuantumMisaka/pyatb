# Sidereus integration and COHP comparison — 2026-10-19

The integration base is QuantumMisaka/pyatb `sidereus-adam@9655a85`, not upstream
`main@80f7c2d`. The CSR patch was cherry-picked without conflict as `d638031`.
It changes H/S format reading and ABACUS input R alignment; the branch's COHP
normalization, Python/C++ positive-volume correction, and corresponding tests
are unchanged. No COHP algorithm changes were made during this assessment.

## Integration verification

- Initialized the pinned Eigen submodule `3147391` and ran the complete suite:
  **47 passed in 37.62s**, no skips. This includes compiled native optical/static
  response handedness regression. The preliminary 46-pass/1-skip result was
  superseded after initializing Eigen.
- Replayed the existing develop/LTS Si2/LCAO/nspin=1 matrices through installed
  Forge prepare/execute/collect with the integrated candidate Python source.
  Both passed, each with three events. Their band.dat files equal the original
  installed PyATB LTS result byte-for-byte; no ABACUS rerun was needed.
- The process carrier uses the complete candidate Python source with the
  existing installed native extension and bundled libraries. This does not
  rebuild the extension with the branch's C++ correction. That correction was
  separately exercised by the compiled native test above. A complete candidate
  wheel build and consumption test remains a release requirement.

Exact identities and raw outputs are under
[validation/2026-10-19-sidereus-csr-cohp](validation/2026-10-19-sidereus-csr-cohp/binding.json).

## Comparison against abacus-cohp-lcao

Compared to the actual current remote HEAD
[`78b3628`](https://github.com/QuantumMisaka/abacus-cohp-lcao/tree/78b3628aa4bb137bb75f465b2b4ecc9aac8f9fb1),
whose local tests passed **12/12**. Upstream main COHP is earlier than this
reference; sidereus-adam already addresses some, but not all, differences.

| Area | PyATB sidereus-adam | Current abacus-cohp-lcao | Assessment |
| --- | --- | --- | --- |
| Method/input | Fourier H(R)/S(R), solve eigenpairs at requested k points | Read ABACUS H(k)/S(k), WFC and eigenvalues from SCF output | Both direct NAO-block COHP/COOP, neither is Forge IAO; input/eigensolver paths differ |
| Atom-pair normalization | Different atom IDs get factor 2; same atom ID gets 1 | Equal selected orbital sets get 1; different sets get 2 | Same-atom disjoint shell channels differ by factor 2; confirmed below |
| k weights/spin | Normalized explicit weights or uniform; nspin=1 sum factor 2, nspin=2 up/down/sum | SCF k weights and corresponding spin convention | Main normalization already repaired on sidereus; compare identical k samples for parity |
| Occupied integral | No discrete occupied ICOHP/ICOOP output | Sum weighted states with E <= EF before broadening | Useful missing capability; do not infer it from a truncated smoothed curve |
| Broadening | Continuous Gaussian sampled on user window; sigma is sqrt(2) times standard deviation | Discrete grid kernel normalized to preserve area; padded support; sigma is standard deviation | Same sigma/de does not define equivalent curves; preservation/window policy must be explicit |
| Sign/output | invert changes data; header, plot labels and metadata do not record inversion | Explicit raw/inverted convention and occupied-integral metadata | Confirmed PyATB output interpretation defect |
| Large systems | Build/interpolate full matrices and diagonalize | Stream selected H/S blocks and WFC rows; optional parallel parsing | Different computational route; do not transplant the WFC streaming implementation blindly |
| Units | H converted to eV before contraction; Ry_to_eV=13.605698066 | Corrected direct-reader path uses 13.605693122994 | Do not copy the old reference's unit fix into already-converted PyATB data; harmonize constants separately if strict parity is required |
| Real evidence | Existing COHP tests and limited process cases | Validation summary withdraws historical quantitative cases pending regeneration | Neither historical case list proves current full scientific parity |

Both paths aggregate selected Bloch orbital blocks. Neither comparison establishes
bond-to-a-specific-periodic-image resolution or numerical equivalence to LOBSTER
pCOHP. Neither is a replacement for the distinct Forge IAO method.

## Reproduced issues

The retained [probe](validation/2026-10-19-sidereus-csr-cohp/probe_cohp.py) runs the
actual PyATB calculate/output functions with controlled one-k eigenpairs, and
compares the current reference kernel. It uses the project's MPI/native test
doubles and is not a materials calculation. [Results](validation/2026-10-19-sidereus-csr-cohp/cohp-comparison.json):

1. With H=[[0,1],[1,0]] and C=(1,1)/sqrt(2), disjoint orbital blocks [0]/[1]
   have unordered Hermitian contribution 1. Different atom IDs produce area
   0.9999999998, matching the reference. Assigning the two blocks to the same
   atom produces 0.5 in PyATB while the reference remains 1. This is not an IAO
   versus NAO difference.
2. With invert=1, the area becomes -0.9999999998, but the header still says
   `energy_eV COHP` and metadata has no inversion/sign field. The generated
   plotting script also labels by method alone.
3. A state exactly at the requested lower energy boundary contributes nothing
   because `_accumulate_one_k` excludes it. This should be corrected together
   with an explicit energy-window/broadening contract; no curve-area equality
   should be claimed for the current truncated window.

The standalone reference's own `docs/validation-summary.md` says its older
processed curves and quantitative/performance claims were withdrawn after unit,
pair-factor and integration fixes. Those examples need regeneration. Its README
case listing must not be treated as fresh acceptance evidence.

## Recommended follow-up

- **First, correctness:** define the selected-block convention as identical sets
  versus disjoint sets, fix same-atom cross-channel factors, record inversion in
  metadata/header/plots, and add finite/positive grid validation plus boundary
  tests. Partially overlapping selectors need an explicit definition or rejection;
  copying the reference's blanket factor-2 rule can double-count the overlap.
- **Then, capability:** accumulate occupied discrete ICOHP/ICOOP with k weights,
  spin and sign metadata independently of plotting window, de and sigma; specify
  E<=EF versus finite-temperature occupations. This needs a small output/meaning
  design before implementation, not merely a new plotted integral.
- **Finally, parity:** compare identical H/S/eigenpairs, k weights, orbital
  selections, spin and energy units, then bind real Si and spin-polarized cases.
  Broadening must be matched explicitly. Keep scientific and process acceptance
  separate.

The CSR integration is ready locally. COHP follow-up is warranted, but it is not
part of this CSR patch and was not silently bundled into the branch integration.

Independent review found no integration blockers: the CSR-only code delta
preserves the branch's existing fixes, all 79 candidate Python files match the
runtime binding, and the controlled COHP findings support the stated follow-up.
The integration is local; no remote branch or package has been published.
