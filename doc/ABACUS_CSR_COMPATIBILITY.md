# ABACUS CSR input compatibility

The ABACUS H/S readers accept both the legacy `Matrix Dimension` / `Matrix number`
layout and the annotated HContainer layout (`number of localized basis`,
`number of Bravais lattice vector R`, `CSR Format`). File names do not select the
parser. The modern layout supports wrapped values/indices/pointers, comments,
empty blocks, and complex `(real,imag)` entries. Truncated arrays, invalid CSR
pointers/indices, and non-finite H/S values raise `ValueError`.

The readers retain the first matrix snapshot, upper-triangular storage, HR unit
conversion (Ry/eV), and SR normalization used by the legacy implementation.
The rR reader remains the legacy-layout reader; the observed develop `rr.csr`
also uses that layout. This does not promise support for an unobserved new rR
layout or every ABACUS output variant.

ABACUS may omit different all-zero lattice-vector blocks in H, S and rR.
The CLI and Python `init_tb` ABACUS paths align all supplied matrices on the
ordered union of their R vectors before setting up the solver. Missing sparse
blocks are zero; S-only, spin-down-only, and rR-only vectors are retained.
The input matrix objects are not modified. Solver interfaces and WANNIER90
loading are unchanged.

## Verification (2026-10-19)

- Reader regression: baseline 11 failed / 7 passed; initial fix 18 passed.
- Input alignment regression: baseline 6 failed; fixed along with reader tests.
- Final repository suite: 39 passed. Tests include real compressed Si H/S
  fixtures, old-format whitespace, complex values, zero blocks, malformed
  arrays, nspin=1/2/4 dense/sparse handoff, and alignment ownership checks.
- Real fixtures are ABACUS `94576a801` (develop) and `f71921fe8` (LTS),
  Si2/LCAO/nspin=1. Source hashes are in
  `tests/fixtures/abacus-csr/provenance.json`. H has 93 R blocks in each source;
  S has 63 in develop and 93 in LTS, with the additional LTS blocks all zero.
  After R alignment, every stored H/S value agrees exactly. Legacy H/S/rR
  outputs also agree elementwise with the prior installed reader.
- Installed Forge successfully consumed both sources through separate
  prepare/execute/collect operations with this candidate Python source and
  the existing PyATB 1.1.2 native extension plus its bundled libraries.
  Both band.dat files match the prior LTS result byte-for-byte. No ABACUS
  calculation was repeated for the fix.

The runtime check is a source-candidate check, **not a newly built or published
wheel**. It covers the recorded Si/nspin=1 case; synthetic nspin=2/4 tests do not
establish real spin-polarized or spin-orbit calculations. Scientific band-gap
accuracy and convergence were not assessed. The corresponding Forge evidence
is `docs/superpowers/evidence/2026-10-19-forge-pyatb-reader-closure.md` in the
sibling abacus-forge repository. Existing installed PyATB packages are not
updated by this source change.

Run the offline regression with an environment providing the project's Python
dependencies:

```bash
python -m pytest -q tests
```

The test configuration replaces MPI/native solver interfaces with test doubles.
Real execution evidence above uses the actual native extension, not those doubles.
