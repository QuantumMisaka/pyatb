"""Replay retained Si matrices with a staged native PyATB carrier, no ABACUS run."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, required=True, help='Prior forge-replay root with develop/lts inputs')
parser.add_argument('--producer', type=Path, required=True, help='Prior dual-track ABACUS output root with orbitals')
parser.add_argument('--runtime', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--mpiexec', type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
results = {}
for track, method, narrow, ranks in [
    ('develop', 'COHP', False, 1), ('lts', 'COHP', False, 1),
    ('develop', 'COHP', False, 2), ('develop', 'COHP', True, 1),
    ('develop', 'COOP', False, 1), ('lts', 'COOP', False, 1),
]:
    name = f'{track}-{method}-narrow{int(narrow)}-ranks{ranks}'
    work = args.output / name
    work.mkdir()
    previous = args.source / track / 'inputs'
    for path in (previous / 'pyatb_sources').iterdir():
        dest = work / 'pyatb_sources' / path.name
        dest.parent.mkdir(exist_ok=True)
        shutil.copyfile(path, dest)
    shutil.copyfile(previous / 'STRU', work / 'STRU')
    orbital = next((args.producer / track).rglob('Si_gga_7au_100Ry_2s2p1d.orb'))
    shutil.copyfile(orbital, work / orbital.name)
    prefix = (previous / 'Input').read_text().split('BAND_STRUCTURE')[0]
    # Exercise uneven MPI partitions and chunk offsets over the eight MP points.
    prefix = prefix.replace('max_kpoint_num  100', 'max_kpoint_num  3')
    selection = ('s', 'p') if method == 'COHP' else ('all', 'all')
    window = '0.0 1.0' if narrow else '-15.0 25.0'
    (work / 'Input').write_text(prefix + f'''COHP
{{
    stru_file STRU
    atom_i_index 1
    atom_j_index 1
    atom_i_orbs {selection[0]}
    atom_j_orbs {selection[1]}
    orbital_dir .
    method {method}
    spin sum
    e_range {window}
    de 0.05
    sigma 0.15
    invert 1
    kpoint_mode mp
    mp_grid 2 2 2
}}
''')
    command = [str(args.runtime / 'pyatb-candidate')]
    if ranks > 1:
        command = [str(args.mpiexec), '-n', str(ranks), *command]
    env = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
    with (work / 'stdout.log').open('w') as out, (work / 'stderr.log').open('w') as err:
        run = subprocess.run(command, cwd=work, env=env, stdout=out, stderr=err, timeout=120)
    if run.returncode:
        raise RuntimeError(f'{name}: exit {run.returncode}; see {work}/stderr.log')
    meta = json.loads((work / 'Out/COHP/COHP.meta.json').read_text())
    data = np.loadtxt(work / 'Out/COHP/COHP.dat')
    assert np.isfinite(data).all()
    assert meta['spectrum_label'] == '-' + method
    results[name] = {'ranks': ranks, 'occupied_integral': meta['occupied_integral'],
                     'pair_factor': meta['pair_factor'], 'samples': len(data),
                     'input_sha256': {str(p.relative_to(work)): hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in [work/'Input', work/'STRU', work/orbital.name,
                                                *sorted((work/'pyatb_sources').iterdir())]}}
    (args.output / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')
for method in ('COHP', 'COOP'):
    left, right = [args.output / f'{track}-{method}-narrow0-ranks1' / 'Out/COHP/COHP.dat'
                   for track in ('develop', 'lts')]
    np.testing.assert_allclose(np.loadtxt(left), np.loadtxt(right), atol=1e-7, rtol=1e-7)
base = results['develop-COHP-narrow0-ranks1']['occupied_integral']['raw_value']
for name in ('develop-COHP-narrow0-ranks2', 'develop-COHP-narrow1-ranks1', 'lts-COHP-narrow0-ranks1'):
    np.testing.assert_allclose(results[name]['occupied_integral']['raw_value'], base, atol=1e-10, rtol=1e-10)
a = np.loadtxt(args.output/'develop-COHP-narrow0-ranks1/Out/COHP/COHP.dat')
b = np.loadtxt(args.output/'develop-COHP-narrow0-ranks2/Out/COHP/COHP.dat')
np.testing.assert_allclose(a, b, atol=1e-8, rtol=1e-8)
print('Six native Si replays passed; dual-track, MPI/chunk and window invariants passed.')
