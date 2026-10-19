"""Bounded comparison of PyATB orchestration and the current NAO reference kernel."""
import argparse
import json
from pathlib import Path
import runpy
import sys
import tempfile
import types

parser=argparse.ArgumentParser()
parser.add_argument('--pyatb-repo',type=Path,required=True)
parser.add_argument('--reference-repo',type=Path,required=True)
args=parser.parse_args()
# Same controlled MPI/native stand-ins as the project's offline tests.
runpy.run_path(str(args.pyatb_repo/'tests/conftest.py'))
import numpy as np
import pytest
from pyatb.fermi.cohp import COHP
import pyatb.fermi.cohp as module
sys.path.insert(0,str(args.reference_repo/'src'))
import cohp as reference

class Generator:
    total_kpoint_num=1
    def __iter__(self):return iter([np.zeros((1,3))])

class Solver:
    def diago_H(self,k):return np.array([[[1/np.sqrt(2)],[1/np.sqrt(2)]]]),np.array([[1.]])
    def get_Hk(self,k):return np.array([[[0.,1.],[1.,0.]]])

report={'reference_commit':'78b3628','pyatb_baseline':'9655a85','cases':{}}
with tempfile.TemporaryDirectory() as directory, pytest.MonkeyPatch.context() as patch:
    patch.setattr(module,'timer',types.SimpleNamespace(start=lambda *a:None,end=lambda *a:None))
    patch.setattr(module,'RUNNING_LOG',str(Path(directory)/'running.log'))
    selection=types.SimpleNamespace(atom_i_orbitals=[0],atom_j_orbitals=[1],orbital_map=types.SimpleNamespace(total_orbitals=2,atoms=[]))
    patch.setattr(module,'resolve_cohp_orbitals',lambda **kw:selection)
    for name,j,invert in [('different_atoms',2,0),('same_atom_disjoint_channels',1,0),('inverted_output',2,1)]:
        obj=COHP.__new__(COHP);obj.nspin=2;obj.output_path=directory
        obj._COHP__k_generator=Generator();obj._COHP__tb_solver=(Solver(),Solver())
        obj.calculate_cohp(fermi_energy=0.,stru_file='unused',atom_i_index=1,atom_j_index=j,spin='up',e_range=[-1,3],de=.01,sigma=.1,invert=invert,output_prefix=name)
        data=np.loadtxt(Path(directory)/(name+'.dat'))
        meta=json.loads((Path(directory)/(name+'.meta.json')).read_text())
        energies,raw=reference.cal_COHPvalskIJ_e(np.array([[0.,1.],[1.,0.]]),np.eye(2),np.array([1.]),np.array([[1/np.sqrt(2)],[1/np.sqrt(2)]]),[0],[1])
        expected=float(raw[0]*reference._unordered_pair_factor([0],[1]))*(-1 if invert else 1)
        report['cases'][name]={'pyatb_integrated_curve':float(np.trapezoid(data[:,1],data[:,0])),
           'reference_discrete_weight':expected,'pyatb_pair_factor':meta['pair_factor'],
           'header':(Path(directory)/(name+'.dat')).read_text().splitlines()[0],
           'metadata_records_invert':('invert' in meta or 'sign_convention' in meta)}
    boundary=np.zeros(201)
    obj._accumulate_one_k(boundary,np.array([[1.]]),np.array([-1.]),np.array([[1.]]),[0],[0],-1.,.01,.1,False)
    report['lower_window_boundary']={'nonzero_points':int(np.count_nonzero(boundary)), 'state_weight':1.}
assert abs(report['cases']['different_atoms']['pyatb_integrated_curve']-1)<1e-6
assert abs(report['cases']['same_atom_disjoint_channels']['pyatb_integrated_curve']-.5)<1e-6
assert report['cases']['inverted_output']['metadata_records_invert'] is False
assert report['lower_window_boundary']['nonzero_points']==0
print(json.dumps(report,indent=2))
