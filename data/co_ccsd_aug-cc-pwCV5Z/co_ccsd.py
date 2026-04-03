#!/usr/bin/env python3

import numpy as np
from pyscf import gto, scf, cc
from relaxed_ccsd import cc_rrdm1

ORBITAL_BASIS = "aug-cc-pwCV5Z"
GEOM = "C 0.000000    0.000000   -0.646514; O 0.000000    0.000000    0.484886"

mol = gto.M(atom=GEOM, basis=ORBITAL_BASIS)
mol.verbose = 0
mol.symmetry = False

mf = scf.RHF(mol)
mf.kernel()
print(f"Hartree-Fock total energy: {mf.e_tot:15.12f}", flush=True)
dm_hf = mf.make_rdm1()

mf_cc = cc.CCSD(mf)
mf_cc.kernel()
print(f"CCSD correlation energy:     {mf_cc.e_corr:15.12f}", flush=True)
print(f"CCSD total energy:         {mf_cc.e_tot:15.12f}", flush=True)
dm_ccsd_unrelaxed = mf_cc.make_rdm1(ao_repr=True)
dm_ccsd = cc_rrdm1(mf_cc)

with open('dm.npy', 'wb') as f:
    np.save(f, dm_ccsd)
with open('energy.txt', 'w') as f:
    print(mf_cc.e_tot, file=f)
