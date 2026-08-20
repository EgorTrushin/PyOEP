#!/usr/bin/env python3

import numpy as np
from pyscf import fci, gto, scf

mol = gto.M(atom="B 0 0 0", basis="aug-cc-pwCVTZ", symmetry=False, verbose=0, spin=1)
mf = scf.UHF(mol)
mf.run()
print(f"Hartree-Fock total energy: {mf.e_tot:15.12f}", flush=True)

cisolver = fci.FCI(mf)
es, fcivec = cisolver.kernel()
print(f"FCI total energy: {es:15.12f}", flush=True)

nelec_a = 3
nelec_b = 2
norb = mf.mo_coeff.shape[1]
dm1a, dm1b = cisolver.make_rdm1s(fcivec, norb, (nelec_a, nelec_b))
dm1a = mf.mo_coeff[0] @ dm1a @ mf.mo_coeff[0].T
dm1b = mf.mo_coeff[1] @ dm1b @ mf.mo_coeff[1].T
dm = np.stack((dm1a, dm1b))

with open("dm.npy", "wb") as f:
    np.save(f, dm)
with open("energy.txt", "w") as f:
    print(es, file=f)
