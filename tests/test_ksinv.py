import numpy as np
from pyscf import gto, scf

from methods.exxoep import EXXOEP
from methods.ksinv import KSINV

ORBITAL_BASIS = "aug-cc-pwCVTZ"
OEP_BASIS = "aug-cc-pVDZ-RIFIT"
DFIT_BASIS = "aug-cc-pwCV5Z-RIFIT"
GEOM = "Ne 0 0 0"


def calc_ne(use_HOMO_condition=False, vh_via_OEP=False):
    mol = gto.M(atom=GEOM, basis=ORBITAL_BASIS)
    mol.verbose = 0
    mol.symmetry = False

    mf = scf.RHF(mol).density_fit(auxbasis=DFIT_BASIS).run()

    mf_oep = EXXOEP(mf, OEP_BASIS, use_HOMO_condition=use_HOMO_condition, vh_via_OEP=vh_via_OEP)
    mf_oep.run(maxit=30, thr_fai_oep=5e-2, e_conv_thr=1e-12)

    if use_HOMO_condition:
        ip = -mf_oep.mf.mo_energy[mf_oep.nelec - 1]
    else:
        ip = None

    mf_inv = KSINV(mf_oep.mf, OEP_BASIS, mf_oep.mf.make_rdm1(), mf_oep.e_tot, ip, vh_via_OEP=vh_via_OEP)
    mf_inv.run(maxit=100, thr_fai_oep=5e-2, conv_thr=1e-12)

    return mf_oep.mf.mo_energy, mf_inv.mf.mo_energy


def test_answer():
    eig1, eig2 = calc_ne()
    assert np.allclose(eig1, eig2)
    eig1, eig2 = calc_ne(True)
    assert np.allclose(eig1, eig2)
    eig1, eig2 = calc_ne(False, True)
    assert np.allclose(eig1, eig2)
