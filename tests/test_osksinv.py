import numpy as np
from pyscf import gto, scf

from methods.osexxoep import OSEXXOEP
from methods.osksinv import OSKSINV

ORBITAL_BASIS = "aug-cc-pwCVTZ"
OEP_BASIS = "aug-cc-pVDZ-RIFIT"
DFIT_BASIS = "aug-cc-pwCV5Z-RIFIT"


def calc_n(use_HOMO_condition=False, vh_via_OEP=False, spin_sym=False):
    mol = gto.M(atom="N 0. 0. 0.", basis=ORBITAL_BASIS, spin=3)
    mol.verbose = 0
    mol.symmetry = False

    mf = scf.UHF(mol).density_fit(auxbasis=DFIT_BASIS).run()

    mf_oep = OSEXXOEP(mf, OEP_BASIS, use_HOMO_condition=use_HOMO_condition, vh_via_OEP=vh_via_OEP, spin_sym=spin_sym)
    mf_oep.run(maxit=30, thr_fai_oep=5e-2, e_conv_thr=1e-11)

    if use_HOMO_condition:
        ip = [-mf_oep.mf.mo_energy[0][mf_oep.nelec[0] - 1], -mf_oep.mf.mo_energy[1][mf_oep.nelec[1] - 1]]
    else:
        ip = None

    mf_inv = OSKSINV(
        mf_oep.mf, OEP_BASIS, mf_oep.mf.make_rdm1(), mf_oep.e_tot, ip=ip, vh_via_OEP=vh_via_OEP, spin_sym=spin_sym
    )
    mf_inv.run(maxit=100, thr_fai_oep=5e-2, conv_thr=1e-12)

    return mf_oep.mf.mo_energy, mf_inv.mf.mo_energy


def calc_c(space_sym=False, spin_sym=False):
    mol = gto.M(atom="C 0. 0. 0.", basis=ORBITAL_BASIS, spin=2)
    mol.verbose = 0
    mol.symmetry = False

    mf = scf.UHF(mol).density_fit(auxbasis=DFIT_BASIS).run()

    mf_oep = OSEXXOEP(mf, OEP_BASIS, space_sym=space_sym, spin_sym=spin_sym)
    mf_oep.run(maxit=30, thr_fai_oep=5e-2, e_conv_thr=1e-11)

    mf_inv = OSKSINV(mf_oep.mf, OEP_BASIS, mf_oep.mf.make_rdm1(), mf_oep.e_tot, space_sym=space_sym, spin_sym=spin_sym)
    mf_inv.run(maxit=100, thr_fai_oep=5e-2, conv_thr=1e-12)

    return mf_oep.mf.mo_energy, mf_inv.mf.mo_energy


def test_answer():
    eig1, eig2 = calc_n()
    assert np.allclose(eig1, eig2)
    eig1, eig2 = calc_n(use_HOMO_condition=True)
    assert np.allclose(eig1, eig2)
    eig1, eig2 = calc_n(vh_via_OEP=True)
    assert np.allclose(eig1, eig2)
    eig1, eig2 = calc_n(spin_sym=True)
    assert np.allclose(eig1, eig2)
    eig1, eig2 = calc_c(space_sym=True)
    assert np.allclose(eig1, eig2)
    eig1, eig2 = calc_c(space_sym=True, spin_sym=True)
    assert np.allclose(eig1, eig2)
