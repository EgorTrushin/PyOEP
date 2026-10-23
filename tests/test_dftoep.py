import numpy as np
import pytest
from pyscf import dft, gto

from methods.dftoep import DFTOEP

ORBITAL_BASIS = "aug-cc-pwCVTZ"
OEP_BASIS = "aug-cc-pVDZ-RIFIT"
DFIT_BASIS = "aug-cc-pwCV5Z-RIFIT"
GEOM = "C 0.000000    0.000000   -0.646514; O 0.000000    0.000000    0.484886"
SMALL_BASIS = "cc-pVTZ"
SMALL_OEP_BASIS = "cc-pVDZ-RIFIT"


def calc_co(xc, xc_homo=None):
    mol = gto.M(atom=GEOM, basis=ORBITAL_BASIS)
    mol.verbose = 0
    mol.symmetry = False

    mf = dft.RKS(mol, xc=xc).density_fit(auxbasis=DFIT_BASIS).run()

    mf_oep = DFTOEP(mf, OEP_BASIS, use_HOMO_condition=xc_homo is not None, xc_homo=xc_homo)
    mf_oep.run(maxit=30, thr_fai_oep=5e-2, e_conv_thr=1e-9)

    return mf_oep.e_tot


def test_answer():
    e_tot = calc_co("PBE")
    assert abs(e_tot + 113.233206090694) < 1e-8
    e_tot = calc_co("PBE", "PBE,")
    assert abs(e_tot + 113.232935514499) < 1e-8
    e_tot = calc_co("PBE0")
    assert abs(e_tot + 113.228120995117) < 1e-8
    e_tot = calc_co("PBE0", "0.25*HF+0.75*GGA_X_PBE")
    assert abs(e_tot + 113.228038173075) < 1e-8


@pytest.mark.parametrize("xc", ["CAM-B3LYP", "wB97X-V"])
def test_range_separated_and_vv10(xc):
    """Short- and long-range exact exchange and VV10 are part of the potential replaced by the local one."""
    mol = gto.M(atom="Ne 0. 0. 0.", basis=SMALL_BASIS, verbose=0)
    mf = dft.RKS(mol, xc=xc).density_fit().run()

    mf_oep = DFTOEP(mf, SMALL_OEP_BASIS)
    mf_oep.run(maxit=30, e_conv_thr=1e-9)

    assert mf_oep.converged
    # OEP is the generalized KS minimization restricted to local potentials
    assert 0.0 < mf_oep.e_tot - mf.e_tot < 1e-2


@pytest.mark.parametrize("auxbasis", ["cc-pVTZ-RIFIT", None], ids=["custom-df", "no-df"])
def test_ip_uses_grids_and_df_of_mf(auxbasis):
    mol = gto.M(atom=GEOM, basis=SMALL_BASIS, verbose=0)
    mf = dft.RKS(mol, xc="PBE")
    if auxbasis is not None:
        mf = mf.density_fit(auxbasis=auxbasis)
    mf.run()

    mf_oep = DFTOEP(mf, SMALL_OEP_BASIS, use_HOMO_condition=True, xc_homo="PBE,")

    # frozen-orbital IP with the grids and density fitting of mf
    mf_x = dft.UKS(mol, xc="PBE,")
    mf_x.grids = mf.grids
    if auxbasis is not None:
        mf_x = mf_x.density_fit(with_df=mf.with_df)
    mo_coeff = np.stack((mf.mo_coeff, mf.mo_coeff))
    mo_occ = np.stack((mf.mo_occ, mf.mo_occ)) / 2.0
    e_n = mf_x.energy_tot(mf_x.make_rdm1(mo_coeff, mo_occ))
    mo_occ[1, mol.nelectron // 2 - 1] = 0
    ip = mf_x.energy_tot(mf_x.make_rdm1(mo_coeff, mo_occ)) - e_n

    assert abs(mf_oep.ip - ip) < 1e-10


def test_homo_condition_with_vh_via_oep():
    mol = gto.M(atom=GEOM, basis=SMALL_BASIS, verbose=0)
    mf = dft.RKS(mol, xc="PBE").density_fit().run()

    mf_oep = DFTOEP(mf, SMALL_OEP_BASIS, use_HOMO_condition=True, xc_homo="PBE,", vh_via_OEP=True)
    mf_oep.run(maxit=40, e_conv_thr=1e-10)

    assert mf_oep.converged
    assert abs(mf_oep.mf.mo_energy[mol.nelectron // 2 - 1] + mf_oep.ip) < 1e-6
