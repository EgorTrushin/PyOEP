import numpy as np
from pyscf import gto, scf

from methods.exxoep import EXXOEP

ORBITAL_BASIS = "aug-cc-pwCVTZ"
OEP_BASIS = "aug-cc-pVDZ-RIFIT"
DFIT_BASIS = "aug-cc-pwCV5Z-RIFIT"
GEOM = "C 0.000000    0.000000   -0.646514; O 0.000000    0.000000    0.484886"
SMALL_BASIS = "cc-pVTZ"
SMALL_OEP_BASIS = "cc-pVDZ-RIFIT"


def calc_co(use_HOMO_condition=False):
    mol = gto.M(atom=GEOM, basis=ORBITAL_BASIS)
    mol.verbose = 0
    mol.symmetry = False

    mf = scf.RHF(mol).density_fit(auxbasis=DFIT_BASIS).run()

    mf_oep = EXXOEP(mf, OEP_BASIS, use_HOMO_condition=use_HOMO_condition)
    mf_oep.run(maxit=30, thr_fai_oep=5e-2, e_conv_thr=1e-12)

    return mf_oep.e_tot


def test_answer():
    e_tot = calc_co()
    assert abs(e_tot + 112.774772513104) < 1e-8
    e_tot = calc_co(True)
    assert abs(e_tot + 112.774365350216) < 1e-8


def test_cartesian_basis():
    mol = gto.M(atom="Ne 0. 0. 0.", basis=SMALL_BASIS, cart=True, verbose=0)
    mf = scf.RHF(mol).run()

    mf_oep = EXXOEP(mf, SMALL_OEP_BASIS)
    mf_oep.run(maxit=30, e_conv_thr=1e-10)

    assert mf_oep.converged
    assert 0.0 < mf_oep.e_tot - mf.e_tot < 1e-2


def test_pmol_follows_mol_geometry():
    """pmol must have the units and the symmetry-adapted orientation of mf.mol."""
    mol = gto.M(atom="H 0. 0. 0.; H 1.398397 0. 0.", basis=SMALL_BASIS, unit="Bohr", verbose=0)
    mf_oep = EXXOEP(scf.RHF(mol).run(), SMALL_OEP_BASIS)
    assert np.allclose(mf_oep.pmol.atom_coords(), mol.atom_coords())

    mol = gto.M(atom="O 0. 0. 0.; H 0.757 0.586 0.; H -0.757 0.586 0.", basis="cc-pVDZ", symmetry=True, verbose=0)
    mf_oep = EXXOEP(scf.RHF(mol).run(), SMALL_OEP_BASIS)
    assert np.allclose(mf_oep.pmol.atom_coords(), mol.atom_coords())
