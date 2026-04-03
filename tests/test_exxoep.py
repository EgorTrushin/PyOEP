from pyscf import scf, gto
from methods.exxoep import EXXOEP

ORBITAL_BASIS = "aug-cc-pwCVTZ"
OEP_BASIS = "aug-cc-pVDZ-RIFIT"
DFIT_BASIS = "aug-cc-pwCV5Z-RIFIT"
GEOM = "C 0.000000    0.000000   -0.646514; O 0.000000    0.000000    0.484886"


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
