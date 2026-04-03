from pyscf import scf, gto
from methods.osexxoep import OSEXXOEP

ORBITAL_BASIS = "aug-cc-pwCVTZ"
OEP_BASIS = "aug-cc-pVDZ-RIFIT"
DFIT_BASIS = "aug-cc-pwCV5Z-RIFIT"


def calc_n(use_HOMO_condition=False, spin_sym=False):
    mol = gto.M(atom="N 0. 0. 0.", basis=ORBITAL_BASIS, spin=3)
    mol.verbose = 0
    mol.symmetry = False

    mf = scf.UHF(mol).density_fit(auxbasis=DFIT_BASIS).run()

    mf_oep = OSEXXOEP(mf, OEP_BASIS, use_HOMO_condition=use_HOMO_condition, spin_sym=spin_sym)
    mf_oep.run(maxit=30, thr_fai_oep=5e-2, e_conv_thr=1e-11)

    return mf_oep.e_tot


def calc_c(space_sym=False, spin_sym=False):
    mol = gto.M(atom="C 0. 0. 0.", basis=ORBITAL_BASIS, spin=2)
    mol.verbose = 0
    mol.symmetry = False

    mf = scf.UHF(mol).density_fit(auxbasis=DFIT_BASIS).run()

    mf_oep = OSEXXOEP(mf, OEP_BASIS, space_sym=space_sym, spin_sym=spin_sym)
    mf_oep.run(maxit=30, thr_fai_oep=5e-2, e_conv_thr=1e-11)

    return mf_oep.e_tot


def test_answer():
    e_tot = calc_n()
    assert abs(e_tot + 54.399240606508) < 1e-8
    e_tot = calc_n(use_HOMO_condition=True)
    assert abs(e_tot + 54.399706354831) < 1e-8
    e_tot = calc_n(spin_sym=True)
    assert abs(e_tot + 54.394473170239) < 1e-8
    e_tot = calc_n(use_HOMO_condition=True, spin_sym=True)
    assert abs(e_tot + 54.394833684022) < 1e-8
    e_tot = calc_c(space_sym=True)
    assert abs(e_tot + 37.686169582145) < 1e-7
    e_tot = calc_c(space_sym=True, spin_sym=True)
    assert abs(e_tot + 37.684519237019) < 1e-7
