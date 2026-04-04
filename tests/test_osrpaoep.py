from pyscf import scf, gto
from methods.osrpaoep import OSRPAOEP

ORBITAL_BASIS = "aug-cc-pVTZ"
OEP_BASIS = "aug-cc-pVDZ-RIFIT"
RI_BASIS = "aug-cc-pVQZ-RIFIT"
DFIT_BASIS = "aug-cc-pV5Z-RIFIT"
GEOM_N = "N 0.000000 0.000000 0.000000"


def calc_n(geom, use_HOMO_condition=False, spin_sym=False):
    mol = gto.M(atom=geom, basis=ORBITAL_BASIS, spin=3)
    mol.verbose = 0
    mol.symmetry = False

    mf = scf.UHF(mol).density_fit(auxbasis=DFIT_BASIS).run()

    mf_rpa = OSRPAOEP(mf, OEP_BASIS, RI_BASIS, use_HOMO_condition=use_HOMO_condition, spin_sym=spin_sym)
    mf_rpa.run(maxit=50, thr_fai_oep=5e-2, e_conv_thr=1e-10)

    return mf_rpa


def test_answer():
    mf_rpa = calc_n(GEOM_N)
    assert abs(mf_rpa.e_tot - -54.614873653347) < 1e-7
    mf_rpa = calc_n(GEOM_N, use_HOMO_condition=True)
    assert abs(mf_rpa.e_tot - -54.614838280433) < 1e-7
    mf_rpa = calc_n(GEOM_N, use_HOMO_condition=False, spin_sym=True)
    assert abs(mf_rpa.e_tot - -54.613253176548) < 1e-7
