from pyscf import gto, scf

from methods.rpaoep import RPAOEP

ORBITAL_BASIS = "aug-cc-pVTZ"
OEP_BASIS = "aug-cc-pVDZ-RIFIT"
RI_BASIS = "aug-cc-pVQZ-RIFIT"
DFIT_BASIS = "aug-cc-pV5Z-RIFIT"
GEOM_H2 = "H 0 0 -0.370946; H 0 0 0.370946"
GEOM_Ne = "Ne 0 0 0"


def calc_ne(geom, use_HOMO_condition=False, vh_via_OEP=False):
    mol = gto.M(atom=geom, basis=ORBITAL_BASIS)
    mol.verbose = 0
    mol.symmetry = False

    mf = scf.RHF(mol).density_fit(auxbasis=DFIT_BASIS).run()

    mf_rpa = RPAOEP(mf, OEP_BASIS, RI_BASIS, use_HOMO_condition=use_HOMO_condition, vh_via_OEP=vh_via_OEP)
    mf_rpa.run(maxit=30, thr_fai_oep=5e-2, e_conv_thr=1e-10)

    return mf_rpa


def test_answer():
    mf_rpa = calc_ne(GEOM_H2)
    assert abs(mf_rpa.e_tot - -1.208955824618) < 1e-9
    mf_rpa = calc_ne(GEOM_H2, use_HOMO_condition=True)
    assert abs(mf_rpa.e_tot - -1.208955865137) < 1e-9
    mf_rpa = calc_ne(GEOM_H2, use_HOMO_condition=False, vh_via_OEP=True)
    assert abs(mf_rpa.e_tot - -1.208956107762) < 1e-9
    mf_rpa = calc_ne(GEOM_Ne)
    assert abs(mf_rpa.e_tot - -128.949562741762) < 1e-9
    mf_rpa = calc_ne(GEOM_Ne, use_HOMO_condition=True)
    assert abs(mf_rpa.e_tot - -128.949336937358) < 1e-9
