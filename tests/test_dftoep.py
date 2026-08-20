from pyscf import dft, gto

from methods.dftoep import DFTOEP

ORBITAL_BASIS = "aug-cc-pwCVTZ"
OEP_BASIS = "aug-cc-pVDZ-RIFIT"
DFIT_BASIS = "aug-cc-pwCV5Z-RIFIT"
GEOM = "C 0.000000    0.000000   -0.646514; O 0.000000    0.000000    0.484886"


def calc_co(xc, xc_homo=None):
    mol = gto.M(atom=GEOM, basis=ORBITAL_BASIS)
    mol.verbose = 0
    mol.symmetry = False

    mf = dft.RKS(mol, xc=xc).density_fit(auxbasis=DFIT_BASIS).run()

    mf_oep = DFTOEP(mf, OEP_BASIS, use_HOMO_condition=False if xc_homo is None else True, xc_homo=xc_homo)
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
