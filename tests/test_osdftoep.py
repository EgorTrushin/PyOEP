from pyscf import dft, gto

from methods.osdftoep import OSDFTOEP

ORBITAL_BASIS = "aug-cc-pwCVTZ"
OEP_BASIS = "aug-cc-pVDZ-RIFIT"
DFIT_BASIS = "aug-cc-pwCV5Z-RIFIT"


def calc_n(xc, xc_homo=None, spin_sym=False):
    mol = gto.M(atom="N 0. 0. 0.", basis=ORBITAL_BASIS, spin=3)
    mol.verbose = 0
    mol.symmetry = False

    mf = dft.UKS(mol, xc=xc).density_fit(auxbasis=DFIT_BASIS).run()

    mf_oep = OSDFTOEP(
        mf,
        OEP_BASIS,
        use_HOMO_condition=xc_homo is not None,
        xc_homo=xc_homo,
        spin_sym=spin_sym,
    )
    mf_oep.run(maxit=30, thr_fai_oep=5e-2, e_conv_thr=1e-10)

    return mf_oep.e_tot


def calc_c(xc, xc_homo=None, space_sym=False, spin_sym=False):
    mol = gto.M(atom="C 0. 0. 0.", basis=ORBITAL_BASIS, spin=2)
    mol.verbose = 0
    mol.symmetry = False

    mf = dft.UKS(mol, xc=xc).density_fit(auxbasis=DFIT_BASIS).run()

    mf_oep = OSDFTOEP(
        mf,
        OEP_BASIS,
        use_HOMO_condition=xc_homo is not None,
        xc_homo=xc_homo,
        space_sym=space_sym,
        spin_sym=spin_sym,
    )
    mf_oep.run(maxit=30, thr_fai_oep=5e-2, e_conv_thr=1e-10)

    return mf_oep.e_tot


def test_answer():
    e_tot = calc_n("PBE")
    assert abs(e_tot + 54.532003117326) < 1e-8
    e_tot = calc_n("PBE0")
    assert abs(e_tot + 54.542686688645) < 1e-8
    e_tot = calc_n("PBE", spin_sym=True)
    assert abs(e_tot + 54.530396151804) < 1e-8
    e_tot = calc_c("PBE", space_sym=True)
    assert abs(e_tot + 37.794812686916) < 1e-7
    e_tot = calc_c("PBE0", space_sym=True, spin_sym=True)
    assert abs(e_tot + 37.802823840794) < 1e-7
    e_tot = calc_n("PBE", xc_homo="GGA_X_PBE")
    assert abs(e_tot + 54.531135832818) < 1e-7
    e_tot = calc_n("PBE", xc_homo="GGA_X_PBE", spin_sym=True)
    assert abs(e_tot + 54.529604545581) < 1e-7
    e_tot = calc_c("PBE", xc_homo="GGA_X_PBE", space_sym=True)
    assert abs(e_tot + 37.794494655618) < 1e-7
