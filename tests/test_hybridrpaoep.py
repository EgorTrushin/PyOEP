from pyscf import gto, scf

from methods.hybridrpaoep import HybridRPAOEP
from methods.rpaoep import RPAOEP

ORBITAL_BASIS = "aug-cc-pVTZ"
OEP_BASIS = "aug-cc-pVDZ-RIFIT"
RI_BASIS = "aug-cc-pVQZ-RIFIT"
DFIT_BASIS = "aug-cc-pV5Z-RIFIT"
GEOM_H2 = "H 0 0 -0.370946; H 0 0 0.370946"
GEOM_Ne = "Ne 0 0 0"


def calc_ne(geom, alpha=0.25, use_HOMO_condition=False):
    mol = gto.M(atom=geom, basis=ORBITAL_BASIS)
    mol.verbose = 0
    mol.symmetry = False

    mf = scf.RHF(mol).density_fit(auxbasis=DFIT_BASIS).run()

    mf_rpa = HybridRPAOEP(mf, OEP_BASIS, RI_BASIS, alpha=alpha, use_HOMO_condition=use_HOMO_condition)
    mf_rpa.run(maxit=30, thr_fai_oep=5e-2, e_conv_thr=1e-10)

    return mf_rpa


def test_answer():
    mf_rpa = calc_ne(GEOM_H2)
    assert abs(mf_rpa.e_tot - -1.202259232157) < 1e-9
    mf_rpa = calc_ne(GEOM_Ne)
    assert abs(mf_rpa.e_tot - -128.929072733831) < 1e-9
    mf_rpa = calc_ne(GEOM_Ne, use_HOMO_condition=True)
    assert abs(mf_rpa.e_tot - -128.928960424257) < 1e-9
    mf_rpa = calc_ne(GEOM_Ne, alpha=0.5)
    assert abs(mf_rpa.e_tot - -128.910774284874) < 1e-9


def test_alpha_zero_is_rpaoep():
    mol = gto.M(atom=GEOM_H2, basis=ORBITAL_BASIS, verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis=DFIT_BASIS).run()

    e_tot = []
    for mf_rpa in (RPAOEP(mf, OEP_BASIS, RI_BASIS), HybridRPAOEP(mf, OEP_BASIS, RI_BASIS, alpha=0.0)):
        mf_rpa.run(maxit=30, thr_fai_oep=5e-2, e_conv_thr=1e-10)
        e_tot.append(mf_rpa.e_tot)

    assert abs(e_tot[1] - e_tot[0]) < 1e-10
