from pyscf import gto, scf

from methods.osrpaoep import OSRPAOEP
from methods.rpaoep import RPAOEP

ORBITAL_BASIS = "aug-cc-pVTZ"
OEP_BASIS = "aug-cc-pVDZ-RIFIT"
RI_BASIS = "aug-cc-pVQZ-RIFIT"
DFIT_BASIS = "aug-cc-pV5Z-RIFIT"
GEOM_N = "N 0.000000 0.000000 0.000000"
SMALL_BASIS = "cc-pVTZ"
SMALL_OEP_BASIS = "cc-pVDZ-RIFIT"


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


def test_closed_shell_matches_rpaoep():
    """One iteration from identical orbitals gives the correlation potential of RPAOEP for both spins."""
    mol = gto.M(atom="H 0 0 -0.370946; H 0 0 0.370946", basis=SMALL_BASIS, verbose=0)
    mf = scf.RHF(mol).run()

    mf_rpa = RPAOEP(mf, SMALL_OEP_BASIS, RI_BASIS)
    mf_rpa.run(maxit=1)

    for spin_sym in (False, True):
        mf_osrpa = OSRPAOEP(mf.to_uhf(), SMALL_OEP_BASIS, RI_BASIS, spin_sym=spin_sym)
        mf_osrpa.run(maxit=1)
        assert abs(mf_osrpa.E_corr - mf_rpa.E_corr) < 1e-12
        assert abs(mf_osrpa.vrest_c_oep_a - mf_rpa.vrest_c_oep).max() < 1e-10
        assert abs(mf_osrpa.vrest_c_oep_b - mf_rpa.vrest_c_oep).max() < 1e-10


def test_h_atom_spin_sym():
    """Without beta electrons the beta channel contributes only to the averaged response function."""
    mol = gto.M(atom="H 0 0 0", basis=SMALL_BASIS, spin=1, verbose=0)

    mf_rpa = OSRPAOEP(scf.UHF(mol).run(), SMALL_OEP_BASIS, RI_BASIS, spin_sym=True)
    mf_rpa.run(maxit=30, e_conv_thr=1e-10)

    assert mf_rpa.converged
    assert mf_rpa.E_corr < 0.0
