import numpy as np
import pytest
from pyscf import cc, gto, lib, scf
from pyscf.cc import ccsd_rdm, ccsd_t_rdm

import utils.relaxed_dm_cc as relaxed_dm_cc
from utils.relaxed_dm_cc import _add_triples_gamma2, relaxed_rdm1, solve_lambda

ORBITAL_BASIS = "cc-pVDZ"
GEOM = "O 0 0 0.117; H 0 0.755 -0.469; H 0 -0.755 -0.469"

# Machine precision is what the two paths actually agree to -- 3e-16 relative on
# every 2-RDM block and 2e-16 on the density, measured here. The margin is for a
# different BLAS or thread count, which changes the reduction order.
ATOL = 1e-12


def converged_ccsd(frozen=None):
    """DF-RHF/cc-pVDZ water, converged CCSD, and the CCSD(T) lambda amplitudes.

    Returns (mycc, eris, l1, l2). The (T) lambdas are what both 2-RDM paths take
    as input, so they are solved once and shared rather than re-derived inside
    relaxed_rdm1, where a failure would surface as a mismatch of densities.
    """
    mol = gto.M(atom=GEOM, basis=ORBITAL_BASIS, verbose=0, symmetry=False)
    mf = scf.RHF(mol).density_fit()
    mf.conv_tol = 1e-12
    mf.kernel()
    assert mf.converged

    mycc = cc.CCSD(mf, frozen=frozen)
    mycc.conv_tol = 1e-10
    mycc.kernel()
    assert mycc.converged

    eris = mycc.ao2mo(mycc.mo_coeff)
    l1, l2 = solve_lambda(mycc, "ccsd(t)", eris=eris)
    assert mycc.converged_lambda
    return mycc, eris, l1, l2


def legacy_gamma_intermediates(mycc, method, t1, t2, l1, l2, eris, h5file, for_grad):
    """The in-core CCSD(T) intermediates this module replaced, for comparison."""
    d1 = ccsd_t_rdm._gamma1_intermediates(mycc, t1, t2, l1, l2, eris, for_grad=for_grad)
    d2 = ccsd_t_rdm._gamma2_outcore(mycc, t1, t2, l1, l2, eris, h5file, True)
    return d1, d2


@pytest.fixture(scope="module")
def water_ccsd():
    return converged_ccsd()


@pytest.mark.parametrize("blksize", [None, 3, 7, 19])
def test_add_triples_gamma2_matches_incore(water_ccsd, blksize):
    """Every 2-RDM block equals the one ccsd_t_rdm builds in core.

    blksize is set explicitly because the automatic choice is a single block at
    this size, which would leave both the blocking and the short final block
    untested.
    """
    mycc, eris, l1, l2 = water_ccsd
    t1, t2 = mycc.t1, mycc.t2

    ref = ccsd_t_rdm._gamma2_outcore(mycc, t1, t2, l1, l2, eris, lib.H5TmpFile(), True)

    h5file = lib.H5TmpFile()
    new = ccsd_rdm._gamma2_outcore(mycc, t1, t2, l1, l2, h5file, True)
    _add_triples_gamma2(mycc, t1, t2, eris, new, blksize=blksize)

    names = ("dovov", "dvvvv", "doooo", "doovv", "dovvo", "dvvov", "dovvv", "dooov")
    for name, a, b in zip(names, ref, new, strict=True):
        if name == "dvvov":
            # ccsd_rdm._gamma2_outcore returns None here and nothing downstream
            # reads it; ccsd_t_rdm returns a transpose of dovvv.
            assert b is None
            continue
        assert np.allclose(np.asarray(a), np.asarray(b), rtol=0, atol=ATOL), name


def test_dvvvv_stays_packed_on_disk(water_ccsd):
    """The point of the change: dvvvv is packed on disk, never a dense nvir**4 array."""
    mycc, eris, l1, l2 = water_ccsd
    nvir = mycc.nmo - mycc.nocc

    h5file = lib.H5TmpFile()
    d2 = ccsd_rdm._gamma2_outcore(mycc, mycc.t1, mycc.t2, l1, l2, h5file, True)
    _add_triples_gamma2(mycc, mycc.t1, mycc.t2, eris, d2)

    dvvvv = d2[1]
    # h5py hands out a fresh Dataset object per lookup, so compare the name.
    assert dvvvv.file == h5file
    assert dvvvv.name == "/dvvvv"
    assert dvvvv.shape == (nvir * (nvir + 1) // 2,) * 2


@pytest.mark.parametrize("frozen", [None, 1])
def test_relaxed_rdm1_matches_incore(frozen, monkeypatch):
    """The AO relaxed density is unchanged, with and without a frozen core."""
    mycc, eris, l1, l2 = converged_ccsd(frozen=frozen)
    dm = relaxed_rdm1(mycc, "ccsd(t)", l1=l1, l2=l2, eris=eris)

    monkeypatch.setattr(relaxed_dm_cc, "_gamma_intermediates", legacy_gamma_intermediates)
    dm_ref = relaxed_rdm1(mycc, "ccsd(t)", l1=l1, l2=l2, eris=eris)

    assert np.allclose(dm, dm_ref, rtol=0, atol=ATOL)
    nelec = np.einsum("pq,qp->", dm, mycc.mol.intor_symmetric("int1e_ovlp"))
    assert abs(nelec - mycc.mol.nelectron) < 1e-8


def pyscf_blksize(max_memory, blk_cap, other):
    """PySCF's own block size in ccsd_t_rdm._gamma1_intermediates, transcribed.

    Deliberately a second copy of what relaxed_dm_cc._pyscf_blksize holds: the
    module inverts this formula, and a test that imported the inversion's own
    idea of it would only check the inversion against itself.
    """
    blksize = min(blk_cap, int(((max_memory * 0.9e6 / 8) / 6.0 / other**3) ** (1 / 3)))
    if blksize < blk_cap:
        blksize = max(min(blksize, (blk_cap + 1) // 2), 1)
    return blksize


class FakeCC:
    """Just the three attributes _gamma1_intermediates_budget reads."""

    def __init__(self, nocc, nmo, max_memory):
        self.nocc = nocc
        self.nmo = nmo
        self.max_memory = max_memory


@pytest.mark.parametrize(
    ("nocc", "nvir", "max_memory"),
    [
        (5, 496, 1_849_844),  # CH4 in the 5Z setup on a 2 TB node -- failed with 5 DGEMM errors
        (8, 394, 1_820_983),  # C2H4_02.6780_000.0 in QZ, same node -- failed with 8
        (8, 394, 1_385_109),  # the same system on a 1.4 TB node -- was already safe
        (5, 288, 1_821_515),  # CH4 in QZ -- already safe
        (2, 67, 342_832),  # LiH in TZ -- small enough that nothing is capped
        (10, 223, 342_895),  # Zundel in TZ
        (25, 1200, 4_000_000),  # beyond anything run so far, where pass 2 binds too
    ],
)
def test_gamma1_budget_keeps_blas_dimensions_in_int32(nocc, nvir, max_memory):
    """The capped budget never lets PySCF pick a block whose contraction overflows int32.

    The inner dimension of the two `abcikl,abcjkl->ij`-shaped contractions
    reaches DGEMM through ctypes.c_int, which truncates rather than raising, so
    an overflow shows up as a silently wrong density.
    """
    mycc = FakeCC(nocc, nocc + nvir, max_memory)
    budget = relaxed_dm_cc._gamma1_intermediates_budget(mycc)
    assert budget <= max_memory

    assert pyscf_blksize(budget, nvir, nocc) ** 3 * nocc**2 < 2**31
    assert pyscf_blksize(budget, nocc, nvir) ** 3 * nvir**2 < 2**31


@pytest.mark.parametrize(
    ("nocc", "nvir", "max_memory"),
    [(8, 394, 1_385_109), (5, 288, 1_821_515), (2, 67, 342_832), (10, 223, 342_895)],
)
def test_gamma1_budget_untouched_when_already_safe(nocc, nvir, max_memory):
    """A run that never had the problem keeps the budget it was given."""
    mycc = FakeCC(nocc, nocc + nvir, max_memory)
    assert relaxed_dm_cc._gamma1_intermediates_budget(mycc) == max_memory
