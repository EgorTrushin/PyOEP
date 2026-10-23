#!/usr/bin/env python

"""
Relaxed (response) one-particle density matrices for RHF-CCSD and RHF-CCSD(T).

The relaxed density is the derivative of the coupled-cluster Lagrangian with
respect to the one-electron Hamiltonian,

    h -> h + lambda V    =>    dE/dlambda|_0 = Tr(D_relaxed V)

for any one-electron perturbation V, with the orbital response included.  It is
therefore the density that reproduces analytic gradients and analytic
one-electron properties (dipoles, quadrupoles, electric fields at nuclei, ...).

It is assembled in three steps:

  1. the CC response density D^CC (occ-occ and vir-vir blocks) from the lambda
     amplitudes;
  2. the two-electron part of the generalized Fock matrix, obtained by
     contracting the 2-RDM with the AO integrals;
  3. the occ-vir block, which follows from the Z-vector (CPHF) equation.

For CCSD(T) two extra ingredients are needed, both supplied by PySCF:

  * lambda amplitudes that include the perturbative-triples source terms
    (``pyscf.cc.ccsd_t_lambda``; JCP 147, 044104 (2017));
  * the triples contribution to the 1-RDM intermediates
    (``pyscf.cc.ccsd_t_rdm._gamma1_intermediates``), requested with
    ``for_grad=True`` so that the off-diagonal blocks are kept.  They do not
    contribute to the CCSD(T) energy, but they do drive the Z-vector equation.

The triples contribution to the 2-RDM is assembled here instead, by
``_add_triples_gamma2``; see ``_gamma_intermediates`` for why.

Only canonical RHF orbitals are supported; for the non-canonical formulation see
JCP 95, 2639 (1991).  Frozen core/virtual orbitals are handled.

Derived from ``pyscf.grad.ccsd.grad_elec`` (Qiming Sun, Apache-2.0) and
originally adapted from
https://github.com/andyzynlove/FCNN_XC_Potential/blob/main/relaxed_ccsd.py

Three deviations from stock PySCF are deliberate and are documented at the
relevant functions: the Z-vector equation is solved by conjugate gradient
instead of ``pyscf.scf.cphf.solve`` (see ``_solve_zvector``), the AO integral
pass computes only the contraction a density actually needs (see
``_gfock_2e``), and the CCSD(T) 2-RDM is assembled out of core rather than
through ``ccsd_t_rdm._gamma2_outcore``, which despite its name is not (see
``_gamma_intermediates``).

Usage
-----
    from pyscf import gto, scf, cc
    from utils.relaxed_dm_cc import relaxed_rdm1

    mf = scf.RHF(gto.M(atom=..., basis=...)).run()
    mycc = cc.CCSD(mf, frozen=None).run()
    dm = relaxed_rdm1(mycc, "ccsd(t)")     # AO basis, spin-summed
"""

import os
from contextlib import contextmanager
from functools import reduce

import numpy as np
from pyscf import lib
from pyscf.cc import ccsd_rdm, ccsd_t_lambda, ccsd_t_rdm

# Reused verbatim from PySCF rather than copied into this file, so that they
# cannot drift out of sync with the installed version.
from pyscf.grad.ccsd import _load_block_tril, _rdm2_mo2ao
from pyscf.grad.mp2 import _index_frozen_active, _shell_prange, has_frozen_orbitals
from pyscf.lib import logger

__all__ = ["relaxed_rdm1", "solve_lambda", "unrelaxed_rdm1"]

CCSD = "ccsd"
CCSD_T = "ccsd(t)"


# --------------------------------------------------------------------------- #
#  public API                                                                  #
# --------------------------------------------------------------------------- #
def relaxed_rdm1(
    mycc, method=CCSD_T, t1=None, t2=None, l1=None, l2=None, eris=None, zvec_tol=1e-11, zvec_max_cycle=200, verbose=None
):
    """Relaxed one-particle density matrix in the AO basis (spin-summed).

    Parameters
    ----------
    mycc : pyscf.cc.ccsd.CCSD
        A converged CCSD object built on canonical RHF orbitals.  The same
        object serves both methods -- CCSD(T) adds the triples corrections on
        top of the CCSD amplitudes, it does not need a separate calculation.
    method : {'ccsd', 'ccsd(t)'}
        Which Lagrangian to differentiate.  Default ``'ccsd(t)'``.
    t1, t2 : ndarray, optional
        CCSD amplitudes; taken from ``mycc`` when omitted.
    l1, l2 : ndarray, optional
        Lambda amplitudes, solved here when omitted.  They must match
        ``method``: passing plain CCSD lambdas together with ``'ccsd(t)'``
        silently produces a meaningless hybrid density.
    eris : optional
        ``mycc.ao2mo()`` output, built here when omitted.  Supplying it avoids
        repeating the integral transformation across several calls.
    zvec_tol, zvec_max_cycle : float, int
        Convergence controls for the Z-vector equation.
    verbose : int or Logger, optional

    Returns
    -------
    dm1 : (nao, nao) ndarray
        Relaxed density matrix; ``np.einsum('pq,qp->', dm1, S) == nelectron``.
    """
    method = _normalize_method(method)
    log = logger.new_logger(mycc, verbose)
    time0 = logger.process_clock(), logger.perf_counter()

    t1, t2 = _get_amplitudes(mycc, t1, t2, log)
    if eris is None:
        eris = mycc.ao2mo(mycc.mo_coeff)
    _check_canonical(eris)
    if l1 is None or l2 is None:
        l1, l2 = solve_lambda(mycc, method, t1, t2, eris, verbose=log)

    # The MO 2-RDM and the AO one get separate scratch files, so that the first
    # can be released before the AO integral pass, which is the longest phase
    # and reads only the second.  At nvir = 496 that is 126 GB of scratch not
    # held for the duration.
    fd2 = lib.H5TmpFile()
    # for_grad=True keeps the full goo/gvv blocks of the triples correction.
    # The energy-consistent variant (for_grad=False, used by unrelaxed_rdm1)
    # keeps only their diagonal and must not be used here.
    d1, d2 = _gamma_intermediates(mycc, method, t1, t2, l1, l2, eris, fd2, for_grad=True)
    # dov/dvo are deliberately unused: the Z-vector right-hand side already
    # carries the occ-vir piece, as the comment at the dm1 assembly explains.
    doo, _dov, _dvo, dvv = d1
    time1 = log.timer_debug1("density intermediates", *time0)

    mo_coeff = mycc.mo_coeff
    mo_energy = mycc._scf.mo_energy
    nao, nmo = mo_coeff.shape
    nocc = np.count_nonzero(mycc.mo_occ > 0)
    OA, VA, OF, VF = _index_frozen_active(mycc.get_frozen_mask(), mycc.mo_occ)

    # Two-electron part of the generalized Fock matrix.
    fdm2 = lib.H5TmpFile()
    with _capped_max_memory(mycc, _rdm2_mo2ao_budget(mycc, nao)):
        _rdm2_mo2ao(mycc, d2, mo_coeff[:, np.hstack((OA, VA))], fdm2)
    d2 = None
    _release_tmpfile(fd2)
    time1 = log.timer_debug1("2-RDM MO->AO transformation", *time1)
    Imat = _gfock_2e(mycc, fdm2["dm2"], nao)
    _release_tmpfile(fdm2)
    Imat = reduce(np.dot, (mo_coeff.T, Imat, mycc._scf.get_ovlp(), mo_coeff)) * -1
    time1 = log.timer_debug1("generalized Fock (2e part)", *time1)

    # Correlation density: occ-occ and vir-vir blocks, plus the frozen<->active
    # blocks, which are fixed by the generalized Fock and the orbital energies.
    dm1mo = np.zeros((nmo, nmo))
    if has_frozen_orbitals(mycc):
        dco = Imat[OF[:, None], OA] / (mo_energy[OF, None] - mo_energy[OA])
        dfv = Imat[VF[:, None], VA] / (mo_energy[VF, None] - mo_energy[VA])
        dm1mo[OA[:, None], OA] = doo + doo.T
        dm1mo[OF[:, None], OA] = dco
        dm1mo[OA[:, None], OF] = dco.T
        dm1mo[VA[:, None], VA] = dvv + dvv.T
        dm1mo[VF[:, None], VA] = dfv
        dm1mo[VA[:, None], VF] = dfv.T
    else:
        dm1mo[:nocc, :nocc] = doo + doo.T
        dm1mo[nocc:, nocc:] = dvv + dvv.T

    # Occ-vir block: orbital relaxation.  Its right-hand side already carries
    # the (e_a - e_i) D^CC_ai piece through the generalized Fock, so the
    # solution is the *total* occ-vir density, not just the response part --
    # which is why d1's dov/dvo are never used explicitly.
    dm1 = reduce(np.dot, (mo_coeff, dm1mo, mo_coeff.T))
    vhf = mycc._scf.get_veff(mycc.mol, dm1) * 2
    Xvo = reduce(np.dot, (mo_coeff[:, nocc:].T, vhf, mo_coeff[:, :nocc]))
    Xvo += Imat[:nocc, nocc:].T - Imat[nocc:, :nocc]
    dm1mo += _response_dm1(mycc, Xvo, eris, tol=zvec_tol, max_cycle=zvec_max_cycle, verbose=log)
    log.timer_debug1("Z-vector equation", *time1)

    dm1 = reduce(np.dot, (mo_coeff, dm1mo, mo_coeff.T))
    dm1 += mycc._scf.make_rdm1(mo_coeff, mycc.mo_occ)  # Hartree-Fock reference
    log.timer(f"{method} relaxed rdm1", *time0)
    return dm1


def unrelaxed_rdm1(mycc, method=CCSD_T, t1=None, t2=None, l1=None, l2=None, eris=None, ao_repr=True, verbose=None):
    """Unrelaxed (expectation-value) density -- no orbital relaxation.

    Provided for comparison; it is *not* an energy derivative.  The triples
    intermediates are taken in their energy-consistent form, so that contracting
    this density with the Hamiltonian reproduces the CCSD(T) energy.

    Note that the split of the relaxed density into "unrelaxed + orbital
    relaxation" is convention-dependent for CCSD(T): Molpro partitions it
    differently (the two differ by ~3e-4 in the dipole of water and cancel
    exactly in the sum).  Only the relaxed total is well defined.
    """
    method = _normalize_method(method)
    log = logger.new_logger(mycc, verbose)
    t1, t2 = _get_amplitudes(mycc, t1, t2, log)
    if eris is None:
        eris = mycc.ao2mo(mycc.mo_coeff)
    if l1 is None or l2 is None:
        l1, l2 = solve_lambda(mycc, method, t1, t2, eris, verbose=log)

    if method == CCSD:
        return mycc.make_rdm1(t1, t2, l1, l2, ao_repr=ao_repr)
    return ccsd_t_rdm.make_rdm1(mycc, t1, t2, l1, l2, eris=eris, ao_repr=ao_repr)


def solve_lambda(mycc, method=CCSD_T, t1=None, t2=None, eris=None, max_cycle=50, tol=None, verbose=None):
    """Lambda amplitudes for CCSD or CCSD(T).

    The CCSD(T) lambda equations carry additional source terms from the triples;
    their solution differs from the CCSD lambdas and is required for a
    CCSD(T) density.
    """
    method = _normalize_method(method)
    log = logger.new_logger(mycc, verbose)
    t1, t2 = _get_amplitudes(mycc, t1, t2, log)
    if eris is None:
        eris = mycc.ao2mo(mycc.mo_coeff)
    if tol is None:
        tol = mycc.conv_tol_normt

    if method == CCSD:
        return mycc.solve_lambda(t1, t2, eris=eris)

    conv, l1, l2 = ccsd_t_lambda.kernel(mycc, eris, t1, t2, max_cycle=max_cycle, tol=tol, verbose=log)
    # ``ccsd_t_lambda.kernel`` does not update the flag itself, unlike
    # ``mycc.solve_lambda``, which leaves it stuck at the ``False`` set in
    # ``CCSD.__init__``.  Record it so that ``mycc.converged_lambda`` means the
    # same thing whichever method was requested.  l1/l2 are deliberately not
    # stored on mycc: they are (T) lambdas and must not be picked up by a later
    # plain-CCSD call.
    mycc.converged_lambda = conv
    if not conv:
        log.warn("CCSD(T) lambda equations did not converge")
    return l1, l2


# --------------------------------------------------------------------------- #
#  input handling                                                              #
# --------------------------------------------------------------------------- #
def _normalize_method(method):
    key = str(method).lower().replace(" ", "").replace("_", "").replace("-", "")
    if key in ("ccsd",):
        return CCSD
    if key in ("ccsd(t)", "ccsdt", "(t)", "t"):
        return CCSD_T
    raise ValueError(f"unknown method {method!r}; expected 'ccsd' or 'ccsd(t)'")


def _get_amplitudes(mycc, t1, t2, log):
    if t1 is None:
        t1 = mycc.t1
    if t2 is None:
        t2 = mycc.t2
    if t1 is None or t2 is None:
        raise RuntimeError("CCSD amplitudes are missing -- run mycc.kernel() first")
    if not getattr(mycc, "converged", True):
        log.warn("CCSD amplitudes are not converged")
    return t1, t2


def _check_canonical(eris, tol=1e-6):
    """The Z-vector step and the triples intermediates assume a diagonal Fock."""
    if abs(eris.fock - np.diag(eris.fock.diagonal())).max() > tol:
        raise RuntimeError(
            "relaxed density requires canonical (diagonal-Fock) orbitals; see JCP 95, 2639 (1991) for the general case"
        )


# --------------------------------------------------------------------------- #
#  scratch and memory budget                                                   #
# --------------------------------------------------------------------------- #
def _release_tmpfile(handle):
    """Close a ``lib.H5TmpFile`` and remove it now, not at garbage collection.

    ``H5TmpFile`` deletes through ``weakref.finalize``, so ``close()`` alone
    leaves the file on disk until the object is collected -- and the datasets
    handed out of it keep references alive.  The two 2-RDM files are the largest
    objects a run writes, so their space is reclaimed explicitly instead.  The
    finalizer guards its own removal with the same ``os.path.exists`` check, so
    running twice is harmless.

    Args:
        handle: the H5TmpFile to close and unlink.
    """
    name = handle.filename
    handle.close()
    if os.path.exists(name):
        os.remove(name)


@contextmanager
def _capped_max_memory(mycc, max_memory):
    """Lower ``mycc.max_memory`` for the duration of a block, then restore it.

    Args:
        mycc:       CCSD object whose max_memory is the budget PySCF blocks on.
        max_memory: ceiling in MB; the smaller of this and the current value wins.
    """
    saved = mycc.max_memory
    mycc.max_memory = min(saved, max_memory)
    try:
        yield
    finally:
        mycc.max_memory = saved


def _rdm2_mo2ao_budget(mycc, nao):
    """The budget to hand ``_rdm2_mo2ao``, corrected for what it under-counts.

    ``pyscf.grad.ccsd._rdm2_mo2ao`` sizes its row block as
    ``max_memory * 1e6 / 8 / (nao_pair + nmo**2)`` but then allocates
    ``blksize * (nvir**2 + nvir_pair + 2 * nao_pair)`` doubles in its first pass
    and ``blksize * (2 * nmo**2 + nao_pair)`` in its third -- an under-count of
    about 1.65x at nao = 501.  Left alone with the 90%-of-node-memory default of
    run_ccsdt.py it would pick a block needing more memory than the machine has,
    so it is given a budget scaled down by that ratio instead.

    Args:
        mycc: CCSD object; nmo and nocc are read as the active-space dimensions.
        nao:  number of atomic orbitals.

    Returns:
        Budget in MB.
    """
    nmo, nocc = mycc.nmo, mycc.nocc
    nvir = nmo - nocc
    nao_pair = nao * (nao + 1) // 2
    nvir_pair = nvir * (nvir + 1) // 2
    assumed = nao_pair + nmo**2
    actual = max(nvir**2 + nvir_pair + 2 * nao_pair, 2 * nmo**2 + nao_pair)
    return mycc.max_memory * assumed / actual


def _pyscf_blksize(max_memory, blk_cap, other):
    """The block ``ccsd_t_rdm._gamma1_intermediates`` picks, mirrored from PySCF.

    Both of its passes use the one formula, with ``(blk_cap, other)`` equal to
    ``(nvir, nocc)`` in the first and ``(nocc, nvir)`` in the second.  Reproduced
    here rather than inferred, because ``_gamma1_intermediates_budget`` has to
    invert it; ``tests/test_relaxed_dm_cc.py`` keeps its own copy against which
    this one is checked.

    Args:
        max_memory: budget in MB, already net of the resident set as PySCF takes it.
        blk_cap:    upper bound on the block, nvir in the first pass, nocc in the second.
        other:      the dimension the formula divides by, nocc and nvir respectively.

    Returns:
        The block size, at least 1.
    """
    blksize = min(blk_cap, int(((max_memory * 0.9e6 / 8) / 6.0 / other**3) ** (1 / 3)))
    if blksize < blk_cap:
        blksize = max(min(blksize, (blk_cap + 1) // 2), 1)
    return blksize


def _gamma1_intermediates_budget(mycc):
    """The budget to hand ``ccsd_t_rdm._gamma1_intermediates``, so its BLAS calls fit in int32.

    Each of that function's two passes ends in a contraction over every index of
    the triples buffer but one -- ``'abcikl,abcjkl->ij'`` and
    ``'ijkacd,ijkbcd->ab'``.  ``lib.einsum`` reshapes those into a matrix product
    whose inner dimension is ``blksize**3 * nocc**2`` in the first pass and
    ``blksize**3 * nvir**2`` in the second, and ``lib.ddot`` passes it to DGEMM
    through ``ctypes.c_int``, which truncates silently.  Past 2**31 the k
    argument goes negative, the BLAS error handler prints "parameter number 5
    had an illegal value" and returns without computing anything, and the
    uninitialised output is accumulated as though it were a result.  The density
    that comes out is wrong -- 4e-2 electrons on C2H4 in the QZ setup, 3e-3 on
    CH4 in 5Z -- and nothing but the electron-count check at the end of
    run_ccsdt.py says so.

    The block size grows with the memory PySCF is given, so this is a failure
    mode of *large* nodes: it needs a budget big enough to put the whole virtual
    space in one block, which is when the ``(nvir + 1) // 2`` fallback that
    otherwise keeps the product small is skipped.  Every affected run in this
    project had --max-memory left at the default 90% of a 2 TB node.

    Args:
        mycc: CCSD object; nmo and nocc are read as the active-space dimensions.

    Returns:
        Budget in MB -- ``mycc.max_memory`` itself wherever that is already safe,
        so nothing is given up on the runs that never had the problem.
    """
    mem_now = lib.current_memory()[0]
    nocc, nvir = mycc.nocc, mycc.nmo - mycc.nocc
    budget = mycc.max_memory
    for blk_cap, other in ((nvir, nocc), (nocc, nvir)):
        if _pyscf_blksize(max(0, budget - mem_now), blk_cap, other) ** 3 * other**2 < 2**31:
            continue
        # Invert the formula for the largest block that stays inside the limit.
        # PySCF halves the block whenever it lands below blk_cap, so that halved
        # value is the most a capped budget can reach. The resident set goes back
        # on because PySCF subtracts it again -- by then it is larger than it is
        # here, so the block it settles on can only be smaller than the one asked
        # for, never larger.
        largest = int(((2**31 - 1) / other**2) ** (1 / 3))
        blk = min(largest, (blk_cap + 1) // 2)
        budget = min(budget, blk**3 * 6.0 * other**3 * 8 / 0.9e6 + mem_now)
    return budget


# --------------------------------------------------------------------------- #
#  density intermediates                                                       #
# --------------------------------------------------------------------------- #
def _gamma_intermediates(mycc, method, t1, t2, l1, l2, eris, h5file, for_grad):
    """1- and 2-RDM intermediates in the MO basis, for either method.

    Both methods take their 2-RDM from ``ccsd_rdm._gamma2_outcore``, which is
    genuinely out of core: it writes ``dvvvv`` into ``h5file`` in packed
    ``(nvir_pair, nvir_pair)`` form, one virtual block at a time, and never
    holds more than ``(blksize, nvir, nvir, nvir)`` in memory.
    ``ccsd_t_rdm`` exports a function of the same name and signature, but it
    discards its file argument and forwards to the in-core routine, whose dense
    ``(nvir, nvir, nvir, nvir)`` intermediate -- 484 GB at nvir = 496, and as
    much again in scratch -- is what used to cap the CCSD(T) density at small
    bases.  CCSD(T) therefore builds the CCSD 2-RDM here as well and adds the
    triples on top; the split is exact because the triples never touch
    ``dvvvv``, so packing it before or after them is the same thing.  See
    ``_add_triples_gamma2``.

    ``for_grad`` reaches ``d1`` only: the triples 2-RDM has no
    energy-consistent variant to choose between.
    """
    if method == CCSD:
        d1 = ccsd_rdm._gamma1_intermediates(mycc, t1, t2, l1, l2)
        d2 = ccsd_rdm._gamma2_outcore(mycc, t1, t2, l1, l2, h5file, True)
    else:
        # d1 first: ccsd_t_rdm._gamma1_intermediates allocates its own pair of
        # triples buffers, which are freed on return.
        with _capped_max_memory(mycc, _gamma1_intermediates_budget(mycc)):
            d1 = ccsd_t_rdm._gamma1_intermediates(mycc, t1, t2, l1, l2, eris, for_grad=for_grad)
        d2 = ccsd_rdm._gamma2_outcore(mycc, t1, t2, l1, l2, h5file, True)
        _add_triples_gamma2(mycc, t1, t2, eris, d2)
    return d1, d2


def _add_triples_gamma2(mycc, t1, t2, eris, d2, blksize=None):
    """Add the perturbative-triples correction to an out-of-core CCSD 2-RDM, in place.

    Reproduces the triples half of ``ccsd_t_rdm._gamma2_intermediates`` term for
    term without ever allocating a dense ``(nvir, nvir, nvir, nvir)`` array.
    Only three of the eight intermediates are affected -- ``dovov``, ``dooov``
    and ``dovvv`` -- which is what makes the split from the CCSD half exact.

    The loop over virtual blocks is reordered from PySCF's ``c, b, a`` to
    ``a, b, c``.  Every (a, b, c) block contributes an independent additive term
    built from read-only quantities, so the order is free -- and with ``a``
    outermost the one large target, ``dovvv[:, a0:a1]``, is read once,
    accumulated over the whole b/c sweep and written back once, two passes over
    it in total.  Narrowing that buffer in ``b`` as well would save memory but no
    I/O: ``dovvv`` is chunked ``(nocc, nocc, 1, nvir)``, so a hyperslab
    restricted in its last index still transfers whole chunks spanning the full
    ``nvir``, and the b loop would re-read the entire slab each pass.  Reading
    the target block first and accumulating into it, rather than summing into a
    zero buffer and adding at the end, also keeps the order of the
    floating-point additions the same as PySCF's.

    Args:
        mycc:    CCSD object; only max_memory, stdout and verbose are read.
        t1, t2:  CCSD amplitudes, active space only.
        eris:    mycc.ao2mo() output.  Needs get_ovvv(), ovoo, ovov, fock and
                 mo_energy; ``vvvv`` is never touched, so the density-fitted
                 dfccsd integrals -- which have none -- are enough.
        d2:      (dovov, dvvvv, doooo, doovv, dovvo, dvvov, dovvv, dooov) as
                 returned by ccsd_rdm._gamma2_outcore(..., compress_vvvv=True).
                 Its dovov, dooov and dovvv members are updated in place,
                 whether they are h5py datasets or plain arrays; the other five
                 are left untouched.
        blksize: Override the virtual block size, otherwise derived from
                 mycc.max_memory.  Passing it explicitly is how the tests reach
                 the blocking on a system whose automatic choice is one block.

    Returns:
        d2: the tuple that was passed in, for convenience.
    """
    log = logger.new_logger(mycc)
    dovov, _dvvvv, _doooo, _doovv, _dovvo, _dvvov, dovvv, dooov = d2
    nocc, nvir = t1.shape

    eris_ovvv = np.asarray(eris.get_ovvv())
    eris_ovoo = np.asarray(eris.ovoo)
    eris_ovov = np.asarray(eris.ovov)
    eia = lib.direct_sum("i-a->ia", eris.mo_energy[:nocc], eris.mo_energy[nocc:])

    if blksize is None:
        # PySCF charges six w/v-sized buffers -- the two arrays plus the einsum,
        # direct_sum and ``2 * w_blk`` temporaries -- and gets two thirds of the
        # budget here; the dovvv slab and the temporary that feeds it take the
        # rest.  The 400 MB floor is not cosmetic: PySCF's bare ``max(0, ...)``
        # collapses blksize to 1 as soon as the resident set exceeds max_memory,
        # and the loop then runs nvir**3 times on scalar blocks.
        max_memory = max(400, mycc.max_memory - lib.current_memory()[0])
        budget = max_memory * 0.9e6 / 8
        blksize = min(
            nvir,
            int((budget * 2.0 / 3.0 / 6.0 / nocc**3) ** (1.0 / 3)),
            int(budget / 3.0 / (2.0 * nocc * nvir**2)),
        )
        if blksize < nvir:
            blksize = min(blksize, (nvir + 1) // 2)
        blksize = max(blksize, 1)
    nblk = (nvir + blksize - 1) // blksize
    log.debug1("triples gamma2: nocc,nvir = %d,%d  blksize = %d  (%d**3 blocks)", nocc, nvir, blksize, nblk)

    # np.zeros, not PySCF's np.empty: ``v_blk += 2 * w_blk`` and the symmetrizer
    # run over the whole preallocated buffer, so an uninitialized tail is
    # rescaled every iteration until it overflows to inf and then nan.  It never
    # contaminates [:ba, :bb, :bc] -- t3_symm_ip permutes only the three
    # occupied indices, so it never mixes virtual blocks -- but it does raise
    # numpy warnings, and zeros stay zeros under both operations.
    w_blk = np.zeros((blksize, blksize, blksize, nocc, nocc, nocc), dtype=t1.dtype)
    v_blk = np.zeros((blksize, blksize, blksize, nocc, nocc, nocc), dtype=t1.dtype)

    # dovov (nocc**2 nvir**2) and dooov (nocc**3 nvir) are small enough to stay
    # resident.  np.asarray copies an h5py dataset and returns a plain ndarray
    # unchanged, so both flavours of d2 work: in the second case the
    # accumulation writes straight through and the writeback is a self-assign.
    dovov_buf = np.asarray(dovov)
    dooov_buf = np.asarray(dooov)

    time1 = logger.process_clock(), logger.perf_counter()
    for a0, a1 in lib.prange(0, nvir, blksize):
        ba = a1 - a0
        dovvv_buf = np.asarray(dovvv[:, a0:a1])
        for b0, b1 in lib.prange(0, nvir, blksize):
            bb = b1 - b0
            for c0, c1 in lib.prange(0, nvir, blksize):
                bc = c1 - c0
                w = w_blk[:ba, :bb, :bc]
                v = v_blk[:ba, :bb, :bc]

                w[:] = lib.einsum("iafb,kjcf->abcijk", eris_ovvv[:, a0:a1, :, b0:b1], t2[:, :, c0:c1, :])
                w += lib.einsum("iafc,jkbf->abcijk", eris_ovvv[:, a0:a1, :, c0:c1], t2[:, :, b0:b1, :])
                w += lib.einsum("jbfa,kicf->abcijk", eris_ovvv[:, b0:b1, :, a0:a1], t2[:, :, c0:c1, :])
                w += lib.einsum("jbfc,ikaf->abcijk", eris_ovvv[:, b0:b1, :, c0:c1], t2[:, :, a0:a1, :])
                w += lib.einsum("kcfa,jibf->abcijk", eris_ovvv[:, c0:c1, :, a0:a1], t2[:, :, b0:b1, :])
                w += lib.einsum("kcfb,ijaf->abcijk", eris_ovvv[:, c0:c1, :, b0:b1], t2[:, :, a0:a1, :])
                w -= lib.einsum("iajm,mkbc->abcijk", eris_ovoo[:, a0:a1, :, :], t2[:, :, b0:b1, c0:c1])
                w -= lib.einsum("iakm,mjcb->abcijk", eris_ovoo[:, a0:a1, :, :], t2[:, :, c0:c1, b0:b1])
                w -= lib.einsum("jbim,mkac->abcijk", eris_ovoo[:, b0:b1, :, :], t2[:, :, a0:a1, c0:c1])
                w -= lib.einsum("jbkm,mica->abcijk", eris_ovoo[:, b0:b1, :, :], t2[:, :, c0:c1, a0:a1])
                w -= lib.einsum("kcim,mjab->abcijk", eris_ovoo[:, c0:c1, :, :], t2[:, :, a0:a1, b0:b1])
                w -= lib.einsum("kcjm,miba->abcijk", eris_ovoo[:, c0:c1, :, :], t2[:, :, b0:b1, a0:a1])

                v[:] = lib.einsum("iajb,kc->abcijk", eris_ovov[:, a0:a1, :, b0:b1], t1[:, c0:c1])
                v += lib.einsum("iakc,jb->abcijk", eris_ovov[:, a0:a1, :, c0:c1], t1[:, b0:b1])
                v += lib.einsum("jbkc,ia->abcijk", eris_ovov[:, b0:b1, :, c0:c1], t1[:, a0:a1])
                v += lib.einsum("ck,ijab->abcijk", eris.fock[nocc + c0 : nocc + c1, :nocc], t2[:, :, a0:a1, b0:b1])
                v += lib.einsum("ai,jkbc->abcijk", eris.fock[nocc + a0 : nocc + a1, :nocc], t2[:, :, b0:b1, c0:c1])
                v += lib.einsum("bj,kica->abcijk", eris.fock[nocc + b0 : nocc + b1, :nocc], t2[:, :, c0:c1, a0:a1])

                d3_blk = lib.direct_sum("ia,jb,kc->abcijk", eia[:, a0:a1], eia[:, b0:b1], eia[:, c0:c1])
                w /= d3_blk
                v /= d3_blk
                d3_blk = None

                # On the full buffers, never on the [:ba, :bb, :bc] views: the C
                # kernel asserts C-contiguity, and a view is strided whenever a
                # block is short.  Safe -- it acts on each abc index alone.
                v_blk += 2.0 * w_blk
                ccsd_t_rdm.t3_symm_ip_py(v_blk, blksize**3, nocc, "4-2-211-2", 1.0, 0.0)
                ccsd_t_rdm.t3_symm_ip_py(w_blk, blksize**3, nocc, "4-2-211-2", 0.5, 0.0)

                dovov_buf[:, a0:a1, :, b0:b1] += lib.einsum("kc,abcijk->iajb", t1[:, c0:c1], w)
                dooov_buf[:, :, :, a0:a1] -= lib.einsum("mkbc,abcijk->jmia", t2[:, :, b0:b1, c0:c1], v)
                dovvv_buf[:, :, :, b0:b1] += lib.einsum("kjcf,abcijk->iafb", t2[:, :, c0:c1, :], v)

        dovvv[:, a0:a1] = dovvv_buf
        time1 = log.timer_debug1(f"triples gamma2 [{a0}:{a1}]", *time1)

    dovov[:] = dovov_buf
    dooov[:] = dooov_buf
    return d2


# --------------------------------------------------------------------------- #
#  generalized Fock matrix                                                     #
# --------------------------------------------------------------------------- #
def _gfock_2e(mycc, dm2ao, nao):
    """Two-electron generalized Fock matrix, ``I_pq = sum_ix (ip|x) Gamma_(iq|x)``.

    ``dm2ao`` is the AO-basis 2-RDM written by ``_rdm2_mo2ao``, with its last
    index in lower-triangular (s2) packing.

    This single contraction is all the AO integral pass has to produce for a
    density.  The gradient code this was derived from additionally contracted
    the derivative integrals ``int2e_ip1`` (three components) and accumulated
    per-atom Hartree-Fock terms into a ``(natm, 3, nao, nao)`` buffer in the
    same loop; for a density that is roughly 2.7x the useful work, plus storage
    growing as ``natm * nao**2``, so it is dropped here.  Blocking over atoms
    served the same gradient-only purpose and is replaced by plain shell
    blocking.
    """
    mol = mycc.mol
    diagidx = np.arange(nao)
    diagidx = diagidx * (diagidx + 1) // 2 + diagidx

    # eri0 and dm2buf are each (nf, nao, nao_pair), i.e. ~nf*nao**3/2 doubles.
    max_memory = max(400, mycc.max_memory - lib.current_memory()[0])
    blksize = max(1, int(max_memory * 0.9e6 / 8 / (nao**3 * 2.0)))

    Imat = np.zeros((nao, nao))
    ip1 = 0
    for b0, b1, nf in _shell_prange(mol, 0, mol.nbas, blksize):
        ip0, ip1 = ip1, ip1 + nf
        dm2buf = _load_block_tril(dm2ao, ip0, ip1, nao)
        dm2buf[:, :, diagidx] *= 0.5
        shls_slice = (b0, b1, 0, mol.nbas, 0, mol.nbas, 0, mol.nbas)
        eri0 = mol.intor("int2e", aosym="s2kl", shls_slice=shls_slice)
        Imat += lib.einsum("ipx,iqx->pq", eri0.reshape(nf, nao, -1), dm2buf)
        eri0 = dm2buf = None
    return Imat


# --------------------------------------------------------------------------- #
#  orbital relaxation (Z-vector / CPHF)                                        #
# --------------------------------------------------------------------------- #
def _solve_zvector(fvind, mo_energy, mo_occ, Xvo, tol=1e-11, max_cycle=200, verbose=None):
    """Solve the Z-vector (CPHF) equation

        (e_a - e_i) z_ai + sum_bj A_ai,bj z_bj = -Xvo_ai

    by preconditioned conjugate gradient.

    The RHF orbital Hessian A_ai,bj = 4(ai|bj) - (ab|ij) - (aj|bi) is symmetric
    and, for a stable SCF solution, positive definite, so CG converges to the
    exact solution.  ``pyscf.scf.cphf.solve`` uses ``lib.krylov`` instead, whose
    trial space is cut off by a linear-dependency threshold; it silently stalls
    at a relative accuracy of ~1e-4 in z no matter which tolerance is requested.
    That shows up as ~5e-7 errors in relaxed one-electron properties -- and in
    PySCF's own CCSD/CCSD(T) gradients.  With CG, agreement with a finite-field
    reference improves from ~4e-7 to ~1e-10.
    """
    ea = mo_energy[mo_occ == 0]
    ei = mo_energy[mo_occ > 0]
    eai = ea[:, None] - ei  # strictly positive for an aufbau reference

    def matvec(x):
        return eai * x + fvind(x).reshape(Xvo.shape)

    b = -Xvo
    bnorm = np.linalg.norm(b)
    if bnorm == 0:
        return np.zeros_like(Xvo)

    x = b / eai  # uncoupled (diagonal) guess
    r = b - matvec(x)
    z = r / eai
    p = z.copy()
    rz = np.einsum("ai,ai->", r, z)
    for _ in range(max_cycle):
        if np.linalg.norm(r) < tol * bnorm:
            break
        Ap = matvec(p)
        alpha = rz / np.einsum("ai,ai->", p, Ap)
        x += alpha * p
        r -= alpha * Ap
        z = r / eai
        rz_new = np.einsum("ai,ai->", r, z)
        p = z + (rz_new / rz) * p
        rz = rz_new
    if np.linalg.norm(r) >= tol * bnorm:
        logger.warn(verbose, "Z-vector CG did not converge: |r|/|b| = %.3e", np.linalg.norm(r) / bnorm)
    return x


def _response_dm1(mycc, Xvo, eris=None, tol=1e-11, max_cycle=200, verbose=None):
    """Occ-vir block of the relaxed density, from the Z-vector equation."""
    nvir, nocc = Xvo.shape
    nmo = nocc + nvir
    mo_coeff = mycc.mo_coeff

    # The MO-integral branch below holds a full (nocc, nvir, nvir, nocc) array,
    # which grows as nvir**2; fall back to the J/K builder when it would not fit
    # in what is left of the budget.
    ovvo_fits = nocc**2 * nvir**2 * 8 / 1e6 < 0.5 * max(0, mycc.max_memory - lib.current_memory()[0])
    if eris is None or has_frozen_orbitals(mycc) or not ovvo_fits:
        # Apply the orbital Hessian through the SCF's J/K builder.  Required
        # with frozen orbitals, where eris only spans the active space.
        mo_energy = mycc._scf.mo_energy
        mo_occ = mycc.mo_occ

        def fvind(x):
            dm = reduce(np.dot, (mo_coeff[:, nocc:], x.reshape(nvir, nocc), mo_coeff[:, :nocc].T))
            v = mycc._scf.get_veff(mycc.mol, dm + dm.T)
            return reduce(np.dot, (mo_coeff[:, nocc:].T, v, mo_coeff[:, :nocc])) * 2

    else:
        # A_ai,bj = 4(ia|bj) - (ib|aj) - (ij|ab), straight out of the MO integrals.
        mo_energy = eris.mo_energy
        mo_occ = np.zeros_like(mo_energy)
        mo_occ[:nocc] = 2
        ovvo = np.empty((nocc, nvir, nvir, nocc))
        for i in range(nocc):
            ovvo[i] = eris.ovvo[i]
            ovvo[i] = ovvo[i] * 4 - ovvo[i].transpose(1, 0, 2)
            ovvo[i] -= eris.oovv[i].transpose(2, 1, 0)

        def fvind(x):
            return np.einsum("iabj,bj->ai", ovvo, x.reshape(nvir, nocc))

    dvo = _solve_zvector(fvind, mo_energy, mo_occ, Xvo, tol=tol, max_cycle=max_cycle, verbose=verbose)
    dm1 = np.zeros((nmo, nmo))
    dm1[nocc:, :nocc] = dvo
    dm1[:nocc, nocc:] = dvo.T
    return dm1
