#!/usr/bin/env python

"""
CCSD relaxed density matrix
Adapted from https://github.com/andyzynlove/FCNN_XC_Potential/blob/main/relaxed_ccsd.py
"""

import ctypes
import numpy as np
from pyscf import lib
from functools import reduce
from pyscf.cc import ccsd
from pyscf.cc import _ccsd
from pyscf.cc import ccsd_rdm
from pyscf.scf import cphf
from pyscf.grad.mp2 import _shell_prange, _index_frozen_active


def cc_rrdm1(cc_grad, t1=None, t2=None, l1=None, l2=None, eris=None, atmlst=None, d1=None, d2=None):
    mycc = cc_grad
    if eris is not None:
        if abs(eris.fock - np.diag(eris.fock.diagonal())).max() > 1e-3:
            raise RuntimeError("CCSD gradients does not support NHF (non-canonical HF)")

    if t1 is None:
        t1 = mycc.t1
    if t2 is None:
        t2 = mycc.t2
    if l1 is None:
        l1 = mycc.l1
    if l2 is None:
        l2 = mycc.l2

    if d1 is None:
        d1 = ccsd_rdm._gamma1_intermediates(mycc, t1, t2, l1, l2)
    doo, dov, dvo, dvv = d1
    fdm2 = lib.H5TmpFile()
    if d2 is None:
        d2 = ccsd_rdm._gamma2_outcore(mycc, t1, t2, l1, l2, fdm2, True)

    mol = cc_grad.mol
    mo_coeff = mycc.mo_coeff
    mo_energy = mycc._scf.mo_energy
    nao, nmo = mo_coeff.shape
    nocc = np.count_nonzero(mycc.mo_occ > 0)
    with_frozen = not (
        (mycc.frozen is None)
        or (isinstance(mycc.frozen, (int, np.integer)) and mycc.frozen == 0)
        or (len(mycc.frozen) == 0)
    )
    OA, VA, OF, VF = _index_frozen_active(mycc.get_frozen_mask(), mycc.mo_occ)
    mo_active = mo_coeff[:, np.hstack((OA, VA))]
    _rdm2_mo2ao(mycc, d2, mo_active, fdm2)  # transform the active orbitals
    hf_dm1 = mycc._scf.make_rdm1(mycc.mo_coeff, mycc.mo_occ)

    if atmlst is None:
        atmlst = range(mol.natm)
    offsetdic = mol.offset_nr_by_atom()
    diagidx = np.arange(nao)
    diagidx = diagidx * (diagidx + 1) // 2 + diagidx
    de = np.zeros((len(atmlst), 3))
    Imat = np.zeros((nao, nao))
    vhf1 = fdm2.create_dataset("vhf1", (len(atmlst), 3, nao, nao), "f8")

    # 2e AO integrals dot 2pdm
    max_memory = max(0, mycc.max_memory - lib.current_memory()[0])
    blksize = max(1, int(max_memory * 0.9e6 / 8 / (nao**3 * 2.5)))

    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = offsetdic[ia]
        ip1 = p0
        vhf = np.zeros((3, nao, nao))
        for b0, b1, nf in _shell_prange(mol, shl0, shl1, blksize):
            ip0, ip1 = ip1, ip1 + nf
            dm2buf = _load_block_tril(fdm2["dm2"], ip0, ip1, nao)
            dm2buf[:, :, diagidx] *= 0.5
            shls_slice = (b0, b1, 0, mol.nbas, 0, mol.nbas, 0, mol.nbas)
            eri0 = mol.intor("int2e", aosym="s2kl", shls_slice=shls_slice)
            Imat += lib.einsum("ipx,iqx->pq", eri0.reshape(nf, nao, -1), dm2buf)
            eri0 = None

            eri1 = mol.intor("int2e_ip1", comp=3, aosym="s2kl", shls_slice=shls_slice).reshape(3, nf, nao, -1)
            de[k] -= np.einsum("xijk,ijk->x", eri1, dm2buf) * 2
            dm2buf = None
            # HF part
            for i in range(3):
                eri1tmp = lib.unpack_tril(eri1[i].reshape(nf * nao, -1))
                eri1tmp = eri1tmp.reshape(nf, nao, nao, nao)
                vhf[i] += np.einsum("ijkl,ij->kl", eri1tmp, hf_dm1[ip0:ip1])
                vhf[i] -= np.einsum("ijkl,il->kj", eri1tmp, hf_dm1[ip0:ip1]) * 0.5
                vhf[i, ip0:ip1] += np.einsum("ijkl,kl->ij", eri1tmp, hf_dm1)
                vhf[i, ip0:ip1] -= np.einsum("ijkl,jk->il", eri1tmp, hf_dm1) * 0.5
            eri1 = eri1tmp = None
        vhf1[k] = vhf

    Imat = reduce(np.dot, (mo_coeff.T, Imat, mycc._scf.get_ovlp(), mo_coeff)) * -1

    dm1mo = np.zeros((nmo, nmo))
    if with_frozen:
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

    dm1 = reduce(np.dot, (mo_coeff, dm1mo, mo_coeff.T))
    vhf = mycc._scf.get_veff(mycc.mol, dm1) * 2
    Xvo = reduce(np.dot, (mo_coeff[:, nocc:].T, vhf, mo_coeff[:, :nocc]))
    Xvo += Imat[:nocc, nocc:].T - Imat[nocc:, :nocc]

    dm1mo += _response_dm1(mycc, Xvo, eris)

    Imat[nocc:, :nocc] = Imat[:nocc, nocc:].T

    zeta = lib.direct_sum("i+j->ij", mo_energy, mo_energy) * 0.5
    zeta[nocc:, :nocc] = mo_energy[:nocc]
    zeta[:nocc, nocc:] = mo_energy[:nocc].reshape(-1, 1)
    zeta = reduce(np.dot, (mo_coeff, zeta * dm1mo, mo_coeff.T))

    dm1 = reduce(np.dot, (mo_coeff, dm1mo, mo_coeff.T))
    dm1 += hf_dm1

    return dm1


def _response_dm1(mycc, Xvo, eris=None):
    nvir, nocc = Xvo.shape
    nmo = nocc + nvir
    with_frozen = not (
        (mycc.frozen is None)
        or (isinstance(mycc.frozen, (int, np.integer)) and mycc.frozen == 0)
        or (len(mycc.frozen) == 0)
    )
    if eris is None or with_frozen:
        mo_energy = mycc._scf.mo_energy
        mo_occ = mycc.mo_occ
        mo_coeff = mycc.mo_coeff

        def fvind(x):
            x = x.reshape(Xvo.shape)
            dm = reduce(np.dot, (mo_coeff[:, nocc:], x, mo_coeff[:, :nocc].T))
            v = mycc._scf.get_veff(mycc.mol, dm + dm.T)
            v = reduce(np.dot, (mo_coeff[:, nocc:].T, v, mo_coeff[:, :nocc]))
            return v * 2

    else:
        mo_energy = eris.mo_energy
        mo_occ = np.zeros_like(mo_energy)
        mo_occ[:nocc] = 2
        ovvo = np.empty((nocc, nvir, nvir, nocc))
        for i in range(nocc):
            ovvo[i] = eris.ovvo[i]
            ovvo[i] = ovvo[i] * 4 - ovvo[i].transpose(1, 0, 2)
            ovvo[i] -= eris.oovv[i].transpose(2, 1, 0)

        def fvind(x):
            return np.einsum("iabj,bj->ai", ovvo, x.reshape(Xvo.shape))

    dvo = cphf.solve(fvind, mo_energy, mo_occ, Xvo, max_cycle=30)[0]
    dm1 = np.zeros((nmo, nmo))
    dm1[nocc:, :nocc] = dvo
    dm1[:nocc, nocc:] = dvo.T
    return dm1


def _rdm2_mo2ao(mycc, d2, mo_coeff, fsave=None):
    if fsave is None:
        incore = True
        fsave = lib.H5TmpFile()
    else:
        incore = False
    dovov, dvvvv, doooo, doovv, dovvo, dvvov, dovvv, dooov = d2

    nocc, nvir = dovov.shape[:2]
    mo_coeff = np.asarray(mo_coeff, order="F")
    nao, nmo = mo_coeff.shape
    nao_pair = nao * (nao + 1) // 2
    nvir_pair = nvir * (nvir + 1) // 2

    fdrv = getattr(_ccsd.libcc, "AO2MOnr_e2_drv")
    ftrans = _ccsd.libcc.AO2MOtranse2_nr_s1
    fmm = _ccsd.libcc.CCmmm_transpose_sum
    pao_loc = ctypes.POINTER(ctypes.c_void_p)()

    def _trans(vin, orbs_slice, out=None):
        nrow = vin.shape[0]
        if out is None:
            out = np.empty((nrow, nao_pair))
        fdrv(
            ftrans,
            fmm,
            out.ctypes.data_as(ctypes.c_void_p),
            vin.ctypes.data_as(ctypes.c_void_p),
            mo_coeff.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nrow),
            ctypes.c_int(nao),
            (ctypes.c_int * 4)(*orbs_slice),
            pao_loc,
            ctypes.c_int(0),
        )
        return out

    fswap = lib.H5TmpFile()
    max_memory = mycc.max_memory - lib.current_memory()[0]
    blksize = int(max_memory * 1e6 / 8 / (nao_pair + nmo**2))
    blksize = min(nvir_pair, max(ccsd.BLKMIN, blksize))
    chunks_vv = (int(min(blksize, 4e8 / blksize)), blksize)
    fswap.create_dataset("v", (nao_pair, nvir_pair), "f8", chunks=chunks_vv)
    for p0, p1 in lib.prange(0, nvir_pair, blksize):
        fswap["v"][:, p0:p1] = _trans(lib.unpack_tril(_cp(dvvvv[p0:p1])), (nocc, nmo, nocc, nmo)).T

    blksize = int(max_memory * 1e6 / 8 / (nao_pair + nmo**2))
    blksize = min(nao_pair, max(ccsd.BLKMIN, blksize))
    fswap.create_dataset("o", (nmo, nocc, nao_pair), "f8", chunks=(nocc, nocc, blksize))
    buf1 = np.zeros((nocc, nocc, nmo, nmo))
    buf1[:, :, :nocc, :nocc] = doooo
    buf1[:, :, nocc:, nocc:] = _cp(doovv)
    buf1 = _trans(buf1.reshape(nocc**2, -1), (0, nmo, 0, nmo))
    fswap["o"][:nocc] = buf1.reshape(nocc, nocc, nao_pair)
    dovoo = np.asarray(dooov).transpose(2, 3, 0, 1)
    for p0, p1 in lib.prange(nocc, nmo, nocc):
        buf1 = np.zeros((nocc, p1 - p0, nmo, nmo))
        buf1[:, :, :nocc, :nocc] = dovoo[:, p0 - nocc : p1 - nocc]
        buf1[:, :, nocc:, :nocc] = dovvo[:, p0 - nocc : p1 - nocc]
        buf1[:, :, :nocc, nocc:] = dovov[:, p0 - nocc : p1 - nocc]
        buf1[:, :, nocc:, nocc:] = dovvv[:, p0 - nocc : p1 - nocc]
        buf1 = buf1.transpose(1, 0, 3, 2).reshape((p1 - p0) * nocc, -1)
        buf1 = _trans(buf1, (0, nmo, 0, nmo))
        fswap["o"][p0:p1] = buf1.reshape(p1 - p0, nocc, nao_pair)
    dovoo = buf1 = None

    # transform dm2_kl then dm2 + dm2.transpose(2,3,0,1)
    gsave = fsave.create_dataset("dm2", (nao_pair, nao_pair), "f8", chunks=chunks_vv)
    for p0, p1 in lib.prange(0, nao_pair, blksize):
        buf1 = np.zeros((p1 - p0, nmo, nmo))
        buf1[:, nocc:, nocc:] = lib.unpack_tril(_cp(fswap["v"][p0:p1]))
        buf1[:, :, :nocc] = fswap["o"][:, :, p0:p1].transpose(2, 0, 1)
        buf2 = _trans(buf1, (0, nmo, 0, nmo))
        if p0 > 0:
            buf1 = _cp(gsave[:p0, p0:p1])
            buf1[:p0, : p1 - p0] += buf2[: p1 - p0, :p0].T
            buf2[: p1 - p0, :p0] = buf1[:p0, : p1 - p0].T
            gsave[:p0, p0:p1] = buf1
        lib.transpose_sum(buf2[:, p0:p1], inplace=True)
        gsave[p0:p1] = buf2
    if incore:
        return fsave["dm2"][:]
    else:
        return fsave


def _load_block_tril(h5dat, row0, row1, nao, out=None):
    nao_pair = nao * (nao + 1) // 2
    if out is None:
        out = np.ndarray((row1 - row0, nao, nao_pair))
    dat = h5dat[row0 * (row0 + 1) // 2 : row1 * (row1 + 1) // 2]
    p1 = 0
    for i in range(row0, row1):
        p0, p1 = p1, p1 + i + 1
        out[i - row0, : i + 1] = dat[p0:p1]
        for j in range(row0, i):
            out[j - row0, i] = out[i - row0, j]
    for i in range(row1, nao):
        i2 = i * (i + 1) // 2
        out[:, i] = h5dat[i2 + row0 : i2 + row1]
    return out


def _cp(a):
    return np.array(a, order="C")
