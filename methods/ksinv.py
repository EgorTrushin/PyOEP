#!/usr/bin/env python3

import numpy as np
import scipy
from pyscf import dft, gto

from .exxoep import EXXOEP


class KSINV(EXXOEP):
    r"""
    Implements response function Kohn-Sham inversion method for closed-shell systems.
    The implementation follows:
    J. Erhard, E. Trushin, A. Görling. J. Chem. Phys. 156, 204124 (2022). https://doi.org/10.1063/5.0087356
    J. Erhard, E. Trushin, A. Görling. J. Chem. Phys. 162, 034116 (2025). https://doi.org/10.1063/5.0239422

    Args:
        mf: PySCF object with RHF or RKS calculation
        oep_basis: auxiliary basis to solve OEP equation
        dm_ref: reference (target) density matrix to use in inversion
        e_ref: reference total energy from calculation which provided dm_ref
        ip: ionization potential (IP), if provided HOMO condition to enforce this IP is employed
        vh_via_OEP: whether to construct AO Hartree potential via OEP basis
        space_sym: whether to perform space-symmetrization
    """

    def __init__(self, mf, oep_basis, dm_ref, e_ref, ip=None, vh_via_OEP=False, space_sym=False):
        self.dm_ref = dm_ref.copy()
        self.e_ref = e_ref
        self.ip = ip
        self.use_HOMO_condition = ip is not None
        self.nelec = sum(i != 0 for i in mf.mo_occ)
        self.init_common(mf, oep_basis, vh_via_OEP, space_sym)
        self.get_inverse_WII()

    def run(self, maxit=100, thr_fai_oep=5e-2, conv_thr=1e-10, mixing_a=0.1, mixing_i=3):
        r"""
        Performs a KS inversion calculation.

        Args:
            maxit: maximal number of iterations
            thr_fai_oep: threshold T_{ai} from Section IIA5 of J. Chem. Phys. 155 (2021) 054109
            conv_thr: threshold for convergence
            mixing_a and mixing_i: parameters to determine mixing value at given iteration
                                   see Appendix A in J. Chem. Phys. 156, 204124 (2022)
        """

        vrest_oep = None
        vrest_ao = None
        delta_rhs_i = None

        h1e = self.mf.get_hcore()
        e1 = np.einsum("ij,ji->", h1e, self.dm_ref)

        e_kin_ref = np.einsum("ij,ji->", self.mf.mol.intor_symmetric("int1e_kin"), self.dm_ref)
        e_ext_ref = np.einsum("ij,ji->", self.mf.mol.intor_symmetric("int1e_nuc"), self.dm_ref)

        vj_ao_ref = self.mf.get_j(dm=self.dm_ref)
        e_coul_ref = np.einsum("ij,ji->", vj_ao_ref, self.dm_ref) * 0.5
        e_ee_ref = self.e_ref - e1 - self.mf.energy_nuc()

        aux = self.dm_ref @ self.ints_3c_ao
        u_ref = np.trace(aux, axis1=1, axis2=2)
        vc2 = self.WII.T @ u_ref
        vc = self.WII @ vc2
        vref_oep = -1.0 / np.dot(self.y, vc) * vc
        vref_ao = np.einsum("ijk,k->ij", self.ints_3c_ao_t, vref_oep[:])

        F = h1e + vj_ao_ref + vref_ao
        S = self.mf.get_ovlp()
        self.mf.mo_energy, self.mf.mo_coeff = self.mf._eigh(F, S)

        print("ITER" + " " * 8 + "ENERGY" + " " * 12 + "RHS" + " " * 8 + "MIXING")
        for current_iter in range(maxit):
            dm = self.mf.make_rdm1()
            e1 = np.einsum("ij,ji->", h1e, dm)

            vj_ao, vxnl_ao = self.mf.get_jk(dm=np.asarray(dm))
            e_coul = np.einsum("ij,ji->", vj_ao, dm) * 0.5
            e_x = -0.5 * np.einsum("ij,ji->", vxnl_ao, dm) * 0.5
            e_ext = np.einsum("ij,ji->", self.mf.mol.intor_symmetric("int1e_nuc"), dm)
            e_kin = np.einsum("ij,ji->", self.mf.mol.intor_symmetric("int1e_kin"), dm)

            t_c = e_kin_ref - e_kin
            v_c = e_ee_ref - e_coul - e_x
            e_c = t_c + v_c

            e_tot = e1 + e_coul + e_x + self.mf.energy_nuc() + e_c

            ints_3c = self.mf.mo_coeff.T @ self.ints_3c_ao @ self.mf.mo_coeff

            if self.vh_via_OEP or self.space_sym:
                vc2 = self.WII.T @ u_ref
                vc = self.WII @ vc2
                vaux = 2 * self.nelec / np.dot(self.y, vc) * vc
                vj_ao_ref = np.einsum("ijk,k->ij", self.ints_3c_ao_t, vaux[:])

            z = None
            if self.use_HOMO_condition:
                z, zII = self.get_z_and_zII(ints_3c, self.mf.mo_energy, self.nelec)
                vref_oep = self.get_v_ref_w_homo(u_ref, vj_ao_ref, zII, self.mf.mo_coeff, self.nelec, self.ip)
                vref_ao = np.einsum("ijk,k->ij", self.ints_3c_ao_t, vref_oep[:])
                W3 = self.get_W3_charge_and_homo(zII)
            else:
                W3 = self.get_W3_charge()

            W = self.get_W(W3, ints_3c, self.nelec, thr_fai_oep)
            W_inv = self.get_inverse_W(W3, ints_3c, self.nelec, thr_fai_oep)
            self.unity_test(W, W_inv)

            X0 = 4.0 * self.get_X0(ints_3c, self.mf.mo_energy, self.nelec, W)

            rhs = self.get_rhs_inv(W)
            delta_rhs = np.sqrt(np.dot(rhs, rhs))

            rhs = scipy.linalg.solve(X0, rhs)

            rhs_exx = self.get_rhs(vref_ao, -0.5 * vxnl_ao, ints_3c, self.mf.mo_coeff, self.mf.mo_energy, self.nelec, W)
            rhs_exx = scipy.linalg.solve(0.25 * X0, rhs_exx)
            self.vx_oep = W @ rhs_exx + vref_oep

            mixing, delta_rhs_i = self.get_mixing(current_iter, delta_rhs, mixing_a, mixing_i, delta_rhs_i)

            if vrest_oep is None:
                vrest_oep = mixing * W @ rhs
            else:
                vrest_oep += mixing * W @ rhs

            vrest_oep = W @ (W_inv @ vrest_oep)

            vrest_ao = np.einsum("ijk,k->ij", self.ints_3c_ao_t, vrest_oep[:])

            self.potentials_test(
                vrest_oep,
                vref_oep,
                vrest_ao,
                vref_ao,
                vj_ao_ref,
                self.mf.mo_coeff,
                self.nelec,
                z,
            )

            F = h1e + vj_ao_ref + vref_ao + vrest_ao
            self.mf.mo_energy, self.mf.mo_coeff = self.mf._eigh(F, S)

            print(f"{current_iter:3}  {e_tot:18.12f}    {delta_rhs:.3e}    {mixing:.3e}")
            if abs(delta_rhs) < conv_thr:
                print("KS inversion converged")
                self.converged = True
                print()
                print(f"Total energy:           {e_tot:18.12f}")
                print(f"One-electron energy:    {e1:18.12f}")
                print(f"Two-electron energy:    {e_coul + e_x:18.12f}")
                print(f"Nuclear energy:         {self.mf.energy_nuc():18.12f}")
                print(f"Coulomb energy:         {e_coul:18.12f}")
                print(f"Exchange energy:        {e_x:18.12f}")
                print(f"Correlation energy:     {e_c:18.12f}")
                print(f"Kinetic corr. energy:   {t_c:18.12f}")
                print(f"Potential corr. energy: {v_c:18.12f}")
                print()
                print(f"KS Hartree energy:      {e_coul:18.12f}")
                print(f"Ref Hartree energy:     {e_coul_ref:18.12f}")
                print(f"Hartree energy error:   {e_coul_ref - e_coul:18.12f}")
                print(f"KS external energy:     {e_ext:18.12f}")
                print(f"Ref external energy:    {e_ext_ref:18.12f}")
                print(f"External energy error:  {e_ext_ref - e_ext:18.12f}")
                print(f"KS kinetic energy:      {e_kin:18.12f}")
                F_s = (
                    2 * sum(self.mf.mo_energy[: self.nelec])
                    - np.trace(self.dm_ref @ self.mf.mol.intor_symmetric("int1e_nuc"))
                    - np.trace(self.dm_ref @ vj_ao_ref)
                    - np.trace(self.dm_ref @ (vref_ao + vrest_ao))
                )
                print(f"Lieb functional:        {F_s:18.12f}")
                print(f"Lieb error:             {F_s - e_kin:18.12f}")

                grid = dft.gen_grid.Grids(self.mf.mol)
                grid.build()
                ao = dft.numint.eval_ao(self.mf.mol, grid.coords)
                den_ref = dft.numint.eval_rho(self.mf.mol, ao, self.dm_ref)
                den_ks = dft.numint.eval_rho(self.mf.mol, ao, dm)
                density_error = np.dot(np.abs(den_ref - den_ks), grid.weights)
                print(f"Density error:          {density_error:18.12f}")

                self.vref_oep = vref_oep
                self.vrest_oep = vrest_oep
                self.vc_oep = vref_oep + vrest_oep - self.vx_oep
                break

            if current_iter == maxit - 1:
                print("KS inversion did not converge")
                self.vref_oep = vref_oep
                self.vrest_oep = vrest_oep
                self.vc_oep = vref_oep + vrest_oep - self.vx_oep

    def get_rhs_inv(self, trans_mat):
        """
        Constructs right-hand side of OEP equation according to
        Eqs. (46)-(48) in J. Chem. Phys. 156, 204124 (2022)
        """
        dm = self.mf.make_rdm1()
        delta_dm = self.dm_ref - dm
        rhs = trans_mat.T @ np.einsum("jk,ijk->i", delta_dm, self.ints_3c_ao)
        return rhs

    def get_inverse_WII(self):
        """
        Determine WII_inv inverse transformation matrix,
        see Section IIC2 in J. Chem. Phys. 155 (2021) 054109.
        """
        smat = self.auxmol.intor("int2c2e", aosym="s1", comp=1)
        diag = 1.0 / np.sqrt(np.diagonal(smat))
        diag_mat = np.diag(diag)
        inv_W1 = np.diag(1.0 / diag)  # inverse W1
        smat = (diag_mat @ smat) @ diag_mat
        eigs, evecs, _ = scipy.linalg.lapack.dsyev(smat)

        if self.space_sym:
            symvec_aux = gto.eval_gto(self.pmol, "GTOval_sph", self.grid.coords)
            symvec = symvec_aux.T @ self.grid.weights
            symvec[abs(symvec) > 1e-10] = 1.0
            eigs[np.abs(evecs.T @ symvec) < 1e-10] = 0.0
            mask = np.abs(eigs) > 1e-10
            smh = np.diag(np.sqrt(eigs[mask])) @ evecs[:, mask].T
        else:
            eigs = 1.0 / np.sqrt(eigs)
            smh = np.diag(1.0 / eigs) @ evecs.T  # inverse W2

        self.WII_inv = smh @ inv_W1

    def get_inverse_W(self, W3, ints_3c, nelec, thr_fai_oep):
        """
        Determine final inverse transformation matrix,
        see Section IIC2 in J. Chem. Phys. 155 (2021) 054109.
        """
        WIII_inv = W3.T @ self.WII_inv
        trans_mat_constraint = self.WII @ W3
        dmat = trans_mat_constraint.T @ ints_3c[:, nelec:, :nelec].reshape(self.naux, (self.nmo - nelec) * nelec)
        amat = dmat @ dmat.T
        eigs, evecs = scipy.linalg.eigh(amat)
        nsing = (eigs < thr_fai_oep).sum()
        trans_mat = evecs[:, nsing:].T @ WIII_inv
        return trans_mat

    def unity_test(self, W, W_inv):
        """Check whether W_inv is inverse of W."""
        aux = W_inv @ W
        if not np.allclose(aux, np.identity(aux.shape[0])):
            print("Identity check failed!")

    def get_mixing(self, curr_iter, delta_rhs, mixing_a, mixing_i, delta_rhs_i):
        """
        Determine mixing parameter according to Appendix A
        in J. Chem. Phys. 155 (2021) 054109.
        """
        if curr_iter < mixing_i:
            mixing = mixing_a
        else:
            if delta_rhs_i is None:
                delta_rhs_i = delta_rhs.copy()
            mixing = 2.0 / (1.0 + (2.0 / mixing_a - 1.0) ** (delta_rhs / delta_rhs_i))
        return mixing, delta_rhs_i

    def get_v_ref_w_homo(self, u, vj_ao, zII, mo_coeff, nelec, ip):
        r"""
        Constructs the Fermi-Amaldi reference potential with HOMO condition,
        see Section IIB5 in J. Chem. Phys. 162, 034116 (2025).
        """
        vcII = self.WII.T @ u
        aux_x = np.zeros([2, 2])
        aux_x[0, 0] = np.dot(self.yII, self.yII)
        aux_x[0, 1] = np.dot(self.yII, vcII)
        aux_x[1, 0] = np.dot(zII, self.yII)
        aux_x[1, 1] = np.dot(zII, vcII)
        aux_y = np.zeros([2])
        aux_y[0] = -1
        aux_y[1] = -ip - (mo_coeff.T @ (self.mf.get_hcore() + vj_ao) @ mo_coeff)[nelec - 1, nelec - 1]
        aux_sol = scipy.linalg.solve(aux_x, aux_y)
        vrefII = aux_sol[0] * self.yII + aux_sol[1] * vcII
        vref_oep = self.WII @ vrefII
        return vref_oep

    def potentials_test(self, vrest_oep, vref_oep, vrest_ao, vref_ao, vj_ao, mo_coeff, nelec, z):
        """Performs consistency checks for potentials."""
        if abs(np.dot(self.y, vrest_oep)) > 1e-12:
            print("Warning! y*vrest_oep =", np.dot(self.y, vrest_oep))
        if abs(np.dot(self.y, vref_oep) + 1.0) > 1e-12:
            print("Warning! y*vref_oep =", np.dot(self.y, vref_oep))
        if self.use_HOMO_condition:
            if abs(np.dot(z, vrest_oep)) > 1e-12:
                print("Warning! z*vrest_oep =", np.dot(z, vrest_oep))
            if abs((mo_coeff.T @ vrest_ao @ mo_coeff)[nelec - 1, nelec - 1]) > 1e-12:
                print("Warning!")
                print(f"v(HOMO) =  {(mo_coeff.T @ vrest_ao @ mo_coeff)[nelec - 1, nelec - 1]:.5f}  (VrestL)")
            v_aux = mo_coeff.T @ (self.mf.get_hcore() + vj_ao) @ mo_coeff
            if (
                abs(-v_aux[nelec - 1, nelec - 1] - self.ip - (mo_coeff.T @ vref_ao @ mo_coeff)[nelec - 1, nelec - 1])
                > 1e-12
            ):
                print("Warning!")
                print(f"v(HOMO) =  {-v_aux[nelec - 1, nelec - 1] - self.ip:.5f}  (VxNL)")
                print(f"v(HOMO) =  {(mo_coeff.T @ vref_ao @ mo_coeff)[nelec - 1, nelec - 1]:.5f}  (Vref)")
