#!/usr/bin/env python3

import numpy as np
import scipy
from pyscf import dft

from .ksinv import KSINV


class OSKSINV(KSINV):
    r"""
    Implements response function Kohn-Sham inversion method for open-shell systems.
    The implementation follows:
    J. Erhard, E. Trushin, A. Görling. J. Chem. Phys. 156, 204124 (2022). https://doi.org/10.1063/5.0087356
    J. Erhard, E. Trushin, A. Görling. J. Chem. Phys. 162, 034116 (2025). https://doi.org/10.1063/5.0239422

    Args:
        mf: PySCF object with RHF or RKS calculation
        oep_basis: auxiliary basis to solve OEP equation
        dm_ref: reference (target) density matrix to use in inversion
        e_ref: reference totat energy from calculation which provided dm_ref
        ip: ionization potential (IP), if provided HOMO condition to enforce this IP is employed
        vh_via_OEP: whether to construct AO Hartree potential via OEP basis
        space_sym: whether to perform space-symmetrization
        spin_sym: whether to use spin symmetrization
    """

    def __init__(self, mf, oep_basis, dm_ref, e_ref, ip=None, vh_via_OEP=False, space_sym=False, spin_sym=False):
        self.dm_ref = dm_ref.copy()
        self.e_ref = e_ref
        self.ip = ip
        self.use_HOMO_condition = ip is not None
        self.spin_sym = spin_sym
        self.nelec = mf.nelec
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

        vrest_oep_a, vref_oep_a = None, None
        vrest_oep_b, vref_oep_b = None, None
        vrest_ao_a, vref_ao_a = None, None
        vrest_ao_b, vref_ao_b = None, None
        delta_rhs_i_a, delta_rhs_i_b = None, None

        h1e = self.mf.get_hcore()
        e1 = np.einsum("ij,ji->", h1e, self.dm_ref[0])
        e1 += np.einsum("ij,ji->", h1e, self.dm_ref[1])

        e_kin_ref = np.einsum("ij,ji->", self.mf.mol.intor_symmetric("int1e_kin"), self.dm_ref[0])
        e_kin_ref += np.einsum("ij,ji->", self.mf.mol.intor_symmetric("int1e_kin"), self.dm_ref[1])
        e_ext_ref = np.einsum("ij,ji->", self.mf.mol.intor_symmetric("int1e_nuc"), self.dm_ref[0])
        e_ext_ref += np.einsum("ij,ji->", self.mf.mol.intor_symmetric("int1e_nuc"), self.dm_ref[1])

        vj_ao_ref, vxnl_ao_ref = self.mf.get_jk(dm=np.asarray(self.dm_ref))
        vj_ao_ref = vj_ao_ref[0] + vj_ao_ref[1]
        e_coul_ref = np.einsum("ij,ji->", vj_ao_ref, self.dm_ref[0] + self.dm_ref[1]) * 0.5
        e_x_ref = -0.5 * np.einsum("ij,ji->", vxnl_ao_ref[0], self.dm_ref[0])
        e_x_ref += -0.5 * np.einsum("ij,ji->", vxnl_ao_ref[1], self.dm_ref[1])
        e_ee_ref = self.e_ref - e1 - self.mf.energy_nuc()

        aux = self.dm_ref[0] @ self.ints_3c_ao
        u_ref_a = np.trace(aux, axis1=1, axis2=2)
        aux = self.dm_ref[1] @ self.ints_3c_ao
        u_ref_b = np.trace(aux, axis1=1, axis2=2)
        if self.spin_sym:
            u_ref = 0.5 * (u_ref_a + u_ref_b)
            vc2 = self.WII.T @ u_ref
            vc = self.WII @ vc2
            vref_oep_a = -1.0 / np.dot(self.y, vc) * vc
            vref_oep_b = vref_oep_a.copy()
        else:
            vc2 = self.WII.T @ u_ref_a
            vc = self.WII @ vc2
            vref_oep_a = -1.0 / np.dot(self.y, vc) * vc
            vc2 = self.WII.T @ u_ref_b
            vc = self.WII @ vc2
            vref_oep_b = -1.0 / np.dot(self.y, vc) * vc
        vref_ao_a = np.einsum("ijk,k->ij", self.ints_3c_ao_t, vref_oep_a[:])
        vref_ao_b = np.einsum("ijk,k->ij", self.ints_3c_ao_t, vref_oep_b[:])

        h1e = self.mf.get_hcore()
        F_a = h1e + vj_ao_ref + vref_ao_a
        F_b = h1e + vj_ao_ref + vref_ao_b
        S = self.mf.get_ovlp()
        mo_energy, mo_coeff = self.mf._eigh(F_a, S)
        self.mf.mo_energy[0], self.mf.mo_coeff[0] = mo_energy, mo_coeff
        mo_energy, mo_coeff = self.mf._eigh(F_b, S)
        self.mf.mo_energy[1], self.mf.mo_coeff[1] = mo_energy, mo_coeff

        if self.spin_sym:
            print("ITER" + " " * 8 + "ENERGY" + " " * 12 + "RHS" + " " * 8 + "MIXING")
        else:
            print(
                "ITER"
                + " " * 8
                + "ENERGY"
                + " " * 11
                + "RHS_a"
                + " " * 7
                + "RHS_b"
                + " " * 6
                + "MIXING_a"
                + " " * 4
                + "MIXING_b"
            )
        for current_iter in range(maxit):
            z_a = z_b = None

            dm = self.mf.make_rdm1()
            e1 = np.einsum("ij,ji->", h1e, dm[0])
            e1 += np.einsum("ij,ji->", h1e, dm[1])

            e_kin = np.einsum("ij,ji->", self.mf.mol.intor_symmetric("int1e_kin"), dm[0])
            e_kin += np.einsum("ij,ji->", self.mf.mol.intor_symmetric("int1e_kin"), dm[1])
            e_ext = np.einsum("ij,ji->", self.mf.mol.intor_symmetric("int1e_nuc"), dm[0])
            e_ext += np.einsum("ij,ji->", self.mf.mol.intor_symmetric("int1e_nuc"), dm[1])

            vj_ao, vxnl_ao = self.mf.get_jk(dm=np.asarray(dm))
            vj_ao = vj_ao[0] + vj_ao[1]
            e_coul = np.einsum("ij,ji->", vj_ao, dm[0] + dm[1]) * 0.5
            e_x = -0.5 * np.einsum("ij,ji->", vxnl_ao[0], dm[0])
            e_x += -0.5 * np.einsum("ij,ji->", vxnl_ao[1], dm[1])

            t_c = e_kin_ref - e_kin
            v_c = e_ee_ref - e_coul - e_x
            e_c = t_c + v_c

            e_tot = e1 + e_coul + e_x + self.mf.energy_nuc() + e_c

            ints_3c_a = self.mf.mo_coeff[0].T @ self.ints_3c_ao @ self.mf.mo_coeff[0]
            ints_3c_b = self.mf.mo_coeff[1].T @ self.ints_3c_ao @ self.mf.mo_coeff[1]

            if self.vh_via_OEP or self.space_sym:
                vc2 = self.WII.T @ (u_ref_a + u_ref_b)
                vc = self.WII @ vc2
                vaux = sum(self.nelec) / np.dot(self.y, vc) * vc
                vj_ao_ref = np.einsum("ijk,k->ij", self.ints_3c_ao_t, vaux[:])

            if self.use_HOMO_condition:
                z_a, zII_a = self.get_z_and_zII(ints_3c_a, self.mf.mo_energy[0], self.nelec[0])
                z_b, zII_b = self.get_z_and_zII(ints_3c_b, self.mf.mo_energy[1], self.nelec[1])
                if self.spin_sym:
                    u_ref_avg = 0.5 * (u_ref_a + u_ref_b)
                    vref_oep_a = self.get_v_ref_w_homo(
                        u_ref_avg, vj_ao_ref, zII_a, self.mf.mo_coeff[0], self.nelec[0], self.ip[0]
                    )
                    vref_oep_b = vref_oep_a.copy()
                    vref_ao_a = np.einsum("ijk,k->ij", self.ints_3c_ao_t, vref_oep_a[:])
                    vref_ao_b = vref_ao_a.copy()
                    W3_a = self.get_W3_charge_and_homo(zII_a)
                    W_a = W_b = self.get_W_spin_sym(W3_a, ints_3c_a, ints_3c_b, self.nelec, thr_fai_oep)
                    W_inv_a = W_inv_b = self.get_inverse_W_spin_sym(W3_a, ints_3c_a, ints_3c_b, self.nelec, thr_fai_oep)
                else:
                    vref_oep_a = self.get_v_ref_w_homo(
                        u_ref_a, vj_ao_ref, zII_a, self.mf.mo_coeff[0], self.nelec[0], self.ip[0]
                    )
                    vref_ao_a = np.einsum("ijk,k->ij", self.ints_3c_ao_t, vref_oep_a[:])
                    W3_a = self.get_W3_charge_and_homo(zII_a)
                    vref_oep_b = self.get_v_ref_w_homo(
                        u_ref_b, vj_ao_ref, zII_b, self.mf.mo_coeff[1], self.nelec[1], self.ip[1]
                    )
                    vref_ao_b = np.einsum("ijk,k->ij", self.ints_3c_ao_t, vref_oep_b[:])
                    W3_b = self.get_W3_charge_and_homo(zII_b)
                    W_a = self.get_W(W3_a, ints_3c_a, self.nelec[0], thr_fai_oep)
                    W_b = self.get_W(W3_b, ints_3c_b, self.nelec[1], thr_fai_oep)
                    W_inv_a = self.get_inverse_W(W3_a, ints_3c_a, self.nelec[0], thr_fai_oep)
                    W_inv_b = self.get_inverse_W(W3_b, ints_3c_b, self.nelec[1], thr_fai_oep)
            else:
                W3 = self.get_W3_charge()
                if self.spin_sym:
                    W_a = W_b = self.get_W_spin_sym(W3, ints_3c_a, ints_3c_b, self.nelec, thr_fai_oep)
                    W_inv_a = W_inv_b = self.get_inverse_W_spin_sym(W3, ints_3c_a, ints_3c_b, self.nelec, thr_fai_oep)
                else:
                    W_a = self.get_W(W3, ints_3c_a, self.nelec[0], thr_fai_oep)
                    W_b = self.get_W(W3, ints_3c_b, self.nelec[1], thr_fai_oep)
                    W_inv_a = self.get_inverse_W(W3, ints_3c_a, self.nelec[0], thr_fai_oep)
                    W_inv_b = self.get_inverse_W(W3, ints_3c_b, self.nelec[1], thr_fai_oep)

            self.unity_test(W_a, W_inv_a)
            self.unity_test(W_b, W_inv_b)

            X0_a = 2.0 * self.get_X0(ints_3c_a, self.mf.mo_energy[0], self.nelec[0], W_a)
            X0_b = 2.0 * self.get_X0(ints_3c_b, self.mf.mo_energy[1], self.nelec[1], W_b)

            rhs_a = W_a.T @ np.einsum("jk,ijk->i", self.dm_ref[0] - dm[0], self.ints_3c_ao)
            rhs_b = W_b.T @ np.einsum("jk,ijk->i", self.dm_ref[1] - dm[1], self.ints_3c_ao)

            if self.spin_sym:
                rhs_exx_a = self.get_rhs(
                    vref_ao_a, -vxnl_ao[0], ints_3c_a, self.mf.mo_coeff[0], self.mf.mo_energy[0], self.nelec[0], W_a
                )
                rhs_exx_b = self.get_rhs(
                    vref_ao_b, -vxnl_ao[1], ints_3c_b, self.mf.mo_coeff[1], self.mf.mo_energy[1], self.nelec[1], W_b
                )
                rhs_exx = scipy.linalg.solve(0.5 * (X0_a + X0_b), rhs_exx_a + rhs_exx_b)
                self.vx_oep_a = self.vx_oep_b = W_a @ rhs_exx + vref_oep_a
            else:
                rhs_exx_a = self.get_rhs(
                    vref_ao_a, -vxnl_ao[0], ints_3c_a, self.mf.mo_coeff[0], self.mf.mo_energy[0], self.nelec[0], W_a
                )
                rhs_exx_a = scipy.linalg.solve(0.5 * X0_a, rhs_exx_a)
                self.vx_oep_a = W_a @ rhs_exx_a + vref_oep_a
                rhs_exx_b = self.get_rhs(
                    vref_ao_b, -vxnl_ao[1], ints_3c_b, self.mf.mo_coeff[1], self.mf.mo_energy[1], self.nelec[1], W_b
                )
                rhs_exx_b = scipy.linalg.solve(0.5 * X0_b, rhs_exx_b)
                self.vx_oep_b = W_b @ rhs_exx_b + vref_oep_b

            if self.spin_sym:
                rhs = rhs_a + rhs_b
                delta_rhs = np.sqrt(np.dot(rhs, rhs))
                rhs = scipy.linalg.solve(X0_a + X0_b, rhs)
                mixing_alpha, delta_rhs_i_a = self.get_mixing(
                    current_iter, delta_rhs, mixing_a, mixing_i, delta_rhs_i_a
                )
                if vrest_oep_a is None:
                    vrest_oep_a = mixing_alpha * W_a @ rhs
                else:
                    vrest_oep_a += mixing_alpha * W_a @ rhs
                vrest_oep_a = W_a @ (W_inv_a @ vrest_oep_a)
                vrest_oep_b = vrest_oep_a
            else:
                delta_rhs_a = np.sqrt(np.dot(rhs_a, rhs_a))
                delta_rhs_b = np.sqrt(np.dot(rhs_b, rhs_b))
                rhs_a = scipy.linalg.solve(X0_a, rhs_a)
                rhs_b = scipy.linalg.solve(X0_b, rhs_b)
                mixing_alpha, delta_rhs_i_a = self.get_mixing(
                    current_iter, delta_rhs_a, mixing_a, mixing_i, delta_rhs_i_a
                )
                mixing_beta, delta_rhs_i_b = self.get_mixing(
                    current_iter, delta_rhs_b, mixing_a, mixing_i, delta_rhs_i_b
                )
                if vrest_oep_a is None:
                    vrest_oep_a = mixing_alpha * W_a @ rhs_a
                    vrest_oep_b = mixing_beta * W_b @ rhs_b
                else:
                    vrest_oep_a += mixing_alpha * W_a @ rhs_a
                    vrest_oep_b += mixing_beta * W_b @ rhs_b
                vrest_oep_a = W_a @ (W_inv_a @ vrest_oep_a)
                vrest_oep_b = W_b @ (W_inv_b @ vrest_oep_b)

            vrest_ao_a = np.einsum("ijk,k->ij", self.ints_3c_ao_t, vrest_oep_a[:])
            vrest_ao_b = np.einsum("ijk,k->ij", self.ints_3c_ao_t, vrest_oep_b[:])

            self.potentials_test(
                vrest_oep_a,
                vref_oep_a,
                vrest_ao_a,
                vref_ao_a,
                vj_ao_ref,
                self.mf.mo_coeff[0],
                self.nelec[0],
                z_a,
                self.ip[0] if self.use_HOMO_condition else None,
            )
            if not self.spin_sym:
                self.potentials_test(
                    vrest_oep_b,
                    vref_oep_b,
                    vrest_ao_b,
                    vref_ao_b,
                    vj_ao_ref,
                    self.mf.mo_coeff[1],
                    self.nelec[1],
                    z_b,
                    self.ip[1] if self.use_HOMO_condition else None,
                )

            F_a = h1e + vj_ao_ref + vref_ao_a + vrest_ao_a
            F_b = h1e + vj_ao_ref + vref_ao_b + vrest_ao_b
            mo_energy, mo_coeff = self.mf._eigh(F_a, S)
            self.mf.mo_energy[0], self.mf.mo_coeff[0] = mo_energy, mo_coeff
            mo_energy, mo_coeff = self.mf._eigh(F_b, S)
            self.mf.mo_energy[1], self.mf.mo_coeff[1] = mo_energy, mo_coeff

            if self.spin_sym:
                print(f"{current_iter:3}  {e_tot:18.12f}    {delta_rhs:.3e}   {mixing_alpha:.3e}")
            else:
                print(
                    f"{current_iter:3}  {e_tot:18.12f}    {delta_rhs_a:.3e}   {delta_rhs_b:.3e}   "
                    f"{mixing_alpha:.3e}   {mixing_beta:.3e}"
                )
            converged = (
                (abs(delta_rhs) < conv_thr)
                if self.spin_sym
                else (abs(delta_rhs_a) < conv_thr and abs(delta_rhs_b) < conv_thr)
            )
            if converged:
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
                    sum(self.mf.mo_energy[0, : self.nelec[0]])
                    + sum(self.mf.mo_energy[1, : self.nelec[1]])
                    - np.trace((self.dm_ref[0] + self.dm_ref[1]) @ self.mf.mol.intor_symmetric("int1e_nuc"))
                    - np.trace((self.dm_ref[0] + self.dm_ref[1]) @ vj_ao_ref)
                    - np.trace(self.dm_ref[0] @ (vref_ao_a + vrest_ao_a))
                    - np.trace(self.dm_ref[1] @ (vref_ao_b + vrest_ao_b))
                )
                print(f"Lieb functional:        {F_s:18.12f}")
                print(f"Lieb error:             {F_s - e_kin:18.12f}")

                if not self.space_sym:
                    grid = dft.gen_grid.Grids(self.mf.mol)
                    grid.build()
                    ao = dft.numint.eval_ao(self.mf.mol, grid.coords)
                    den_ref = dft.numint.eval_rho(self.mf.mol, ao, self.dm_ref[0] + self.dm_ref[1])
                    den_ks = dft.numint.eval_rho(self.mf.mol, ao, dm[0] + dm[1])
                    density_error = np.dot(np.abs(den_ref - den_ks), grid.weights)
                    print(f"Density error:          {density_error:18.12f}")

                self.vref_oep_a = vref_oep_a
                self.vrest_oep_a = vrest_oep_a
                self.vref_oep_b = vref_oep_b
                self.vrest_oep_b = vrest_oep_b
                self.vc_oep_a = vref_oep_a + vrest_oep_a - self.vx_oep_a
                self.vc_oep_b = vref_oep_b + vrest_oep_b - self.vx_oep_b
                break

            if current_iter == maxit - 1:
                print("KS inversion did not converge")
                self.vref_oep_a = vref_oep_a
                self.vrest_oep_a = vrest_oep_a
                self.vref_oep_b = vref_oep_b
                self.vrest_oep_b = vrest_oep_b
                self.vc_oep_a = vref_oep_a + vrest_oep_a - self.vx_oep_a
                self.vc_oep_b = vref_oep_b + vrest_oep_b - self.vx_oep_b

    def get_inverse_W_spin_sym(self, W3, ints_3c_a, ints_3c_b, nelec, thr_fai_oep):
        r"""
        Determine final inverse transformation matrix for spin-symmetrized case.
        See Sections IIB in J. Chem. Phys. 159, 244109 (2023)
        """
        WIII_inv = W3.T @ self.WII_inv
        trans_mat_constraint = self.WII @ W3
        dmat_a = trans_mat_constraint.T @ ints_3c_a[:, nelec[0] :, : nelec[0]].reshape(
            self.naux, (self.nmo - nelec[0]) * nelec[0]
        )
        amat_a = dmat_a @ dmat_a.T
        dmat_b = trans_mat_constraint.T @ ints_3c_b[:, nelec[1] :, : nelec[1]].reshape(
            self.naux, (self.nmo - nelec[1]) * nelec[1]
        )
        amat_b = dmat_b @ dmat_b.T
        amat = 0.5 * (amat_a + amat_b)
        aux = scipy.linalg.lapack.dsyev(amat)
        eigs = aux[0]
        evecs = aux[1]
        nsing = (eigs < thr_fai_oep).sum()
        trans_mat = evecs[:, nsing:].T @ WIII_inv
        return trans_mat

    def potentials_test(self, vrest_oep, vref_oep, vrest_ao, vref_ao, vj_ao, mo_coeff, nelec, z, ip=None):
        """Performs consistency checks for potentials.
        ip is the ionization potential for the spin channel being checked; pass None when use_HOMO_condition is False.
        """
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
            if abs(-v_aux[nelec - 1, nelec - 1] - ip - (mo_coeff.T @ vref_ao @ mo_coeff)[nelec - 1, nelec - 1]) > 1e-12:
                print("Warning!")
                print(f"v(HOMO) =  {-v_aux[nelec - 1, nelec - 1] - ip:.5f}  (VxNL)")
                print(f"v(HOMO) =  {(mo_coeff.T @ vref_ao @ mo_coeff)[nelec - 1, nelec - 1]:.5f}  (Vref)")
