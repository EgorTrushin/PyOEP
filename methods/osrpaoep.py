#!/usr/bin/env python3

import scipy
import numpy as np
from pyscf import dft, gto, df, lib
from pyscf.gw.rpa import _get_scaled_legendre_roots
from .osexxoep import OSEXXOEP
from .rpaoep import RPAOEP


class OSRPAOEP(OSEXXOEP):
    r"""
    Implements self-consistent RPA optimized effective potential method for open-shell systems.
    The implementation follows:
    P. Bleiziffer, A. Heßelmann, A. Görling. J. Chem. Phys. 139, 084113 (2013).  https://doi.org/10.1063/1.4818984
    E. Trushin, S. Fauser, A. Mölkner, J. Erhard, A. Görling. Phys. Rev. Lett. 134, 016402 (2025) https://doi.org/10.1103/PhysRevLett.134.016402

    Args:
        mf: PySCF object with UHF or UKS calculation
        oep_basis: auxiliary basis to solve the EXX-OEP equation
        ri_basis: auxiliary RI basis for the RPA response function
        use_HOMO_condition: whether to use HOMO condition for the exchange potential
        vh_via_OEP: whether to construct AO Hartree potential via OEP basis
        space_sym: whether to perform space-symmetrization
        spin_sym: whether to use spin symmetrization (exchange part only)
    """

    def __init__(self, mf, oep_basis, ri_basis,
                 use_HOMO_condition=False, vh_via_OEP=False, space_sym=False, spin_sym=False):
        self.ri_basis = ri_basis
        self.E_corr = 0.0
        super().__init__(mf, oep_basis, use_HOMO_condition, vh_via_OEP, space_sym, spin_sym)
        self._init_ri_basis()

    # Reuse RI basis setup and helper functions from RPAOEP (identical logic)
    _init_ri_basis = RPAOEP._init_ri_basis
    _get_lambda_rpa = RPAOEP._get_lambda_rpa
    _get_gamma_rpa = RPAOEP._get_gamma_rpa

    def _get_W_ri_os(self, ints_3c_ri_mo_a, nelec_a, ints_3c_ri_mo_b, nelec_b, thr_fai_ri):
        """
        RI basis coupling filtering for open-shell: combines alpha and beta occ-virt blocks.
        Analogous to get_W_spin_sym but for the RI basis (no threshold on OEP side).
        """
        trans_mat_constraint = self.WII_ri @ self.W3_ri
        dmat = trans_mat_constraint.T @ ints_3c_ri_mo_a[:, nelec_a:, :nelec_a].reshape(self.naux_ri, -1)
        amat = dmat @ dmat.T
        if nelec_b > 0:
            dmat = trans_mat_constraint.T @ ints_3c_ri_mo_b[:, nelec_b:, :nelec_b].reshape(self.naux_ri, -1)
            amat += dmat @ dmat.T
        eigs, evecs = scipy.linalg.eigh(amat)
        nsing = (eigs < thr_fai_ri).sum()
        return trans_mat_constraint @ evecs[:, nsing:]

    def get_energies_and_potentials(self):
        """Determines energy contributions and potentials, including RPA correlation energy."""
        super().get_energies_and_potentials()
        self.e_tot += self.E_corr

    def run(self, maxit=50, thr_fai_oep=5e-2, thr_fai_ri=1e-8,
            nw=20, x0=2.5, linear_mixing=-1.0, e_conv_thr=1e-8):
        r"""
        Performs a self-consistent RPA-OEP calculation for open-shell systems.

        Args:
            maxit: maximal number of iterations
            thr_fai_oep: threshold T_{ai} for OEP basis (exchange potential)
            thr_fai_ri: threshold T_{ai} for RI basis (correlation response)
            nw: number of Gauss-Legendre frequency quadrature points
            x0: frequency scaling parameter for quadrature
            linear_mixing: if > 0, use linear mixing instead of DIIS
            e_conv_thr: threshold for energy convergence
        """
        if linear_mixing > 0:
            fock_old_a, fock_old_b = None, None
        else:
            adiis = lib.diis.DIIS()

        e_tot_old = None

        print("ITER" + " " * 8 + "ENERGY" + " " * 12 + "EDIFF" + " " * 13 + "E_CORR")
        for current_iter in range(maxit):

            ints_3c_a = self.mf.mo_coeff[0].T @ self.ints_3c_ao @ self.mf.mo_coeff[0]
            ints_3c_b = self.mf.mo_coeff[1].T @ self.ints_3c_ao @ self.mf.mo_coeff[1]
            z_a = z_b = None

            # --- Exchange OEP (same as OSEXXOEP) ---
            if self.vh_via_OEP or self.space_sym:
                self.get_vh_via_OEP(ints_3c_a, ints_3c_b, self.nelec)

            if self.use_HOMO_condition:
                if self.spin_sym:
                    z_a, zII_a = self.get_z_and_zII(ints_3c_a, self.mf.mo_energy[0], self.nelec[0])
                    vref_oep_a = self.get_v_ref_w_homo_spin_sym(
                        zII_a, ints_3c_a, ints_3c_b, self.mf.mo_coeff[0], self.nelec, self.vxnl_ao[0]
                    )
                    vref_oep_b = vref_oep_a.copy()
                    W3_a = self.get_W3_charge_and_homo(zII_a)
                    W_a = W_b = self.get_W_spin_sym(W3_a, ints_3c_a, ints_3c_b, self.nelec, thr_fai_oep)
                else:
                    z_a, zII_a = self.get_z_and_zII(ints_3c_a, self.mf.mo_energy[0], self.nelec[0])
                    z_b, zII_b = self.get_z_and_zII(ints_3c_b, self.mf.mo_energy[1], self.nelec[1])
                    vref_oep_a = self.get_v_ref_w_homo(
                        zII_a, ints_3c_a, self.mf.mo_coeff[0], self.nelec[0], self.vxnl_ao[0], self.ip[0]
                    )
                    vref_oep_b = self.get_v_ref_w_homo(
                        zII_b, ints_3c_b, self.mf.mo_coeff[1], self.nelec[1], self.vxnl_ao[1], self.ip[1]
                    )
                    W3_a = self.get_W3_charge_and_homo(zII_a)
                    W3_b = self.get_W3_charge_and_homo(zII_b)
                    W_a = self.get_W(W3_a, ints_3c_a, self.nelec[0], thr_fai_oep)
                    W_b = self.get_W(W3_b, ints_3c_b, self.nelec[1], thr_fai_oep)
            else:
                W3 = self.get_W3_charge()
                if self.spin_sym:
                    vref_oep_a = self.get_v_ref_spin_sym(ints_3c_a, ints_3c_b, self.nelec)
                    vref_oep_b = vref_oep_a.copy()
                    W_a = W_b = self.get_W_spin_sym(W3, ints_3c_a, ints_3c_b, self.nelec, thr_fai_oep)
                else:
                    vref_oep_a = self.get_v_ref(ints_3c_a, self.nelec[0])
                    vref_oep_b = self.get_v_ref(ints_3c_b, self.nelec[1])
                    W_a = self.get_W(W3, ints_3c_a, self.nelec[0], thr_fai_oep)
                    W_b = self.get_W(W3, ints_3c_b, self.nelec[1], thr_fai_oep)

            X0_a = self.get_X0(ints_3c_a, self.mf.mo_energy[0], self.nelec[0], W_a)
            X0_b = self.get_X0(ints_3c_b, self.mf.mo_energy[1], self.nelec[1], W_b)

            vref_ao_a = np.einsum("ijk,k->ij", self.ints_3c_ao_t, vref_oep_a[:])
            vref_ao_b = np.einsum("ijk,k->ij", self.ints_3c_ao_t, vref_oep_b[:])
            rhs_a = self.get_rhs(
                vref_ao_a, self.vxnl_ao[0], ints_3c_a, self.mf.mo_coeff[0], self.mf.mo_energy[0], self.nelec[0], W_a
            )
            rhs_b = self.get_rhs(
                vref_ao_b, self.vxnl_ao[1], ints_3c_b, self.mf.mo_coeff[1], self.mf.mo_energy[1], self.nelec[1], W_b
            )

            if self.spin_sym:
                rhs_a = scipy.linalg.solve(X0_a + X0_b, rhs_a + rhs_b)
                vrest_oep_a = vrest_oep_b = W_a @ rhs_a
            else:
                rhs_a = scipy.linalg.solve(X0_a, rhs_a)
                rhs_b = scipy.linalg.solve(X0_b, rhs_b)
                vrest_oep_a = W_a @ rhs_a
                vrest_oep_b = W_b @ rhs_b

            vrest_ao_a = np.einsum("ijk,k->ij", self.ints_3c_ao_t, vrest_oep_a[:])
            vrest_ao_b = np.einsum("ijk,k->ij", self.ints_3c_ao_t, vrest_oep_b[:])

            self.potentials_test(
                vrest_oep_a, vref_oep_a, vrest_ao_a, vref_ao_a,
                self.vxnl_ao[0], self.mf.mo_coeff[0], self.nelec[0], z_a,
                self.ip[0] if self.use_HOMO_condition else None,
            )
            if not self.spin_sym:
                self.potentials_test(
                    vrest_oep_b, vref_oep_b, vrest_ao_b, vref_ao_b,
                    self.vxnl_ao[1], self.mf.mo_coeff[1], self.nelec[1], z_b,
                    self.ip[1] if self.use_HOMO_condition else None,
                )

            # --- Correlation OEP (RPA) ---
            # Always use charge-only W_c (no HOMO condition), matching Molpro behavior.
            W3_c = self.get_W3_charge()

            ints_3c_ri_mo_a = self.mf.mo_coeff[0].T @ self.ints_3c_ao_ri @ self.mf.mo_coeff[0]
            ints_3c_ri_mo_b = self.mf.mo_coeff[1].T @ self.ints_3c_ao_ri @ self.mf.mo_coeff[1]

            W_ri = self._get_W_ri_os(ints_3c_ri_mo_a, self.nelec[0],
                                     ints_3c_ri_mo_b, self.nelec[1], thr_fai_ri)
            naux_ri_c = W_ri.shape[1]

            ints_3c_ri_W_a = (W_ri.T @ ints_3c_ri_mo_a.reshape(self.naux_ri, -1)).reshape(naux_ri_c, self.nmo, self.nmo)
            ints_3c_ri_W_b = (W_ri.T @ ints_3c_ri_mo_b.reshape(self.naux_ri, -1)).reshape(naux_ri_c, self.nmo, self.nmo)

            if self.spin_sym:
                W_c = self.get_W_spin_sym(W3_c, ints_3c_a, ints_3c_b, self.nelec, thr_fai_oep)
                naux_c = W_c.shape[1]
                ints_3c_oep_W_a = (W_c.T @ ints_3c_a.reshape(self.naux, -1)).reshape(naux_c, self.nmo, self.nmo)
                ints_3c_oep_W_b = (W_c.T @ ints_3c_b.reshape(self.naux, -1)).reshape(naux_c, self.nmo, self.nmo)

                rhs_c_a, self.E_corr = self.get_oscrpa_rhs(
                    ints_3c_oep_W_a, ints_3c_ri_W_a, ints_3c_ri_W_b,
                    self.mf.mo_energy[0], self.nelec[0],
                    self.mf.mo_energy[1], self.nelec[1], nw, x0
                )
                rhs_c_b, _ = self.get_oscrpa_rhs(
                    ints_3c_oep_W_b, ints_3c_ri_W_b, ints_3c_ri_W_a,
                    self.mf.mo_energy[1], self.nelec[1],
                    self.mf.mo_energy[0], self.nelec[0], nw, x0
                )

                X0_c_a = 4.0 * self.get_X0(ints_3c_a, self.mf.mo_energy[0], self.nelec[0], W_c)
                X0_c_b = 4.0 * self.get_X0(ints_3c_b, self.mf.mo_energy[1], self.nelec[1], W_c)
                rhs_c = scipy.linalg.solve(X0_c_a + X0_c_b, -(rhs_c_a + rhs_c_b))
                vrest_c_oep_a = vrest_c_oep_b = W_c @ rhs_c
                vrest_c_ao_a = np.einsum("ijk,k->ij", self.ints_3c_ao_t, vrest_c_oep_a[:])
                vrest_c_ao_b = vrest_c_ao_a
            else:
                W_c_a = self.get_W(W3_c, ints_3c_a, self.nelec[0], thr_fai_oep)
                naux_c_a = W_c_a.shape[1]
                ints_3c_oep_W_a = (W_c_a.T @ ints_3c_a.reshape(self.naux, -1)).reshape(naux_c_a, self.nmo, self.nmo)

                if self.nelec[1] > 0:
                    W_c_b = self.get_W(W3_c, ints_3c_b, self.nelec[1], thr_fai_oep)
                    naux_c_b = W_c_b.shape[1]
                    ints_3c_oep_W_b = (W_c_b.T @ ints_3c_b.reshape(self.naux, -1)).reshape(naux_c_b, self.nmo, self.nmo)

                    rhs_c_a, self.E_corr = self.get_oscrpa_rhs(
                        ints_3c_oep_W_a, ints_3c_ri_W_a, ints_3c_ri_W_b,
                        self.mf.mo_energy[0], self.nelec[0],
                        self.mf.mo_energy[1], self.nelec[1], nw, x0
                    )
                    rhs_c_b, _ = self.get_oscrpa_rhs(
                        ints_3c_oep_W_b, ints_3c_ri_W_b, ints_3c_ri_W_a,
                        self.mf.mo_energy[1], self.nelec[1],
                        self.mf.mo_energy[0], self.nelec[0], nw, x0
                    )

                    X0_c_b = 4.0 * self.get_X0(ints_3c_b, self.mf.mo_energy[1], self.nelec[1], W_c_b)
                    rhs_c_b = scipy.linalg.solve(X0_c_b, -rhs_c_b)
                    vrest_c_oep_b = W_c_b @ rhs_c_b
                    vrest_c_ao_b = np.einsum("ijk,k->ij", self.ints_3c_ao_t, vrest_c_oep_b[:])
                else:
                    rhs_c_a, self.E_corr = self.get_oscrpa_rhs(
                        ints_3c_oep_W_a, ints_3c_ri_W_a, None,
                        self.mf.mo_energy[0], self.nelec[0],
                        None, 0, nw, x0
                    )
                    vrest_c_ao_b = np.zeros_like(ints_3c_a[0])
                    vrest_c_oep_b = np.zeros(self.naux)

                X0_c_a = 4.0 * self.get_X0(ints_3c_a, self.mf.mo_energy[0], self.nelec[0], W_c_a)
                rhs_c_a = scipy.linalg.solve(X0_c_a, -rhs_c_a)
                vrest_c_oep_a = W_c_a @ rhs_c_a
                vrest_c_ao_a = np.einsum("ijk,k->ij", self.ints_3c_ao_t, vrest_c_oep_a[:])

            # --- Fock matrices: h0 + vJ + vx_local + vc ---
            h1e = self.mf.get_hcore(self.mf.mol)
            F_a = h1e + self.vj_ao + vref_ao_a + vrest_ao_a + vrest_c_ao_a
            F_b = h1e + self.vj_ao + vref_ao_b + vrest_ao_b + vrest_c_ao_b

            if linear_mixing > 0:
                if fock_old_a is None:
                    fock_old_a = F_a.copy()
                    fock_old_b = F_b.copy()
                else:
                    F_a = (1.0 - linear_mixing) * F_a + linear_mixing * fock_old_a
                    F_b = (1.0 - linear_mixing) * F_b + linear_mixing * fock_old_b
                    fock_old_a = F_a.copy()
                    fock_old_b = F_b.copy()
            else:
                S = self.mf.get_ovlp()
                D_a = self.mf.mo_coeff[0][:, :self.nelec[0]] @ self.mf.mo_coeff[0][:, :self.nelec[0]].T
                D_b = self.mf.mo_coeff[1][:, :self.nelec[1]] @ self.mf.mo_coeff[1][:, :self.nelec[1]].T
                e_a = F_a @ D_a @ S - S @ D_a @ F_a
                e_b = F_b @ D_b @ S - S @ D_b @ F_b
                nao = F_a.shape[0]
                F_combined = adiis.update(
                    np.concatenate([F_a.ravel(), F_b.ravel()]),
                    xerr=np.concatenate([e_a.ravel(), e_b.ravel()])
                )
                F_a = F_combined[:nao * nao].reshape(nao, nao)
                F_b = F_combined[nao * nao:].reshape(nao, nao)

            S = self.mf.get_ovlp()
            mo_energy, mo_coeff = scipy.linalg.eigh(F_a, S)
            self.mf.mo_energy[0], self.mf.mo_coeff[0] = mo_energy, mo_coeff
            mo_energy, mo_coeff = scipy.linalg.eigh(F_b, S)
            self.mf.mo_energy[1], self.mf.mo_coeff[1] = mo_energy, mo_coeff

            self.get_energies_and_potentials()

            if e_tot_old is None:
                print(f"{current_iter:3}  {self.e_tot:18.12f}  {'':18}  {self.E_corr:14.8f}")
                e_tot_old = self.e_tot
            else:
                ediff = self.e_tot - e_tot_old
                print(f"{current_iter:3}  {self.e_tot:18.12f}  {ediff:18.12f}  {self.E_corr:14.8f}")
                if abs(e_tot_old - self.e_tot) < e_conv_thr:
                    print("SCF converged")
                    self.converged = True
                    self.vref_oep_a = vref_oep_a
                    self.vref_oep_b = vref_oep_b
                    self.vrest_oep_a = vrest_oep_a
                    self.vrest_oep_b = vrest_oep_b
                    self.vrest_c_oep_a = vrest_c_oep_a
                    self.vrest_c_oep_b = vrest_c_oep_b
                    break
                e_tot_old = self.e_tot

            if current_iter == maxit - 1:
                print("SCF was not converged")
                self.vref_oep_a = vref_oep_a
                self.vref_oep_b = vref_oep_b
                self.vrest_oep_a = vrest_oep_a
                self.vrest_oep_b = vrest_oep_b
                self.vrest_c_oep_a = vrest_c_oep_a
                self.vrest_c_oep_b = vrest_c_oep_b

    def get_oscrpa_rhs(self, ints_3c_oep, ints_3c_ri_1, ints_3c_ri_2,
                       mo_energy_1, nelec_1, mo_energy_2, nelec_2, nw=20, x0=2.5):
        r"""
        Computes the open-shell RPA-OEP right-hand side and RPA correlation energy.

        The X0(omega) response matrix is averaged over alpha and beta spin channels.
        The RHS contributions (Rii, Ria, Yia) use spin-1 orbitals only, matching the
        Fortran uscrpa_rhs where int_fst_oep and int_fst_ri are built from orb1.

        Args:
            ints_3c_oep: (naux_c, nmo_1, nmo_1) W-filtered OEP integrals for spin 1
            ints_3c_ri_1: (naux_ri_c, nmo_1, nmo_1) W_ri-filtered RI integrals for spin 1
            ints_3c_ri_2: (naux_ri_c, nmo_2, nmo_2) W_ri-filtered RI integrals for spin 2,
                          or None if nelec_2 == 0
            mo_energy_1: (nmo_1,) orbital energies for spin 1
            nelec_1: number of occupied orbitals for spin 1
            mo_energy_2: (nmo_2,) orbital energies for spin 2, or None if nelec_2 == 0
            nelec_2: number of occupied orbitals for spin 2
            nw: number of Gauss-Legendre frequency quadrature points
            x0: frequency scaling parameter

        Returns:
            rhs: (naux_c,) RPA-OEP right-hand side for spin 1
            E_corr: RPA correlation energy
        """
        nmo_1 = len(mo_energy_1)
        nv_1 = nmo_1 - nelec_1
        naux_c = ints_3c_oep.shape[0]
        naux_ri_c = ints_3c_ri_1.shape[0]

        freqs, wts = _get_scaled_legendre_roots(nw, x0)

        fai_ri_1 = ints_3c_ri_1[:, nelec_1:, :nelec_1]  # (naux_ri_c, nv_1, no_1)
        N_diag = np.diagonal(ints_3c_oep, axis1=1, axis2=2)  # (naux_c, nmo_1)

        if nelec_2 > 0:
            fai_ri_2 = ints_3c_ri_2[:, nelec_2:, :nelec_2]  # (naux_ri_c, nv_2, no_2)

        # Precompute frequency-independent quantities for vectorized Rii/Ria
        fai_ri_1_flat = fai_ri_1.reshape(naux_ri_c, nv_1 * nelec_1)  # (naux_ri_c, nv_1*no_1)
        M_virt_T = ints_3c_ri_1[:, nelec_1:, :].reshape(naux_ri_c * nv_1, nmo_1).T  # (nmo_1, naux_ri_c*nv_1)
        M_occ_T  = ints_3c_ri_1[:, :nelec_1, :].reshape(naux_ri_c * nelec_1, nmo_1).T  # (nmo_1, naux_ri_c*no_1)
        oep_occ_flat  = ints_3c_oep[:, :nelec_1, :].transpose(0, 2, 1).reshape(naux_c, -1)
        oep_virt_flat = ints_3c_oep[:, nelec_1:, :].transpose(0, 2, 1).reshape(naux_c, -1)
        _d = mo_energy_1[:nelec_1][None, :] - mo_energy_1[:, None]  # (nmo_1, no_1)
        _m = np.abs(_d) > 1e-6
        inv_denom_i = np.where(_m, 1.0 / np.where(_m, _d, 1.0), 0.0)
        _d = mo_energy_1[nelec_1:][None, :] - mo_energy_1[:, None]  # (nmo_1, nv_1)
        _m = np.abs(_d) > 1e-6
        inv_denom_a = np.where(_m, 1.0 / np.where(_m, _d, 1.0), 0.0)
        if nelec_2 > 0:
            fai_ri_2_flat = fai_ri_2.reshape(naux_ri_c, -1)  # (naux_ri_c, nv_2*no_2)

        E_corr = 0.0
        rhs = np.zeros(naux_c)

        for omega, wt in zip(freqs, wts):

            # --- Build averaged X0(omega) in RI basis and diagonalize ---
            lam_1 = self._get_lambda_rpa(mo_energy_1, nelec_1, omega)
            X0_ri = fai_ri_1_flat @ (fai_ri_1_flat * lam_1.reshape(1, -1)).T
            if nelec_2 > 0:
                lam_2 = self._get_lambda_rpa(mo_energy_2, nelec_2, omega)
                X0_ri += fai_ri_2_flat @ (fai_ri_2_flat * lam_2.reshape(1, -1)).T
            X0_ri /= 2.0  # average over spin channels

            eval_n, evec = scipy.linalg.eigh(X0_ri)

            # --- Correlation energy integrand ---
            E_corr += wt * np.sum(np.log(1.0 + eval_n) - eval_n) / (2.0 * np.pi)

            # --- Non-linear screening factor and rotated integrals (spin 1 only) ---
            K_n = -eval_n / (eval_n + 1.0)

            # --- F_ai: rotate occ-virt RI integrals (spin 1) to eigenbasis ---
            F_ai = (evec.T @ fai_ri_1_flat).reshape(naux_ri_c, nv_1, nelec_1)  # (naux_ri_c, nv_1, no_1)

            lam_neg_1 = -lam_1
            gamma_1 = self._get_gamma_rpa(mo_energy_1, nelec_1, omega)

            # Kmat[m, m'] = sum_n K_n * evec[m, n] * evec[m', n]
            Kmat = (evec * K_n) @ evec.T  # (naux_ri_c, naux_ri_c)

            # R[m, a, i] = sum_{m'} Kmat[m, m'] * lam_neg_1[a, i] * fai_ri_1[m', a, i]
            R = (Kmat @ (lam_neg_1[None] * fai_ri_1).reshape(naux_ri_c, nv_1 * nelec_1)).reshape(
                naux_ri_c, nv_1, nelec_1)  # (naux_ri_c, nv_1, no_1)

            # --- Rii: t_i[s, i] = sum_{m,a} ints_3c_ri_1[m, nelec_1+a, s] * R[m, a, i] ---
            T_si = (M_virt_T @ R.reshape(naux_ri_c * nv_1, nelec_1)) * inv_denom_i  # (nmo_1, no_1)
            rhs_om = 2.0 * (oep_occ_flat @ T_si.reshape(-1))

            # --- Ria: t_a[s, a] = sum_{m,i} ints_3c_ri_1[m, i, s] * R[m, a, i] ---
            T_sa = (M_occ_T @ R.transpose(0, 2, 1).reshape(naux_ri_c * nelec_1, nv_1)) * inv_denom_a  # (nmo_1, nv_1)
            rhs_om += 2.0 * (oep_virt_flat @ T_sa.reshape(-1))

            # --- Yia term ---
            Y_ai = gamma_1 * (K_n @ (F_ai ** 2).reshape(naux_ri_c, -1)).reshape(nv_1, nelec_1)
            rhs_om += N_diag[:, nelec_1:] @ Y_ai.sum(axis=1) - N_diag[:, :nelec_1] @ Y_ai.sum(axis=0)

            rhs += rhs_om * wt / (2.0 * np.pi)

        return rhs, E_corr
