#!/usr/bin/env python3

import numpy as np
import scipy
from pyscf import df, dft, gto
from pyscf.gw.rpa import _get_scaled_legendre_roots
from pyscf.scf.diis import CDIIS

from .exxoep import EXXOEP


class RPAOEP(EXXOEP):
    r"""
    Implements self-consistent RPA optimized effective potential method for closed-shell systems.
    The implementation follows:
    P. Bleiziffer, A. Heßelmann, A. Görling. J. Chem. Phys. 139, 084113 (2013).  https://doi.org/10.1063/1.4818984
    E. Trushin, S. Fauser, A. Mölkner, J. Erhard, A. Görling. Phys. Rev. Lett. 134, 016402 (2025) https://doi.org/10.1103/PhysRevLett.134.016402

    Args:
        mf: PySCF object with RHF or RKS calculation
        oep_basis: auxiliary basis to solve the EXX-OEP equation
        ri_basis: auxiliary RI basis for the RPA response function
        use_HOMO_condition: whether to use HOMO condition for the exchange potential
        vh_via_OEP: whether to construct AO Hartree potential via OEP basis
        space_sym: whether to perform space-symmetrization
    """

    def __init__(self, mf, oep_basis, ri_basis, use_HOMO_condition=False, vh_via_OEP=False, space_sym=False):
        self.ri_basis = ri_basis
        self.E_corr = 0.0
        super().__init__(mf, oep_basis, use_HOMO_condition, vh_via_OEP, space_sym)
        self._init_ri_basis()

    def _init_ri_basis(self):
        """
        Builds the RI auxiliary mol, 3-center integrals, and RI basis preprocessing.
        Computes WII_ri (overlap preprocessing) and W3_ri (charge constraint projection),
        analogous to get_WII and get_W3_charge for the OEP basis.
        The coupling filtering step (analogous to get_W) is done per SCF iteration
        inside run() since it depends on the current MOs.
        """
        self.auxmol_ri = df.addons.make_auxmol(self.mf.mol, self.ri_basis)
        self.ints_3c_ao_ri = df.incore.aux_e2(self.mf.mol, self.auxmol_ri, intor="int3c2e").transpose(
            2, 1, 0
        )  # (naux_ri, nao, nao)
        self.naux_ri = self.ints_3c_ao_ri.shape[0]

        # Overlap preprocessing for RI basis (analogous to get_WII)
        smat = self.auxmol_ri.intor("int2c2e", aosym="s1", comp=1)
        diag = 1.0 / np.sqrt(np.diagonal(smat))
        smat = np.diag(diag) @ smat @ np.diag(diag)
        eigs, evecs, _ = scipy.linalg.lapack.dsyev(smat)
        self.WII_ri = np.diag(diag) @ evecs @ np.diag(1.0 / np.sqrt(eigs))

        # Charge constraint for RI basis (analogous to get_y_and_yII + get_W3_charge)
        grid_ri = dft.gen_grid.Grids(self.auxmol_ri)
        grid_ri.verbose = 0
        grid_ri.build()
        y_ri = gto.eval_gto(self.auxmol_ri, "GTOval", grid_ri.coords).T @ grid_ri.weights
        yII_ri = self.WII_ri.T @ y_ri
        charge_norm_ri = np.dot(yII_ri, yII_ri)

        auxmat = np.eye(yII_ri.shape[0]) - np.outer(yII_ri, yII_ri) / charge_norm_ri
        auxmat2 = auxmat.T @ auxmat + np.eye(self.WII_ri.shape[1])
        _, evecs_W3 = scipy.linalg.eigh(auxmat2)
        self.W3_ri = evecs_W3[:, 1:]

    def _get_W_ri(self, ints_3c_ri_mo, nelec, thr_fai_ri):
        """
        RI basis coupling filtering: analogous to get_W but for the RI basis.
        Uses a much smaller threshold (thr_fai_ri ~ 1e-8) so almost no vectors
        are removed; the main purpose is numerical stability.
        """
        trans_mat_constraint = self.WII_ri @ self.W3_ri
        dmat = trans_mat_constraint.T @ ints_3c_ri_mo[:, nelec:, :nelec].reshape(
            self.naux_ri, (self.nmo - nelec) * nelec
        )
        amat = dmat @ dmat.T
        eigs, evecs = scipy.linalg.eigh(amat)
        nsing = (eigs < thr_fai_ri).sum()
        return trans_mat_constraint @ evecs[:, nsing:]

    def get_energies_and_potentials(self):
        """Determines energy contributions and potentials, including RPA correlation energy."""
        super().get_energies_and_potentials()
        self.e_tot += self.E_corr

    def get_vx_fock(self, vx_local_ao, vxnl_ao):
        """
        Exchange part of the Fock matrix: the local exchange potential vx_local_ao (vref + vrest).
        Hybrid variants override this to mix in the non-local exchange operator vxnl_ao.
        """
        return vx_local_ao

    def run(self, maxit=50, thr_fai_oep=5e-2, thr_fai_ri=1e-8, nw=20, x0=2.5, linear_mixing=-1.0, e_conv_thr=1e-8):
        r"""
        Performs a self-consistent RPA-OEP calculation.

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
            fock_old = None
        else:
            adiis = CDIIS()

        e_tot_old = None

        print("ITER" + " " * 8 + "ENERGY" + " " * 12 + "EDIFF" + " " * 13 + "E_CORR")
        for current_iter in range(maxit):
            ints_3c = self.mf.mo_coeff.T @ self.ints_3c_ao @ self.mf.mo_coeff
            ints_3c_ri_mo = self.mf.mo_coeff.T @ self.ints_3c_ao_ri @ self.mf.mo_coeff
            z = None

            # --- Exchange OEP (same as EXXOEP) ---
            if self.vh_via_OEP or self.space_sym:
                self.get_vh_via_OEP(ints_3c, self.nelec)

            if self.use_HOMO_condition:
                z, zII = self.get_z_and_zII(ints_3c, self.mf.mo_energy, self.nelec)
                vref_oep = self.get_v_ref_w_homo(zII, ints_3c, self.mf.mo_coeff, self.nelec, self.vxnl_ao)
                W3 = self.get_W3_charge_and_homo(zII)
            else:
                vref_oep = self.get_v_ref(ints_3c, self.nelec)
                W3 = self.get_W3_charge()

            W = self.get_W(W3, ints_3c, self.nelec, thr_fai_oep)
            X0_x = self.get_X0(ints_3c, self.mf.mo_energy, self.nelec, W)
            vref_ao = np.einsum("ijk,k->ij", self.ints_3c_ao_t, vref_oep[:])
            rhs_x = self.get_rhs(vref_ao, self.vxnl_ao, ints_3c, self.mf.mo_coeff, self.mf.mo_energy, self.nelec, W)
            rhs_x = scipy.linalg.solve(X0_x, rhs_x)
            vrest_oep = W @ rhs_x
            vrest_ao = np.einsum("ijk,k->ij", self.ints_3c_ao_t, vrest_oep[:])

            self.potentials_test(vrest_oep, vref_oep, vrest_ao, vref_ao, self.vxnl_ao, self.mf.mo_coeff, self.nelec, z)

            # --- Correlation OEP (RPA) ---
            # The correlation part always uses a charge-only W (no HOMO condition),
            # even when use_HOMO_condition=True for the exchange part.
            # This matches Molpro: when l_homo=True, basis_process_opm is called
            # a second time with l_homo=False before scrpa_rhs.
            # X0 for correlation uses RPA prefactor (4x compared to EXX).
            if self.use_HOMO_condition:
                W_c = self.get_W(self.get_W3_charge(), ints_3c, self.nelec, thr_fai_oep)
                X0_c = 4.0 * self.get_X0(ints_3c, self.mf.mo_energy, self.nelec, W_c)
            else:  # identical to the exchange part
                W_c = W
                X0_c = 4.0 * X0_x
            W_ri = self._get_W_ri(ints_3c_ri_mo, self.nelec, thr_fai_ri)
            naux_c = W_c.shape[1]
            naux_ri_c = W_ri.shape[1]

            # OEP integrals in W_c-filtered basis: (naux_c, nmo, nmo)
            ints_3c_oep_W = (W_c.T @ ints_3c.reshape(self.naux, -1)).reshape(naux_c, self.nmo, self.nmo)

            # RI integrals in W_ri-filtered basis: (naux_ri_c, nmo, nmo)
            ints_3c_ri_W = (W_ri.T @ ints_3c_ri_mo.reshape(self.naux_ri, -1)).reshape(naux_ri_c, self.nmo, self.nmo)

            [rhs_c], self.E_corr = self.get_rpa_rhs(
                [(ints_3c_oep_W, ints_3c_ri_W, self.mf.mo_energy, self.nelec)], nw, x0
            )

            # get_X0 uses e_ov = 1/(eps_i - eps_a) < 0, so X0_c is negative.
            # get_rpa_rhs returns a positive rhs (no e_ov factor), so negate
            # it here to match the Python sign convention (both sides of the
            # linear system must carry the same sign flip for the solve to cancel).
            rhs_c = scipy.linalg.solve(X0_c, -rhs_c)
            vrest_c_oep = W_c @ rhs_c
            vrest_c_ao = np.einsum("ijk,k->ij", self.ints_3c_ao_t, vrest_c_oep[:])

            # --- Fock matrix: h0 + vJ + vx + vc ---
            h1e = self.mf.get_hcore(self.mf.mol)
            F = h1e + self.vj_ao + self.get_vx_fock(vref_ao + vrest_ao, self.vxnl_ao) + vrest_c_ao

            if linear_mixing > 0:
                if fock_old is None:
                    fock_old = F.copy()
                else:
                    F = (1.0 - linear_mixing) * F + linear_mixing * fock_old
                    fock_old = F.copy()
            else:
                S = self.mf.get_ovlp()
                D = 2.0 * self.mf.mo_coeff[:, : self.nelec] @ self.mf.mo_coeff[:, : self.nelec].T
                F = adiis.update(S, D, F)

            S = self.mf.get_ovlp()
            self.mf.mo_energy, self.mf.mo_coeff = self.mf._eigh(F, S)

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
                    self.vref_oep = vref_oep
                    self.vrest_oep = vrest_oep
                    self.vrest_c_oep = vrest_c_oep
                    break
                e_tot_old = self.e_tot

            if current_iter == maxit - 1:
                print("SCF was not converged")
                self.vref_oep = vref_oep
                self.vrest_oep = vrest_oep
                self.vrest_c_oep = vrest_c_oep

    def get_rpa_rhs(self, spins, nw=20, x0=2.5):
        r"""
        Computes the RPA-OEP right-hand sides and RPA correlation energy via frequency quadrature.

        At each frequency omega, the RPA response matrix X0(omega) is built in the RI basis,
        averaged over the spin channels, and diagonalized once. The RHS of each spin channel
        receives three contributions per frequency point:
          - Rii: coupling through occupied orbital pairs
          - Ria: coupling through virtual orbital pairs
          - Yia: diagonal correction from orbital energy derivatives
        For open-shell systems, the RHS of a spin channel uses the orbitals of that channel only,
        matching the Fortran uscrpa_rhs.

        Args:
            spins: list with a tuple (ints_3c_oep, ints_3c_ri, mo_energy, nelec) for each spin channel,
                   one channel for closed-shell and two (alpha, beta) for open-shell systems:
                     ints_3c_oep: (naux_c, nmo, nmo) 3-center integrals in W-filtered OEP basis (MO),
                                  or None if the RHS of this channel is not needed
                     ints_3c_ri: (naux_ri_c, nmo, nmo) 3-center integrals in W_ri-filtered RI basis (MO)
                     mo_energy: (nmo,) orbital energies
                     nelec: number of occupied orbitals
            nw: number of Gauss-Legendre frequency quadrature points
            x0: frequency scaling parameter

        Returns:
            rhs: list with the (naux_c,) RPA-OEP right-hand side of each spin channel
                 (None if its ints_3c_oep is None, zero if it has no electrons)
            E_corr: RPA correlation energy
        """
        naux_ri_c = spins[0][1].shape[0]

        freqs, wts = _get_scaled_legendre_roots(nw, x0)

        # Occ-virt blocks of RI integrals of each channel: (naux_ri_c, nv*no)
        fai_ri_flat = [ints_3c_ri[:, nelec:, :nelec].reshape(naux_ri_c, -1) for _, ints_3c_ri, _, nelec in spins]

        # Precompute frequency-independent quantities for vectorized Rii/Ria of each channel
        rhs_data = []
        for ints_3c_oep, ints_3c_ri, mo_energy, nelec in spins:
            if ints_3c_oep is None or nelec == 0:
                rhs_data.append(None)
                continue
            nmo = len(mo_energy)
            nv = nmo - nelec
            naux_c = ints_3c_oep.shape[0]
            # Occ-virt block of RI integrals: (naux_ri_c, nv, no)
            fai_ri = ints_3c_ri[:, nelec:, :nelec]
            # Diagonal OEP integrals N_diag[nu, s] = ints_3c_oep[nu, s, s], used in Yia term
            N_diag = np.diagonal(ints_3c_oep, axis1=1, axis2=2)  # (naux_c, nmo)
            M_virt_T = ints_3c_ri[:, nelec:, :].reshape(naux_ri_c * nv, nmo).T  # (nmo, naux_ri_c*nv)
            M_occ_T = ints_3c_ri[:, :nelec, :].reshape(naux_ri_c * nelec, nmo).T  # (nmo, naux_ri_c*no)
            oep_occ_flat = ints_3c_oep[:, :nelec, :].transpose(0, 2, 1).reshape(naux_c, -1)  # (naux_c, nmo*no)
            oep_virt_flat = ints_3c_oep[:, nelec:, :].transpose(0, 2, 1).reshape(naux_c, -1)  # (naux_c, nmo*nv)
            _d = mo_energy[:nelec][None, :] - mo_energy[:, None]  # (nmo, no)
            _m = np.abs(_d) > 1e-6
            inv_denom_i = np.where(_m, 1.0 / np.where(_m, _d, 1.0), 0.0)  # (nmo, no)
            _d = mo_energy[nelec:][None, :] - mo_energy[:, None]  # (nmo, nv)
            _m = np.abs(_d) > 1e-6
            inv_denom_a = np.where(_m, 1.0 / np.where(_m, _d, 1.0), 0.0)  # (nmo, nv)
            rhs_data.append((fai_ri, N_diag, M_virt_T, M_occ_T, oep_occ_flat, oep_virt_flat, inv_denom_i, inv_denom_a))

        E_corr = 0.0
        rhs = [None if ints_3c_oep is None else np.zeros(ints_3c_oep.shape[0]) for ints_3c_oep, *_ in spins]

        for omega, wt in zip(freqs, wts, strict=True):
            # --- Build X0(omega) in RI basis, averaged over spin channels, and diagonalize ---
            lam = [self._get_lambda_rpa(mo_energy, nelec, omega) for _, _, mo_energy, nelec in spins]  # (nv, no)
            X0_ri = np.zeros((naux_ri_c, naux_ri_c))
            for (*_, nelec), fai_flat, lam_s in zip(spins, fai_ri_flat, lam, strict=True):
                if nelec > 0:
                    X0_ri += fai_flat @ (fai_flat * lam_s.reshape(1, -1)).T
            X0_ri /= len(spins)
            eval_n, evec = scipy.linalg.eigh(X0_ri)  # (naux_ri_c,), (naux_ri_c, naux_ri_c)

            # --- Correlation energy integrand: sum_n [ln(1+lambda_n) - lambda_n] ---
            E_corr += wt * np.sum(np.log(1.0 + eval_n) - eval_n) / (2.0 * np.pi)

            # --- Non-linear screening factor K_n = -lambda_n / (lambda_n + 1) ---
            K_n = -eval_n / (eval_n + 1.0)  # (naux_ri_c,)

            # Kmat[m, m'] = sum_n K_n * evec[m, n] * evec[m', n]
            Kmat = (evec * K_n) @ evec.T  # (naux_ri_c, naux_ri_c)

            for s, data in enumerate(rhs_data):
                if data is None:
                    continue
                fai_ri, N_diag, M_virt_T, M_occ_T, oep_occ_flat, oep_virt_flat, inv_denom_i, inv_denom_a = data
                _, nv, nelec = fai_ri.shape
                mo_energy = spins[s][2]

                # --- F_ai: rotate occ-virt RI integrals to eigenbasis ---
                F_ai = (evec.T @ fai_ri_flat[s]).reshape(naux_ri_c, nv, nelec)  # (naux_ri_c, nv, no)

                # lambda with sign flip (as in Fortran: lambda1 = -lambda1)
                lam_neg = -lam[s]  # (nv, no)

                # --- gamma: d(lambda_rpa)/d(eps_ai) with opposite sign, used in Yia term ---
                gamma = self._get_gamma_rpa(mo_energy, nelec, omega)  # (nv, no)

                # R[m, a, i] = sum_{m'} Kmat[m, m'] * lam_neg[a, i] * fai_ri[m', a, i]
                R = (Kmat @ (lam_neg[None] * fai_ri).reshape(naux_ri_c, nv * nelec)).reshape(
                    naux_ri_c, nv, nelec
                )  # (naux_ri_c, nv, no)

                # --- Rii: t_i[s, i] = sum_{m,a} ints_3c_ri[m, nelec+a, s] * R[m, a, i] ---
                T_si = (M_virt_T @ R.reshape(naux_ri_c * nv, nelec)) * inv_denom_i  # (nmo, no)
                rhs_om = 2.0 * (oep_occ_flat @ T_si.reshape(-1))

                # --- Ria: t_a[s, a] = sum_{m,i} ints_3c_ri[m, i, s] * R[m, a, i] ---
                T_sa = (M_occ_T @ R.transpose(0, 2, 1).reshape(naux_ri_c * nelec, nv)) * inv_denom_a  # (nmo, nv)
                rhs_om += 2.0 * (oep_virt_flat @ T_sa.reshape(-1))

                # --- Yia term: diagonal correction ---
                # Y_ai[a, i] = gamma[a, i] * sum_n K_n * F_ai[n, a, i]^2
                Y_ai = gamma * (K_n @ (F_ai**2).reshape(naux_ri_c, -1)).reshape(nv, nelec)  # (nv, no)

                # rhs[nu] += sum_{a,i} Y_ai[a,i] * (N_diag[nu, no+a] - N_diag[nu, i])
                rhs_om += N_diag[:, nelec:] @ Y_ai.sum(axis=1) - N_diag[:, :nelec] @ Y_ai.sum(axis=0)

                rhs[s] += rhs_om * wt / (2.0 * np.pi)

        return rhs, E_corr

    def _get_lambda_rpa(self, mo_energy, nelec, omega, thr=1e-6):
        """
        RPA energy denominator: lambda_{ai}(omega) = 4 * eps_{ai} / (eps_{ai}^2 + omega^2).
        Factor 4 comes from the electron-hole propagator (vs factor 1 for EXX).
        """
        eps_ai = mo_energy[nelec:, None] - mo_energy[None, :nelec]  # (nv, no), positive
        eps_ai = np.where(np.abs(eps_ai) > thr, eps_ai, np.sign(eps_ai + 1e-300) * thr)
        return 4.0 * eps_ai / (eps_ai**2 + omega**2)

    def _get_gamma_rpa(self, mo_energy, nelec, omega, thr=1e-6):
        """
        Frequency-dependent weight for the Yia diagonal term:
        gamma_{ai}(omega) = -4 / (eps_{ai}^2 + omega^2) + 8 * eps_{ai}^2 / (eps_{ai}^2 + omega^2)^2.
        This is -d(lambda_rpa)/d(eps_{ai}), accounting for orbital energy variations.
        """
        eps_ai = mo_energy[nelec:, None] - mo_energy[None, :nelec]  # (nv, no)
        eps_ai = np.where(np.abs(eps_ai) > thr, eps_ai, np.sign(eps_ai + 1e-300) * thr)
        denom = eps_ai**2 + omega**2
        return -4.0 / denom + 8.0 * eps_ai**2 / denom**2
