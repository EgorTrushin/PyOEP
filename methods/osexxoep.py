#!/usr/bin/env python3

import scipy
import numpy as np
from pyscf import lib
from .exxoep import EXXOEP


class OSEXXOEP(EXXOEP):
    r"""
    Implements exact-exchange optimized effective potential method for open-shell systems.
    The implementation follows:
    E. Trushin, A. Görling. J. Chem. Phys. 155, 054109 (2021). https://doi.org/10.1063/5.0056431
    E. Trushin, A. Görling. J. Chem. Phys. 159, 244109 (2023). https://doi.org/10.1063/5.0171546

    Args:
        mf: PySCF object with UHF or UKS calculation
        oep_basis: auxiliary basis to solve OEP equation
        use_HOMO_condition: whether to use HOMO condition
        vh_via_OEP: whether to construct AO Hartree potential via OEP basis
        space_sym: whether to perform space-symmetrization
        spin_sym: whether to use spin symmetrization
    """

    def __init__(self, mf, oep_basis, use_HOMO_condition=False, vh_via_OEP=False, space_sym=False, spin_sym=False):
        self.ip = [None, None]  # subclasses (e.g. OSDFTOEP) may set this to [ip_alpha, ip_beta]
        super().__init__(mf, oep_basis, use_HOMO_condition, vh_via_OEP, space_sym)
        self.nelec = mf.nelec
        self.spin_sym = spin_sym

    def run(self, maxit=50, thr_fai_oep=5e-2, linear_mixing=-1.0, e_conv_thr=1e-8):
        r"""
        Performs a self-consistent calculation.

        Args:
            maxit: maximal number of iterations
            thr_fai_oep: threshold T_{ai} from Section IIA5 of J. Chem. Phys. 155 (2021) 054109
            linear_mixing: if specified larger than 0, linear mixing scheme is used instead of DIIS
                           with given linear_mixing coefficient
            e_conv_thr: threshold for energy convergence
        """
        if linear_mixing > 0:
            fock_old_a, fock_old_b = None, None
        else:
            adiis = lib.diis.DIIS()

        e_tot_old = None

        print("ITER" + " " * 8 + "ENERGY" + " " * 15 + "EDIFF")
        for current_iter in range(maxit):
            ints_3c_a = self.mf.mo_coeff[0].T @ self.ints_3c_ao @ self.mf.mo_coeff[0]
            ints_3c_b = self.mf.mo_coeff[1].T @ self.ints_3c_ao @ self.mf.mo_coeff[1]
            z_a = z_b = None

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
                    )  # ip[0] is None for EXX (unused), or alpha IP for DFT subclasses
                    vref_oep_b = self.get_v_ref_w_homo(
                        zII_b, ints_3c_b, self.mf.mo_coeff[1], self.nelec[1], self.vxnl_ao[1], self.ip[1]
                    )  # ip[1] is None for EXX (unused), or beta IP for DFT subclasses
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
                vrest_oep_a,
                vref_oep_a,
                vrest_ao_a,
                vref_ao_a,
                self.vxnl_ao[0],
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
                    self.vxnl_ao[1],
                    self.mf.mo_coeff[1],
                    self.nelec[1],
                    z_b,
                    self.ip[1] if self.use_HOMO_condition else None,
                )

            h1e = self.mf.get_hcore(self.mf.mol)
            F_a = h1e + self.vj_ao + vref_ao_a + vrest_ao_a
            F_b = h1e + self.vj_ao + vref_ao_b + vrest_ao_b

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
                D_a = self.mf.mo_coeff[0][:, : self.nelec[0]] @ self.mf.mo_coeff[0][:, : self.nelec[0]].T
                D_b = self.mf.mo_coeff[1][:, : self.nelec[1]] @ self.mf.mo_coeff[1][:, : self.nelec[1]].T
                e_a = F_a @ D_a @ S - S @ D_a @ F_a
                e_b = F_b @ D_b @ S - S @ D_b @ F_b
                nao = F_a.shape[0]
                F_combined = adiis.update(
                    np.concatenate([F_a.ravel(), F_b.ravel()]), xerr=np.concatenate([e_a.ravel(), e_b.ravel()])
                )
                F_a = F_combined[: nao * nao].reshape(nao, nao)
                F_b = F_combined[nao * nao :].reshape(nao, nao)

            S = self.mf.get_ovlp()
            mo_energy, mo_coeff = self.mf._eigh(F_a, S)
            self.mf.mo_energy[0], self.mf.mo_coeff[0] = mo_energy, mo_coeff
            mo_energy, mo_coeff = self.mf._eigh(F_b, S)
            self.mf.mo_energy[1], self.mf.mo_coeff[1] = mo_energy, mo_coeff

            self.get_energies_and_potentials()

            if e_tot_old is None:
                print(f"{current_iter:3}  {self.e_tot:18.12f}")
                e_tot_old = self.e_tot
            else:
                print(f"{current_iter:3}  {self.e_tot:18.12f}  {self.e_tot - e_tot_old:18.12f}")
                if abs(e_tot_old - self.e_tot) < e_conv_thr:
                    print("SCF converged")
                    self.converged = True
                    self.vref_oep_a = vref_oep_a
                    self.vref_oep_b = vref_oep_b
                    self.vrest_oep_a = vrest_oep_a
                    self.vrest_oep_b = vrest_oep_b
                    break
                e_tot_old = self.e_tot

            if current_iter == maxit - 1:
                print("SCF was not converged")
                self.vref_oep_a = vref_oep_a
                self.vref_oep_b = vref_oep_b
                self.vrest_oep_a = vrest_oep_a
                self.vrest_oep_b = vrest_oep_b

    def get_energies_and_potentials(self):
        """Determines energy contributions and potentials."""
        dm = self.mf.make_rdm1()
        h1e = self.mf.get_hcore()
        e1 = np.einsum("ij,ji->", h1e, dm[0])
        e1 += np.einsum("ij,ji->", h1e, dm[1])
        self.vj_ao, self.vxnl_ao = self.mf.get_jk(dm=np.asarray(dm))

        self.vj_ao = self.vj_ao[0] + self.vj_ao[1]
        self.E_Coul = np.einsum("ij,ji->", self.vj_ao, dm[0] + dm[1]) * 0.5
        self.E_x = -0.5 * np.einsum("ij,ji->", self.vxnl_ao[0], dm[0])
        self.E_x += -0.5 * np.einsum("ij,ji->", self.vxnl_ao[1], dm[1])
        self.e_tot = e1 + self.E_Coul + self.E_x + self.mf.energy_nuc()
        self.vxnl_ao *= -1.0

    def get_v_ref_spin_sym(self, ints_3c_a, ints_3c_b, nelec):
        r"""
        Constructs the Fermi-Amaldi reference potential in spin-symmetrized case.
        See Sections IIB in J. Chem. Phys. 159, 244109 (2023)
        """
        u = np.einsum("ijj->i", ints_3c_a[:, : nelec[0], : nelec[0]])
        u += np.einsum("ijj->i", ints_3c_b[:, : nelec[1], : nelec[1]])
        u *= 0.5
        vcII = self.WII.T @ u
        vc = self.WII @ vcII
        v_ref = -1 / np.dot(self.y, vc) * vc
        return v_ref

    def get_v_ref_w_homo_spin_sym(self, zII, ints_3c_a, ints_3c_b, mo_coeff_a, nelec, vxnl_ao):
        r"""
        Constructs the Fermi-Amaldi reference potential with HOMO condition
        for spin-symmetrized case.
        See Appendix C in J. Chem. Phys. 155 (2021) 054109
        and J. Chem. Phys. 159, 244109 (2023)
        """
        u = np.einsum("ijj->i", ints_3c_a[:, : nelec[0], : nelec[0]])
        u += np.einsum("ijj->i", ints_3c_b[:, : nelec[1], : nelec[1]])
        u *= 0.5
        vcII = self.WII.T @ u
        aux_x = np.zeros([2, 2])
        aux_x[0, 0] = self.charge_norm
        aux_x[0, 1] = np.dot(self.yII, vcII)
        aux_x[1, 0] = np.dot(zII, self.yII)
        aux_x[1, 1] = np.dot(zII, vcII)
        aux_y = np.zeros([2])
        aux_y[0] = -1
        aux_y[1] = (mo_coeff_a.T @ vxnl_ao @ mo_coeff_a)[nelec[0] - 1, nelec[0] - 1]
        aux_sol = scipy.linalg.solve(aux_x, aux_y)
        vrefII = aux_sol[0] * self.yII + aux_sol[1] * vcII
        vref_oep = self.WII @ vrefII
        return vref_oep

    def get_vh_via_OEP(self, ints_3c_a, ints_3c_b, nelec):
        """Constructs AO Hartree potential via OEP basis."""
        u = np.einsum("ijj->i", ints_3c_a[:, : nelec[0], : nelec[0]])
        u += np.einsum("ijj->i", ints_3c_b[:, : nelec[1], : nelec[1]])
        vcII = self.WII.T @ u
        vc = self.WII @ vcII
        vc = sum(nelec) / np.dot(self.y, vc) * vc
        self.vj_ao = np.einsum("ijk,k->ij", self.ints_3c_ao_t, vc[:])
