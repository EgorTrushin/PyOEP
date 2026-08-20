#!/usr/bin/env python3

import sys
import scipy
import numpy as np
from copy import deepcopy
from pyscf import dft
from .exxoep import EXXOEP


class DFTOEP(EXXOEP):
    r"""
    Implements density functional theory optimized effective potential method for closed-shell systems.
    The implementation follows:
    E. Trushin, A. Görling. J. Chem. Theory Comput. 2025, 21, 4, 1667–1683. https://doi.org/10.1021/acs.jctc.4c01477
    E. Trushin, A. Görling. J. Chem. Phys. 155, 054109 (2021). https://doi.org/10.1063/5.0056431

    Args:
        mf: PySCF object with RKS calculation
        oep_basis: auxiliary basis to solve OEP equation
        vh_via_OEP: whether to construct AO Hartree potential via OEP basis
        space_sym: whether to perform space-symmetrization
        use_HOMO_condition: whether to use HOMO condition
        xc_homo: exchange-functional to use for evaluation of the HOMO condition
        first_iter: whether to calculate HOMO energy for HOMO condition only at first iteration
    """

    def __init__(
        self,
        mf,
        oep_basis,
        use_HOMO_condition=False,
        vh_via_OEP=False,
        space_sym=False,
        xc_homo=None,
        ip_first_iter=False,
    ):
        self.xc_homo = xc_homo
        self.ip_first_iter = ip_first_iter
        self.ip = None
        super().__init__(mf, oep_basis, use_HOMO_condition, vh_via_OEP, space_sym)

    def get_energies_and_potentials(self):
        """Determines energy contributions and potentials."""

        self.mf.max_cycle = 0
        self.mf.run()
        self.e_tot = self.mf.e_tot

        dm = self.mf.make_rdm1()

        ni = dft.numint.NumInt()
        _, _, vxc = ni.nr_rks(self.mf.mol, self.mf.grids, self.mf.xc, dm)

        if dft.libxc.is_hybrid_xc(self.mf.xc):
            omega, alpha, hyb = ni.rsh_and_hybrid_coeff(self.mf.xc)
            if omega == 0:
                self.vj_ao, self.vxnl_ao = self.mf.get_jk(self.mf.mol, dm)
            elif alpha == 0:  # LR=0, only SR exchange
                self.vj_ao = self.mf.get_j(self.mf.mol, dm)
                self.vxnl_ao = self.mf.get_k(self.mf.mol, dm, omega=-omega)
            else:
                sys.exit("Not implemented case for hybrid functionals")
        else:
            self.vj_ao = self.mf.get_j(self.mf.mol, dm)

        self.E_Coul = np.einsum("ij,ji->", self.vj_ao, dm) * 0.5

        if dft.libxc.is_hybrid_xc(self.mf.xc):
            self.E_x = -0.5 * np.einsum("ij,ji->", self.vxnl_ao, dm) * 0.5
            self.vxnl_ao *= -0.5

        if dft.libxc.is_hybrid_xc(self.mf.xc):
            self.vxnl_ao = vxc + hyb * self.vxnl_ao
        else:
            self.vxnl_ao = vxc

        if self.use_HOMO_condition and not (self.ip_first_iter and self.ip is not None):
            mf1 = deepcopy(self.mf)
            mf1.xc = self.xc_homo
            mf1.max_cycle = 0
            mf1.kernel(dm0=dm)
            E_n_x = mf1.e_tot

            mol1 = deepcopy(self.mf.mol)
            mol1.spin = 1
            mol1.charge += 1
            mol1.build(False, False)
            occ = np.stack((mf1.mo_occ, mf1.mo_occ)) / 2.0
            occ[1][self.nelec - 1] = 0
            mo0 = np.stack((mf1.mo_coeff, mf1.mo_coeff))
            mf1 = dft.UKS(mol1, xc=self.xc_homo).density_fit()
            mf1.max_cycle = 0
            dm1 = mf1.make_rdm1(mo0, occ)
            mf1.kernel(dm0=dm1)
            E_nm1_x = mf1.e_tot

            self.ip = E_nm1_x - E_n_x

    def get_v_ref_w_homo(self, zII, ints_3c, mo_coeff, nelec, vxnl_ao):
        r"""
        Constructs the Fermi-Amaldi reference potential with HOMO condition.
        See Appendix C in J. Chem. Phys. 155 (2021) 054109
        and J. Chem. Theory Comput. 2025, 21, 4, 1667–1683
        """
        u = 2 * np.einsum("ijj->i", ints_3c[:, :nelec, :nelec])
        vcII = self.WII.T @ u
        aux_x = np.zeros([2, 2])
        aux_x[0, 0] = np.dot(self.yII, self.yII)
        aux_x[0, 1] = np.dot(self.yII, vcII)
        aux_x[1, 0] = np.dot(zII, self.yII)
        aux_x[1, 1] = np.dot(zII, vcII)
        aux_y = np.zeros([2])
        aux_y[0] = -1
        aux_y[1] = -self.ip - (mo_coeff.T @ (self.mf.get_hcore() + self.vj_ao) @ mo_coeff)[nelec - 1, nelec - 1]
        aux_sol = scipy.linalg.solve(aux_x, aux_y)
        vrefII = aux_sol[0] * self.yII + aux_sol[1] * vcII
        vref_oep = self.WII @ vrefII
        return vref_oep

    def potentials_test(
        self, vrest_oep, vref_oep, vrest_ao, vref_ao, vxnl_ao, mo_coeff, nelec, z
    ):  # note that vxnl_ao is unused intentionally
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
            v_aux = mo_coeff.T @ (self.mf.get_hcore() + self.vj_ao) @ mo_coeff
            if (
                abs(-v_aux[nelec - 1, nelec - 1] - self.ip - (mo_coeff.T @ vref_ao @ mo_coeff)[nelec - 1, nelec - 1])
                > 1e-12
            ):
                print("Warning!")
                print(f"v(HOMO) =  {-v_aux[nelec - 1, nelec - 1] - self.ip:.5f}  (VxNL)")
                print(f"v(HOMO) =  {(mo_coeff.T @ vref_ao @ mo_coeff)[nelec - 1, nelec - 1]:.5f}  (Vref)")
