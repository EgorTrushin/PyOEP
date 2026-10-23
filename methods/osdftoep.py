#!/usr/bin/env python3

from copy import deepcopy

import numpy as np
import scipy
from pyscf import dft

from .dftoep import DFTOEP
from .osexxoep import OSEXXOEP


class OSDFTOEP(OSEXXOEP):
    r"""
    Implements density functional theory optimized effective potential method for open-shell systems.
    The implementation follows:
    E. Trushin, A. Görling. J. Chem. Theory Comput. 2025, 21, 4, 1667–1683. https://doi.org/10.1021/acs.jctc.4c01477
    E. Trushin, A. Görling. J. Chem. Phys. 155, 054109 (2021). https://doi.org/10.1063/5.0056431
    E. Trushin, A. Görling. J. Chem. Phys. 159, 244109 (2023). https://doi.org/10.1063/5.0171546

    The exchange-correlation part of the potential of mf.xc (including exact exchange of hybrid and
    range-separated hybrid functionals and VV10 nonlocal correlation) is replaced by a local potential.

    Args:
        mf: PySCF object with UKS calculation
        oep_basis: auxiliary basis to solve OEP equation
        vh_via_OEP: whether to construct AO Hartree potential via OEP basis
        space_sym: whether to perform space-symmetrization
        spin_sym: whether to use spin symmetrization
        use_HOMO_condition: whether to use HOMO condition
        xc_homo: exchange-functional to use for evaluation of the HOMO condition (required with use_HOMO_condition)
        ip_first_iter: whether to calculate HOMO energy for HOMO condition only at first iteration
    """

    def __init__(
        self,
        mf,
        oep_basis,
        use_HOMO_condition=False,
        vh_via_OEP=False,
        space_sym=False,
        spin_sym=False,
        xc_homo=None,
        ip_first_iter=False,
    ):
        if use_HOMO_condition and xc_homo is None:
            raise ValueError("xc_homo is required when use_HOMO_condition=True")
        self.xc_homo = xc_homo
        self.ip_first_iter = ip_first_iter
        super().__init__(mf, oep_basis, use_HOMO_condition, vh_via_OEP, space_sym, spin_sym)

    _xc_homo_mf = DFTOEP._xc_homo_mf

    def get_energies_and_potentials(self):
        """Determines energy contributions and potentials."""
        dm = self.mf.make_rdm1()
        veff = self.mf.get_veff(self.mf.mol, dm)
        self.e_tot = self.mf.energy_tot(dm, self.mf.get_hcore(), veff)
        self.E_Coul = veff.ecoul
        self.E_xc = veff.exc
        self.vj_ao = veff.vj
        self.vxnl_ao = np.asarray(veff) - veff.vj

        if self.use_HOMO_condition and not (self.ip_first_iter and self.ip[0] is not None):
            self.ip = self.get_ip(dm)

    def get_ip(self, dm):
        """
        Determines the ionization potentials [ip_alpha, ip_beta] for the HOMO condition as the xc_homo
        energy differences between the cation with the alpha or beta HOMO electron removed
        (frozen orbitals) and the neutral system.
        """
        E_n_x = self._xc_homo_mf(self.mf.mol, dft.UKS).energy_tot(dm)

        ip = []
        for spin in (0, 1):
            mol1 = deepcopy(self.mf.mol)
            mol1.spin += 1 if spin else -1  # removing an alpha (beta) electron lowers (raises) 2S
            mol1.charge += 1
            mol1.build(False, False)
            occ = self.mf.mo_occ.copy()
            occ[spin, self.nelec[spin] - 1] = 0
            mf1 = self._xc_homo_mf(mol1, dft.UKS)
            E_nm1_x = mf1.energy_tot(mf1.make_rdm1(self.mf.mo_coeff, occ))
            ip.append(E_nm1_x - E_n_x)

        return ip

    def get_v_ref_w_homo(self, zII, ints_3c, mo_coeff, nelec, vxnl_ao, ip):
        r"""
        Constructs the Fermi-Amaldi reference potential with HOMO condition.
        ip is the ionization potential for the spin channel being processed.
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
        aux_y[1] = -ip - (mo_coeff.T @ (self.mf.get_hcore() + self.vj_ao) @ mo_coeff)[nelec - 1, nelec - 1]
        aux_sol = scipy.linalg.solve(aux_x, aux_y)
        vrefII = aux_sol[0] * self.yII + aux_sol[1] * vcII
        vref_oep = self.WII @ vrefII
        return vref_oep

    def get_v_ref_w_homo_spin_sym(self, zII, ints_3c_a, ints_3c_b, mo_coeff_a, nelec, vxnl_ao):
        r"""
        Constructs the Fermi-Amaldi reference potential with HOMO condition
        for spin-symmetrized case.
        See Appendix C in J. Chem. Phys. 155 (2021) 054109
        and J. Chem. Phys. 159, 244109 (2023)
        """
        u = np.einsum("ijj->i", ints_3c_a[:, : nelec[0], : nelec[0]])
        u += np.einsum("ijj->i", ints_3c_b[:, : nelec[1], : nelec[1]])
        vcII = self.WII.T @ u
        aux_x = np.zeros([2, 2])
        aux_x[0, 0] = np.dot(self.yII, self.yII)
        aux_x[0, 1] = np.dot(self.yII, vcII)
        aux_x[1, 0] = np.dot(zII, self.yII)
        aux_x[1, 1] = np.dot(zII, vcII)
        aux_y = np.zeros([2])
        aux_y[0] = -1
        aux_y[1] = (
            -self.ip[0] - (mo_coeff_a.T @ (self.mf.get_hcore() + self.vj_ao) @ mo_coeff_a)[nelec[0] - 1, nelec[0] - 1]
        )
        aux_sol = scipy.linalg.solve(aux_x, aux_y)
        vrefII = aux_sol[0] * self.yII + aux_sol[1] * vcII
        vref_oep = self.WII @ vrefII
        return vref_oep

    def potentials_test(
        self, vrest_oep, vref_oep, vrest_ao, vref_ao, vxnl_ao, mo_coeff, nelec, z, ip=None
    ):  # note that vxnl_ao is unused intentionally
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
            v_aux = mo_coeff.T @ (self.mf.get_hcore() + self.vj_ao) @ mo_coeff
            if abs(-v_aux[nelec - 1, nelec - 1] - ip - (mo_coeff.T @ vref_ao @ mo_coeff)[nelec - 1, nelec - 1]) > 1e-12:
                print("Warning!")
                print(f"v(HOMO) =  {-v_aux[nelec - 1, nelec - 1] - ip:.5f}  (VxNL)")
                print(f"v(HOMO) =  {(mo_coeff.T @ vref_ao @ mo_coeff)[nelec - 1, nelec - 1]:.5f}  (Vref)")
