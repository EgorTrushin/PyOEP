#!/usr/bin/env python3

from .hybridrpaoep import HybridRPAOEP
from .osrpaoep import OSRPAOEP


class OSHybridRPAOEP(OSRPAOEP):
    r"""
    Implements hybrid RPA optimized effective potential method for open-shell systems.
    A fraction alpha of non-local HF exchange is combined with (1-alpha) of the local RPA-OEP
    exchange potential in the Fock matrices:

        F_s = h + vJ + alpha * vx_HF_s + (1-alpha) * (vref_s + vrest_s) + vc_OEP_s

    where s = alpha, beta. The OEP equation is solved as in OSRPAOEP (targeting the full vxnl_ao);
    only the Fock matrix construction is modified.
    Note: vxnl_ao[0/1] = -K_alpha/beta (scaled by -1, not -0.5, unlike closed-shell).
    The implementation follows:
    A. Thierbach, D. Schmidtel, A. Görling. J. Chem. Phys. 151, 144117 (2019). https://doi.org/10.1063/1.5120587
    E. Trushin, R. Mandalia, A. Görling. J. Chem. Phys. 165, 134115 (2026). https://doi.org/10.1063/5.0345799

    Args:
        mf: PySCF object with UHF or UKS calculation
        oep_basis: auxiliary basis to solve the EXX-OEP equation
        ri_basis: auxiliary RI basis for the RPA response function
        alpha: fraction of non-local HF exchange (default 0.25)
        use_HOMO_condition: whether to use HOMO condition for the exchange potential
        vh_via_OEP: whether to construct AO Hartree potential via OEP basis
        space_sym: whether to perform space-symmetrization
        spin_sym: whether to use spin symmetrization
    """

    def __init__(
        self,
        mf,
        oep_basis,
        ri_basis,
        alpha=0.25,
        use_HOMO_condition=False,
        vh_via_OEP=False,
        space_sym=False,
        spin_sym=False,
    ):
        self.alpha = alpha
        super().__init__(mf, oep_basis, ri_basis, use_HOMO_condition, vh_via_OEP, space_sym, spin_sym)

    get_vx_fock = HybridRPAOEP.get_vx_fock
