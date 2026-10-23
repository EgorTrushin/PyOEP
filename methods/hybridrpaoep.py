#!/usr/bin/env python3

from .rpaoep import RPAOEP


class HybridRPAOEP(RPAOEP):
    r"""
    Implements hybrid RPA optimized effective potential method for closed-shell systems.
    A fraction alpha of non-local HF exchange is combined with (1-alpha) of the local RPA-OEP
    exchange potential in the Fock matrix:

        F = h + vJ + alpha * vx_HF + (1-alpha) * (vref + vrest) + vc_OEP

    The OEP equation is solved as in RPAOEP (targeting the full vxnl_ao); only the Fock matrix
    construction is modified.
    The implementation follows:
    A. Thierbach, D. Schmidtel, A. Görling. J. Chem. Phys. 151, 144117 (2019). https://doi.org/10.1063/1.5120587
    E. Trushin, R. Mandalia, A. Görling. J. Chem. Phys. 165, 134115 (2026). https://doi.org/10.1063/5.0345799

    Args:
        mf: PySCF object with RHF or RKS calculation
        oep_basis: auxiliary basis to solve the EXX-OEP equation
        ri_basis: auxiliary RI basis for the RPA response function
        alpha: fraction of non-local HF exchange (default 0.25)
        use_HOMO_condition: whether to use HOMO condition for the exchange potential
        vh_via_OEP: whether to construct AO Hartree potential via OEP basis
        space_sym: whether to perform space-symmetrization
    """

    def __init__(
        self, mf, oep_basis, ri_basis, alpha=0.25, use_HOMO_condition=False, vh_via_OEP=False, space_sym=False
    ):
        self.alpha = alpha
        super().__init__(mf, oep_basis, ri_basis, use_HOMO_condition, vh_via_OEP, space_sym)

    def get_vx_fock(self, vx_local_ao, vxnl_ao):
        """Exchange part of the Fock matrix: alpha * vxnl_ao + (1 - alpha) * vx_local_ao."""
        return self.alpha * vxnl_ao + (1.0 - self.alpha) * vx_local_ao
