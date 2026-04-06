#!/usr/bin/env python3

import scipy
import numpy as np
from copy import deepcopy
from pyscf import dft, gto, df, lib

import functools

print = functools.partial(print, flush=True)


class EXXOEP:
    r"""
    Implements exact-exchange optimized effective potential method for closed-shell systems.
    The implementation follows:
    E. Trushin, A. Görling. J. Chem. Phys. 155, 054109 (2021). https://doi.org/10.1063/5.0056431

    Args:
        mf: PySCF object with RHF or RKS calculation
        oep_basis: auxiliary basis to solve OEP equation
        use_HOMO_condition: whether to use HOMO condition
        vh_via_OEP: whether to construct AO Hartree potential via OEP basis
        space_sym: whether to perform space-symmetrization
    """

    def __init__(self, mf, oep_basis, use_HOMO_condition=False, vh_via_OEP=False, space_sym=False):
        self.use_HOMO_condition = use_HOMO_condition
        self.nelec = sum(i != 0 for i in mf.mo_occ)
        self.init_common(mf, oep_basis, vh_via_OEP, space_sym)
        self.get_energies_and_potentials()

    def init_common(self, mf, oep_basis, vh_via_OEP, space_sym):
        """
        Shared initialization for all OEP and KS inversion classes.
        Builds the auxiliary mol, three-center integrals, integration grid,
        and performs the auxiliary basis set preprocessing (WII, y, yII).
        Called from __init__ of EXXOEP and from subclasses that bypass
        EXXOEP.__init__ (KSINV, OSKSINV) to avoid triggering
        get_energies_and_potentials().
        """
        self.mf = deepcopy(mf)
        self.oep_basis = oep_basis
        self.vh_via_OEP = vh_via_OEP
        self.space_sym = space_sym
        self.auxmol = df.addons.make_auxmol(self.mf.mol, self.oep_basis)
        self.ints_3c_ao = df.incore.aux_e2(self.mf.mol, self.auxmol, intor="int3c2e").transpose(2, 1, 0)
        self.naux, self.nmo, _ = self.ints_3c_ao.shape
        self.ints_3c_ao_t = self.ints_3c_ao.transpose(1, 2, 0)
        self.build_pmol_and_grid()
        self.get_WII()
        self.get_y_and_yII()
        self.converged = False

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
            fock_old = None
        else:
            adiis = lib.diis.DIIS()

        e_tot_old = None

        print("ITER" + " " * 8 + "ENERGY" + " " * 15 + "EDIFF")
        for current_iter in range(maxit):
            ints_3c = self.mf.mo_coeff.T @ self.ints_3c_ao @ self.mf.mo_coeff
            z = None
            if self.use_HOMO_condition:
                z, zII = self.get_z_and_zII(ints_3c, self.mf.mo_energy, self.nelec)
                vref_oep = self.get_v_ref_w_homo(zII, ints_3c, self.mf.mo_coeff, self.nelec, self.vxnl_ao)
                W3 = self.get_W3_charge_and_homo(zII)
            else:
                vref_oep = self.get_v_ref(ints_3c, self.nelec)
                W3 = self.get_W3_charge()

            if self.vh_via_OEP or self.space_sym:
                self.get_vh_via_OEP(ints_3c, self.nelec)

            W = self.get_W(W3, ints_3c, self.nelec, thr_fai_oep)

            X0 = self.get_X0(ints_3c, self.mf.mo_energy, self.nelec, W)

            vref_ao = np.einsum("ijk,k->ij", self.ints_3c_ao_t, vref_oep[:])

            rhs = self.get_rhs(vref_ao, self.vxnl_ao, ints_3c, self.mf.mo_coeff, self.mf.mo_energy, self.nelec, W)

            rhs = scipy.linalg.solve(X0, rhs)
            vrest_oep = W @ rhs

            vrest_ao = np.einsum("ijk,k->ij", self.ints_3c_ao_t, vrest_oep[:])

            self.potentials_test(
                vrest_oep,
                vref_oep,
                vrest_ao,
                vref_ao,
                self.vxnl_ao,
                self.mf.mo_coeff,
                self.nelec,
                z,
            )

            h1e = self.mf.get_hcore(self.mf.mol)
            F = h1e + self.vj_ao + vref_ao + vrest_ao

            if linear_mixing > 0:
                if fock_old is None:
                    fock_old = F.copy()
                else:
                    F = (1.0 - linear_mixing) * F + linear_mixing * fock_old
                    fock_old = F.copy()
            else:
                F = adiis.update(F)

            S = self.mf.get_ovlp()
            self.mf.mo_energy, self.mf.mo_coeff = scipy.linalg.eigh(F, S)

            self.get_energies_and_potentials()

            if e_tot_old is None:
                print(f"{current_iter:3}  {self.e_tot:18.12f}")
                e_tot_old = self.e_tot
            else:
                print(f"{current_iter:3}  {self.e_tot:18.12f}  {self.e_tot - e_tot_old:18.12f}")
                if abs(e_tot_old - self.e_tot) < e_conv_thr:
                    print("SCF converged")
                    self.converged = True
                    self.vref_oep = vref_oep
                    self.vrest_oep = vrest_oep
                    break
                e_tot_old = self.e_tot

            if current_iter == maxit - 1:
                print("SCF was not converged")
                self.vref_oep = vref_oep
                self.vrest_oep = vrest_oep

    def get_energies_and_potentials(self):
        """Determines energy contributions and potentials."""
        dm = self.mf.make_rdm1()
        h1e = self.mf.get_hcore()
        e1 = np.einsum("ij,ji->", h1e, dm)
        self.vj_ao, self.vxnl_ao = self.mf.get_jk(dm=np.asarray(dm))
        self.E_Coul = np.einsum("ij,ji->", self.vj_ao, dm) * 0.5
        self.E_x = -0.5 * np.einsum("ij,ji->", self.vxnl_ao, dm) * 0.5
        self.e_tot = e1 + self.E_Coul + self.E_x + self.mf.energy_nuc()
        self.vxnl_ao *= -0.5

    def build_pmol_and_grid(self):
        """Builds auxiliary mol and integration grid, shared by get_WII and get_y_and_yII."""
        self.pmol = gto.M(
            atom=self.mf.mol.atom, basis=self.oep_basis, spin=self.mf.mol.spin, charge=self.mf.mol.charge
        )
        self.pmol.verbose = 0
        self.grid = dft.gen_grid.Grids(self.pmol)
        self.grid.build()

    def get_WII(self):
        r"""
        Performs the first two steps of the auxiliary basis set preprocessing.
        See Sections IIB1 and IIB2 in J. Chem. Phys. 155 (2021) 054109.
        """
        smat = self.auxmol.intor("int2c2e", aosym="s1", comp=1)
        diag = 1.0 / np.sqrt(np.diagonal(smat))
        diag_mat = np.diag(diag)
        smat = (diag_mat @ smat) @ diag_mat
        eigs, evecs, _ = scipy.linalg.lapack.dsyev(smat)

        if self.space_sym:
            symvec_aux = gto.eval_gto(self.pmol, "GTOval_sph", self.grid.coords)
            symvec = symvec_aux.T @ self.grid.weights
            symvec[abs(symvec) > 1e-10] = 1.0
            eigs[np.abs(evecs.T @ symvec) < 1e-10] = 0.0
            mask = np.abs(eigs) > 1e-10
            smh = evecs[:, mask] / np.sqrt(eigs[mask])
        else:
            eigs = 1.0 / np.sqrt(eigs)
            smh = evecs @ np.diag(eigs)

        self.WII = diag_mat @ smh

    def get_y_and_yII(self):
        r"""
        Performs part of the third step of the auxiliary basis set preprocessing.
        See Sections IIB3 in J. Chem. Phys. 155 (2021) 054109.
        """
        y_aux = gto.eval_gto(self.pmol, "GTOval_sph", self.grid.coords)
        self.y = y_aux.T @ self.grid.weights
        self.yII = np.matmul(np.transpose(self.WII), self.y)
        self.charge_norm = np.dot(self.yII, self.yII)

    def get_W3_charge(self):
        r"""
        Performs part of the third step of the auxiliary basis set preprocessing.
        See Sections IIB3 in J. Chem. Phys. 155 (2021) 054109.
        """
        auxmat = np.eye(self.yII.shape[0]) - np.outer(self.yII, self.yII) / self.charge_norm
        auxmat2 = auxmat.T @ auxmat + np.eye(self.WII.shape[1])
        _, evecs = scipy.linalg.eigh(auxmat2)
        return evecs[:, 1:]

    def get_v_ref(self, ints_3c, nelec):
        r"""
        Constructs the Fermi-Amaldi reference potential.
        See Sections IIB4 in J. Chem. Phys. 155 (2021) 054109.
        """
        u = 2 * np.einsum("ijj->i", ints_3c[:, :nelec, :nelec])
        vcII = self.WII.T @ u
        vc = self.WII @ vcII
        v_ref = -1 / np.dot(self.y, vc) * vc
        return v_ref

    def get_z_and_zII(self, ints_3c, mo_energy, nelec):
        r"""
        Determines quantities required for the auxiliary basis set preprocessing with HOMO condition.
        See Appendix C in J. Chem. Phys. 155 (2021) 054109.
        """
        ndeg = 1
        for i in range(1, nelec - 1):
            if abs(mo_energy[nelec - 1 - i] - mo_energy[nelec - 1]) < 1e-12:
                ndeg += 1
            else:
                break
        z = ints_3c[:, nelec - 1, nelec - 1].copy()
        for ideg in range(1, ndeg):
            z += ints_3c[:, nelec - 1 - ideg, nelec - 1 - ideg]
        z /= ndeg
        zII = self.WII.T @ z
        return z, zII

    def get_W3_charge_and_homo(self, zII):
        r"""
        Performs part of the third step of the auxiliary basis set preprocessing with HOMO condition.
        See Appendix C in J. Chem. Phys. 155 (2021) 054109.
        """
        constraint_vector = np.stack((self.yII, zII)).T
        overlap_cv = constraint_vector.T @ constraint_vector
        eval_overlap_cv, evec_overlap_cv = scipy.linalg.eigh(overlap_cv)
        eval_ocv_sqrtinv = np.diag(1.0 / np.sqrt(eval_overlap_cv))
        orth_constraint = constraint_vector @ (evec_overlap_cv @ eval_ocv_sqrtinv)
        proj_mat = np.eye(orth_constraint.shape[0]) - orth_constraint @ orth_constraint.T
        orth_overlap = proj_mat.T @ proj_mat
        _, evecs = scipy.linalg.eigh(orth_overlap)
        return evecs[:, 2:]

    def get_v_ref_w_homo(self, zII, ints_3c, mo_coeff, nelec, vxnl_ao, ip=None):
        r"""
        Constructs the Fermi-Amaldi reference potential with HOMO condition.
        See Appendix C in J. Chem. Phys. 155 (2021) 054109.
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
        aux_y[1] = (mo_coeff.T @ vxnl_ao @ mo_coeff)[nelec - 1, nelec - 1]
        aux_sol = scipy.linalg.solve(aux_x, aux_y)
        vrefII = aux_sol[0] * self.yII + aux_sol[1] * vcII
        vref_oep = self.WII @ vrefII
        return vref_oep

    def get_W(self, W3, ints_3c, nelec, thr_fai_oep):
        r"""
        Performs the last step of the auxiliary basis set preprocessing and returns the final transformation matrix.
        See Sections IIB5 in J. Chem. Phys. 155 (2021) 054109.
        """
        trans_mat_constraint = self.WII @ W3
        dmat = trans_mat_constraint.T @ ints_3c[:, nelec:, :nelec].reshape(self.naux, (self.nmo - nelec) * nelec)
        amat = dmat @ dmat.T
        eigs, evecs = scipy.linalg.eigh(amat)
        nsing = (eigs < thr_fai_oep).sum()
        trans_mat = trans_mat_constraint @ evecs[:, nsing:]
        return trans_mat

    def get_W_spin_sym(self, W3, ints_3c_a, ints_3c_b, nelec, thr_fai_oep):
        r"""
        Spin-symmetrized variant of get_W: performs the last step of the auxiliary
        basis set preprocessing for open-shell systems when alpha and beta potentials
        are constrained to be equal.

        The occupied-virtual coupling matrix A is averaged over alpha and beta spins,
        A = 0.5 * (A_alpha + A_beta), and its eigenvectors with eigenvalues above
        thr_fai_oep define the final transformation matrix. This enforces a single
        shared OEP expansion space for both spin channels.

        See Section IIB in J. Chem. Phys. 159, 244109 (2023).
        """
        trans_mat_constraint = self.WII @ W3
        dmat_a = trans_mat_constraint.T @ ints_3c_a[:, nelec[0]:, :nelec[0]].reshape(
            self.naux, (self.nmo - nelec[0]) * nelec[0]
        )
        amat_a = dmat_a @ dmat_a.T
        dmat_b = trans_mat_constraint.T @ ints_3c_b[:, nelec[1]:, :nelec[1]].reshape(
            self.naux, (self.nmo - nelec[1]) * nelec[1]
        )
        amat_b = dmat_b @ dmat_b.T
        amat = 0.5 * (amat_a + amat_b)
        eigs, evecs, _ = scipy.linalg.lapack.dsyev(amat)
        nsing = (eigs < thr_fai_oep).sum()
        trans_mat = trans_mat_constraint @ evecs[:, nsing:]
        return trans_mat

    def get_e_ov(self, mo_energy, nelec):
        """Returns inverse occupied-virtual energy differences (virtual, occupied)."""
        e_ov = (mo_energy[:nelec, None] - mo_energy[None, nelec:]).T
        e_ov = np.where(abs(e_ov) > 1e-6, e_ov, np.sign(e_ov + 1e-12) * 1e-6)
        return 1.0 / e_ov

    def get_X0(self, ints_3c, mo_energy, nelec, trans_mat):
        """Constructs Kohn-Sham response matrix."""
        e_ov = self.get_e_ov(mo_energy, nelec)
        Pia = ints_3c[:, nelec:, :nelec] * e_ov
        X0 = np.einsum("Pia, Qia -> PQ", Pia, ints_3c[:, nelec:, :nelec])
        X0 = trans_mat.T @ X0 @ trans_mat
        return X0

    def get_rhs(self, v_ref_ao, vxnl_ao, ints_3c, mo_coeff, mo_energy, nelec, trans_mat):
        """Constructs RHS of EXX-OEP equation."""
        vxmo = mo_coeff.T @ (vxnl_ao - v_ref_ao) @ mo_coeff
        e_ov = self.get_e_ov(mo_energy, nelec)
        aux = np.einsum("ra,ra,ira->i", vxmo[nelec:, :nelec], e_ov, ints_3c[:, nelec:, :nelec])
        rhs = trans_mat.T @ aux
        return rhs

    def potentials_test(self, vrest_oep, vref_oep, vrest_ao, vref_ao, vxnl_ao, mo_coeff, nelec, z, ip=None):
        """Performs consistency checks for potentials.
        ip is accepted for interface consistency with subclass overrides; unused in the base class.
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
                print(f"v(HOMO) =  {(mo_coeff.T@vrest_ao@mo_coeff)[nelec-1, nelec-1]:.5f}  (VrestL)")
            if (
                abs(
                    (mo_coeff.T @ vxnl_ao @ mo_coeff)[nelec - 1, nelec - 1]
                    - (mo_coeff.T @ vref_ao @ mo_coeff)[nelec - 1, nelec - 1]
                )
                > 1e-12
            ):
                print("Warning!")
                print(f"v(HOMO) =  {(mo_coeff.T@vxnl_ao@mo_coeff)[nelec-1, nelec-1]:.5f}  (VxNL)")
                print(f"v(HOMO) =  {(mo_coeff.T@vref_ao@mo_coeff)[nelec-1, nelec-1]:.5f}  (Vref)")

    def get_vh_via_OEP(self, ints_3c, nelec):
        """Constructs AO Hartree potential via OEP basis."""
        u = 2 * np.einsum("ijj->i", ints_3c[:, :nelec, :nelec])
        vcII = self.WII.T @ u
        vc = self.WII @ vcII
        vc = 2.0 * nelec / np.dot(self.y, vc) * vc
        self.vj_ao = np.einsum("ijk,k->ij", self.ints_3c_ao_t, vc[:])
