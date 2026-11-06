import numpy as np
import pytest
from pyscf import cc, gto, scf
from pyscf.grad import ccsd_t as ccsd_t_grad

GEOM = "O 0 0 0.117; H 0 0.755 -0.469; H 0 -0.755 -0.469"
ORBITAL_BASIS = "cc-pVDZ"
FIELD = 2e-3


def ccsd_energy(method, field=0.0):
    """CCSD or CCSD(T) energy with a uniform field coupled to the electrons along z."""
    mol = gto.M(atom=GEOM, basis=ORBITAL_BASIS, verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    hcore = mf.get_hcore() + field * mol.intor("int1e_r", comp=3)[2]
    mf.get_hcore = lambda *args: hcore
    mf.kernel()
    mycc = cc.CCSD(mf)
    mycc.conv_tol = 1e-11
    mycc.conv_tol_normt = 1e-8
    mycc.kernel()
    return mycc.e_tot + (mycc.ccsd_t() if method == "ccsd(t)" else 0.0)


@pytest.mark.parametrize("method", ["ccsd", "ccsd(t)"])
def test_rdm1_relaxed_is_the_energy_derivative(method):
    """PySCF's relaxed density reproduces dE/dF, and holds the right number of electrons."""
    mol = gto.M(atom=GEOM, basis=ORBITAL_BASIS, verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    mycc = cc.CCSD(mf)
    mycc.conv_tol = 1e-11
    mycc.conv_tol_normt = 1e-8
    mycc.kernel()
    mf_grad = mycc.Gradients() if method == "ccsd" else ccsd_t_grad.Gradients(mycc)
    mf_grad.kernel()
    dm = mf_grad.rdm1_relaxed

    energies = [ccsd_energy(method, s * FIELD) for s in (2, 1, -1, -2)]
    finite_field = (-energies[0] + 8 * energies[1] - 8 * energies[2] + energies[3]) / (12 * FIELD)

    assert abs(np.sum(dm * mol.intor("int1e_r", comp=3)[2]) - finite_field) < 1e-7
    assert abs(np.sum(dm * mol.intor_symmetric("int1e_ovlp")) - mol.nelectron) < 1e-8
