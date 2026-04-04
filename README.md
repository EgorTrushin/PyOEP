# PyOEP

Gaussian basis set optimized effective potential methods which are implemented by means of PySCF.

- Exact-exchange optimized effective potential method for closed- and open-shell systems
  - E. Trushin, A. Görling. J. Chem. Phys. 155, 054109 (2021). https://doi.org/10.1063/5.0056431
  - E. Trushin, A. Görling. J. Chem. Phys. 159, 244109 (2023). https://doi.org/10.1063/5.0171546

- Density functional theory optimized effective potential method for closed-shell and open-shell systems
  - E. Trushin, A. Görling. J. Chem. Theory Comput. 2025, 21, 4, 1667–1683. https://doi.org/10.1021/acs.jctc.4c01477

- Response function Kohn-Sham inversion method for closed- and open-shell systems
  - J. Erhard, E. Trushin, A. Görling. J. Chem. Phys. 156, 204124 (2022). https://doi.org/10.1063/5.0087356
  - J. Erhard, E. Trushin, A. Görling. J. Chem. Phys. 162, 034116 (2025). https://doi.org/10.1063/5.0239422

- Random phase approximation optimized effective potential method for closed- and open-shell systems
  - P. Bleiziffer, A. Heßelmann, A. Görling. J. Chem. Phys. 139, 084113 (2013). https://doi.org/10.1063/1.4818984
  - E. Trushin, S. Fauser, A. Mölkner, J. Erhard, A. Görling. Phys. Rev. Lett. 134, 016402 (2025). https://doi.org/10.1103/PhysRevLett.134.016402

## Installation

Clone the repository and install the required packages:

```bash
pip install pyscf basis_set_exchange matplotlib
```

NumPy and SciPy are included with PySCF. For running tests, `pytest` is also required:

```bash
pip install pytest
```

To make the `methods/` and `utils/` packages importable, either run scripts from the project root or add the project root to `PYTHONPATH`.

**Performance note:** Using an Intel Python distribution with MKL (e.g. via `conda install -c intel intelpython3_core`) is strongly recommended over the standard OpenBLAS-based NumPy/SciPy. MKL-accelerated linear algebra routines can provide a substantial speedup for the matrix operations used throughout PyOEP.

## Class overview

| Class | File | Method | System |
|---|---|---|---|
| `EXXOEP` | methods/exxoep.py | EXX-OEP | closed-shell |
| `OSEXXOEP` | methods/osexxoep.py | EXX-OEP | open-shell |
| `DFTOEP` | methods/dftoep.py | DFT-OEP | closed-shell |
| `OSDFTOEP` | methods/osdftoep.py | DFT-OEP | open-shell |
| `KSINV` | methods/ksinv.py | KS inversion | closed-shell |
| `OSKSINV` | methods/osksinv.py | KS inversion | open-shell |
| `RPAOEP` | methods/rpaoep.py | RPA-OEP | closed-shell |
| `OSRPAOEP` | methods/osrpaoep.py | RPA-OEP | open-shell |

`OSEXXOEP` inherits from `EXXOEP`, `OSDFTOEP` from `OSEXXOEP`, `KSINV` from `EXXOEP`, `OSKSINV` from `KSINV`, `RPAOEP` from `EXXOEP`, and `OSRPAOEP` from `OSEXXOEP`.

## Repository structure

```
methods/    OEP and KS inversion classes (see class overview above)
tests/      test suite
data/       reference data (FCI, CCSD densities and energies) used in tutorials
utils/      helper scripts for pre- and post-processing, used in tutorials
*.ipynb     tutorial notebooks
```

## Examples

Four brief examples are provided below. More detailed examples with discussion can be found in the [tutorials](#tutorials).

### EXX-OEP for a closed-shell molecule (CO)

```python
import matplotlib.pyplot as plt
from pyscf import scf, gto
from methods.exxoep import EXXOEP
from utils.coulomb_potential_on_grid import coulomb_potential_on_grid
from utils.gen_coords_1d import gen_coords_1d

# Three separate basis sets are used:
#   ORBITAL_BASIS  — high-quality orbital basis for the SCF calculation
#   OEP_BASIS      — auxiliary basis in which the OEP is expanded
#   DFIT_BASIS     — density-fitting basis for efficient Coulomb integrals in RHF
ORBITAL_BASIS = "aug-cc-pwCVQZ"
OEP_BASIS = "aug-cc-pVDZ-RIFIT"
DFIT_BASIS = "aug-cc-pwCV5Z-RIFIT"
GEOM = "C 0.000000    0.000000   -0.646514; O 0.000000    0.000000    0.484886"

mol = gto.M(atom=GEOM, basis=ORBITAL_BASIS)
mol.verbose = 0
mol.symmetry = False  # symmetry must be disabled for OEP calculations

# Run RHF with density fitting to obtain reference orbitals and integrals
mf = scf.RHF(mol).density_fit(auxbasis=DFIT_BASIS).run()

# Set up and run the EXX-OEP self-consistent calculation
mf_oep = EXXOEP(mf, OEP_BASIS)
mf_oep.run(maxit=15, thr_fai_oep=1.7e-2)

# Print orbital energies; HOMO and LUMO-HOMO gap are key outputs
for i in range(10):
    print(f"{i+1:2}  {mf_oep.mf.mo_energy[i]:10.5f}")
print(f"HOMO:      {mf_oep.mf.mo_energy[mf_oep.nelec-1]:8.5f}")
print(f"LUMO-HOMO: {mf_oep.mf.mo_energy[mf_oep.nelec]-mf_oep.mf.mo_energy[mf_oep.nelec-1]:8.5f}")

# Evaluate the converged potentials on a 1D grid along the molecular axis
coords = gen_coords_1d(-5.0, 5.0, 1000)
vcoul_on_grid = coulomb_potential_on_grid(mf_oep.pmol, coords)
vrest_on_grid = vcoul_on_grid @ mf_oep.vrest_oep
vref_on_grid  = vcoul_on_grid @ mf_oep.vref_oep
vx_on_grid    = vcoul_on_grid @ (mf_oep.vrest_oep + mf_oep.vref_oep)

# vref is the Fermi-Amaldi reference potential; vrest is the remainder;
# their sum is the total EXX exchange potential
plt.plot(coords[:, 2], vref_on_grid,  color="orangered",  label="$v_{x}^{ref}$")
plt.plot(coords[:, 2], vrest_on_grid, color="dodgerblue", label="$v_{x}^{rest}$")
plt.plot(coords[:, 2], vx_on_grid,    color="orange",     label="$v_x$")
plt.xlim(-5, 5)
plt.ylabel("Potential (a.u.)", fontsize=16)
plt.xlabel("r (a.u.)", fontsize=16)
plt.legend()
plt.show()
```

### DFT-OEP for a closed-shell molecule (CO)

```python
import matplotlib.pyplot as plt
from pyscf import dft, gto
from methods.dftoep import DFTOEP
from utils.coulomb_potential_on_grid import coulomb_potential_on_grid
from utils.gen_coords_1d import gen_coords_1d

ORBITAL_BASIS = "aug-cc-pwCVQZ"
OEP_BASIS = "aug-cc-pVDZ-RIFIT"
DFIT_BASIS = "aug-cc-pwCV5Z-RIFIT"
GEOM = "C 0.000000    0.000000   -0.646514; O 0.000000    0.000000    0.484886"

mol = gto.M(atom=GEOM, basis=ORBITAL_BASIS)
mol.verbose = 0
mol.symmetry = False  # symmetry must be disabled for OEP calculations

# Run RKS with the desired xc functional
mf = dft.RKS(mol, xc="MGGA_X_R2SCAN, MGGA_C_R2SCAN").density_fit(auxbasis=DFIT_BASIS)
mf.grids.level = 4
mf.run()

# Set up and run the DFT-OEP self-consistent calculation
mf_oep = DFTOEP(mf, OEP_BASIS)
mf_oep.run(maxit=15, thr_fai_oep=1.7e-2)

# Print orbital energies; HOMO and LUMO-HOMO gap are key outputs
for i in range(10):
    print(f"{i+1:2}  {mf_oep.mf.mo_energy[i]:10.5f}")
print(f"HOMO:      {mf_oep.mf.mo_energy[mf_oep.nelec-1]:8.5f}")
print(f"LUMO-HOMO: {mf_oep.mf.mo_energy[mf_oep.nelec]-mf_oep.mf.mo_energy[mf_oep.nelec-1]:8.5f}")

# Evaluate the converged potentials on a 1D grid along the molecular axis;
# vref is the Fermi-Amaldi reference potential, vrest is the remainder,
# their sum is the total xc OEP potential
coords = gen_coords_1d(-5.0, 5.0, 1000)
vcoul_on_grid = coulomb_potential_on_grid(mf_oep.pmol, coords)
vrest_on_grid = vcoul_on_grid @ mf_oep.vrest_oep
vref_on_grid  = vcoul_on_grid @ mf_oep.vref_oep
vxc_on_grid   = vcoul_on_grid @ (mf_oep.vrest_oep + mf_oep.vref_oep)

plt.plot(coords[:, 2], vref_on_grid,  color="orangered",  label="$v_{xc}^{ref}$")
plt.plot(coords[:, 2], vrest_on_grid, color="dodgerblue", label="$v_{xc}^{rest}$")
plt.plot(coords[:, 2], vxc_on_grid,   color="orange",     label="$v_{xc}$")
plt.xlim(-5, 5)
plt.ylabel("Potential (a.u.)", fontsize=16)
plt.xlabel("r (a.u.)", fontsize=16)
plt.legend()
plt.show()
```

### RPA-OEP for a closed-shell molecule (CO)

The RPA-OEP method requires three auxiliary basis sets: an OEP basis for the potential expansion, an RI basis for the RPA response function, and a density-fitting basis for the Coulomb integrals.

```python
import matplotlib.pyplot as plt
from pyscf import scf, gto
from methods.rpaoep import RPAOEP
from utils.coulomb_potential_on_grid import coulomb_potential_on_grid
from utils.gen_coords_1d import gen_coords_1d

ORBITAL_BASIS = "aug-cc-pwCVQZ"
OEP_BASIS = "aug-cc-pVDZ-RIFIT"
RI_BASIS = "aug-cc-pwCVQZ-RIFIT"
DFIT_BASIS = "aug-cc-pwCV5Z-RIFIT"
GEOM = "C 0.000000    0.000000   -0.646514; O 0.000000    0.000000    0.484886"

mol = gto.M(atom=GEOM, basis=ORBITAL_BASIS)
mol.verbose = 0
mol.symmetry = False  # symmetry must be disabled for OEP calculations

# Run RHF with density fitting to obtain reference orbitals and integrals
mf = scf.RHF(mol).density_fit(auxbasis=DFIT_BASIS).run()

# Set up and run the RPA-OEP self-consistent calculation
mf_oep = RPAOEP(mf, OEP_BASIS, RI_BASIS)
mf_oep.run(maxit=30, thr_fai_oep=1.7e-2)

# Print orbital energies
for i in range(10):
    print(f"{i+1:2}  {mf_oep.mf.mo_energy[i]:10.5f}")
print(f"HOMO:      {mf_oep.mf.mo_energy[mf_oep.nelec-1]:8.5f}")
print(f"LUMO-HOMO: {mf_oep.mf.mo_energy[mf_oep.nelec]-mf_oep.mf.mo_energy[mf_oep.nelec-1]:8.5f}")

# Evaluate exchange and correlation potentials on a 1D grid along the molecular axis
coords = gen_coords_1d(-5.0, 5.0, 1000)
vcoul_on_grid = coulomb_potential_on_grid(mf_oep.pmol, coords)
vx_on_grid  = vcoul_on_grid @ (mf_oep.vref_oep + mf_oep.vrest_oep)
vc_on_grid  = vcoul_on_grid @ mf_oep.vrest_c_oep
vxc_on_grid = vcoul_on_grid @ (mf_oep.vref_oep + mf_oep.vrest_oep + mf_oep.vrest_c_oep)

plt.plot(coords[:, 2], vx_on_grid,  color="orangered", label="$v_x$")
plt.plot(coords[:, 2], vc_on_grid,  color="orange",    label="$v_c$")
plt.plot(coords[:, 2], vxc_on_grid, color="dodgerblue", label="$v_{xc}$")
plt.xlim(-5, 5)
plt.ylabel("Potential (a.u.)", fontsize=16)
plt.xlabel("r (a.u.)", fontsize=16)
plt.legend()
plt.show()
```

### KS inversion for a closed-shell molecule (CO)

The KS inversion recovers the xc potential corresponding to a given reference density, here taken from a CCSD relaxed density matrix.

```python
import matplotlib.pyplot as plt
from pyscf import scf, gto, cc
from methods.ksinv import KSINV
from utils.relaxed_ccsd import cc_rrdm1
from utils.coulomb_potential_on_grid import coulomb_potential_on_grid
from utils.gen_coords_1d import gen_coords_1d

ORBITAL_BASIS = "aug-cc-pwCVTZ"
GEOM = "C 0.000000    0.000000   -0.646514; O 0.000000    0.000000    0.484886"

mol = gto.M(atom=GEOM, basis=ORBITAL_BASIS)
mol.verbose = 0
mol.symmetry = False  # symmetry must be disabled for KS inversion

# Run RHF to obtain the reference orbitals for CCSD
mf = scf.RHF(mol)
mf.kernel()
print(f"Hartree-Fock total energy: {mf.e_tot:15.12f}")

# Run CCSD
mf_cc = cc.CCSD(mf)
mf_cc.kernel()
print(f"CCSD correlation energy:   {mf_cc.e_corr:15.12f}")
print(f"CCSD total energy:         {mf_cc.e_tot:15.12f}")
dm_ccsd = cc_rrdm1(mf_cc)  # relaxed CCSD density matrix

# Run KS inversion
mf_inv = KSINV(mf, "aug-cc-pVDZ-RIFIT", dm_ccsd, mf_cc.e_tot)
mf_inv.run(maxit=100, thr_fai_oep=5e-2)

# Evaluate and plot the total xc potential on a 1D grid along the molecular axis
coords = gen_coords_1d(-15.0, 15.0, 1000)
vcoul_on_grid = coulomb_potential_on_grid(mf_inv.pmol, coords)
vxc_on_grid = vcoul_on_grid @ (mf_inv.vrest_oep + mf_inv.vref_oep)

plt.plot(coords[:, 2], vxc_on_grid, color="orangered", label="$v_{xc}$")
plt.xlim(-5, 5)
plt.ylabel("Potential (a.u.)", fontsize=16)
plt.xlabel("r (a.u.)", fontsize=16)
plt.legend(frameon=False)
plt.show()
```

## Tutorials

- [Exact-exchange optimized effective potential method](./EXXOEP.ipynb)
- [Density functional theory optimized effective potential method](./DFTOEP.ipynb)
- [Response function Kohn-Sham inversion method](./KSINV.ipynb)
- [Self-consistent RPA optimized effective potential method](./RPAOEP.ipynb)
- [Advanced topics](./Advanced.ipynb)

## Running tests

Run all tests from the project root:

```bash
python -m pytest tests/
```

Run a specific test file:

```bash
python -m pytest tests/test_exxoep.py
python -m pytest tests/test_osexxoep.py
python -m pytest tests/test_dftoep.py
python -m pytest tests/test_osdftoep.py
python -m pytest tests/test_ksinv.py
python -m pytest tests/test_osksinv.py
python -m pytest tests/test_rpaoep.py
python -m pytest tests/test_osrpaoep.py
```

The `-m` flag ensures the `methods/` package is importable when running from the project root. Alternatively, use plain `pytest tests/` if `PYTHONPATH` is set (see Installation).

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
