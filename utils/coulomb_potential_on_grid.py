#!/usr/bin/env python3

import numpy as np
from pyscf import gto


def coulomb_potential_on_grid(mol, grid_coords, batch_size=2000):
    """
    V_n(r_i) = integral[ phi_n(r') / |r_i - r'| dr' ]
    int2c2e gives this directly, no normalization correction needed.
    """
    n_pts = len(grid_coords)
    nao = mol.nao_nr()
    V = np.zeros((n_pts, nao))

    for start in range(0, n_pts, batch_size):
        end = min(start + batch_size, n_pts)
        fakemol = gto.fakemol_for_charges(grid_coords[start:end], expnt=1e16)
        ints = gto.mole.intor_cross("int2c2e", fakemol, mol)  # (n_batch, nao)
        V[start:end] = ints

    return V
