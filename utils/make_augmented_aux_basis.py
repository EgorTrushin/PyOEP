#!/usr/bin/env python3

from pyscf import gto


def make_augmented_aux_basis(mol, aux_basis, extra_exponents=None):
    """
    Add tight s-type Gaussians to an auxiliary basis to improve cusp representation.

    Args:
        mol:             PySCF Mole object
        aux_basis:       str, base auxiliary basis name
        extra_exponents: dict {element: [exp1, exp2, ...]}

    Returns:
        augmented basis dict suitable for gto.M(basis=...)
    """
    augmented = {}
    for elem in set([mol.atom_symbol(i) for i in range(mol.natm)]):
        basis = gto.basis.load(aux_basis, elem)  # list of shells
        exps = extra_exponents.get(elem, [])

        # Append each extra exponent as a new s-shell
        for exp in exps:
            basis.append([0, [exp, 1.0]])  # [l, [exponent, coeff]]

        augmented[elem] = basis

    return augmented
