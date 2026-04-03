#!/usr/bin/env python3

import numpy as np


def gen_coords_1d(z_min, z_max, n_points, mode="z"):
    z = np.linspace(z_min, z_max, n_points)
    coords = np.zeros([z.shape[0], 3])
    if mode == "z":
        coords[:, 2] = z
    elif mode == "x":
        coords[:, 0] = z
    elif mode == "y":
        coords[:, 1] = z
    else:
        print("gen_coords_1d: unknown mode")
        return None
    return coords
