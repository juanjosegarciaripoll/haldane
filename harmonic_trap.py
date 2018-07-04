"""Build an harmonic potential trap."""

import numpy as np


def build_harmonic_trap(lattice, k):
    """Build the energy levels of an harmonic trap.

    The trap potential is quadratic like:
        V(x, y) = k*[(x-center[0])**2 + (y-center[1])**2]
    The center of the trap is located at the middle between the first
    and last lattice points.

    Args:
        lattice (OpenLattice): lattice where we set the trap.
        k (float): quadratic coefficient of the trap.

    Returns:
        single_levels (1darray of floats): single particle levels that
            make the harmonic trap.

    """
    # Get center position.
    center = (lattice.coords_pts[0] + lattice.coords_pts[-1])/2

    single_levels = np.zeros(lattice.L, np.float64)
    for i in range(lattice.L):
        i_coords = lattice.coords_pts[i]
        single_levels[i] = k*np.linalg.norm(i_coords - center)**2

    return single_levels
