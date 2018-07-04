"""Compute the currents of a state."""

import numpy as np

from lattice import OpenLattice
from circular_lattice import CircularLattice


def compute_currents(lattice, J, v):
    """Compute the currents of a state.

    Args:
        lattice (OpenLattice): lattice where we compute the currents.
        J (2darray of complex): hopping matrix.
        v (2darray of floats): state from which we compute the currents.

    Returns:
        currents (1darray of floats): local variation of the number of
            particles at each site.

    """
    expected = np.einsum('ik,jk->ij', np.conj(v), v, optimize=True)

    L = lattice.L
    currents = np.zeros(2*L, np.float64)
    for i in range(L):
        if i%2 == 1:
            # Sublattice A.
            for delta in lattice.first_neigh_A:
                ic = lattice.index_to_position(i)
                jc = ic + delta
                j = lattice.position_to_index(jc)

                not_crosses_boundary = True
                if isinstance(lattice, OpenLattice):
                    Nx = lattice.Nx
                    Ny = lattice.Ny
                    not_crosses_boundary = ((0 <= jc[0] <= Nx)
                                            and (0 <= jc[1] <= Ny)
                                            and (0 <= j < L))
                elif isinstance(lattice, CircularLattice):
                    not_crosses_boundary = j >= 0

                if not_crosses_boundary:
                    # Count currents of both spin populations.
                    currents[i] += 2*np.imag(J[i, j]*expected[i, j])
                    currents[i] += 2*np.imag(J[i, j]*expected[i+L, j+L])

        else:
            # Sublattice B.
            for delta in lattice.first_neigh_B:
                ic = lattice.index_to_position(i)
                jc = ic + delta
                j = lattice.position_to_index(jc)

                not_crosses_boundary = True
                if isinstance(lattice, OpenLattice):
                    Nx = lattice.Nx
                    Ny = lattice.Ny
                    not_crosses_boundary = ((0 <= jc[0] <= Nx)
                                            and (0 <= jc[1] <= Ny)
                                            and (0 <= j < L))
                elif isinstance(lattice, CircularLattice):
                    not_crosses_boundary = j >= 0

                if not_crosses_boundary:
                    # Count currents of both spin populations.
                    currents[i] += 2*np.imag(J[i, j]*expected[i, j])
                    currents[i] += 2*np.imag(J[i, j]*expected[i+L, j+L])

    return currents
