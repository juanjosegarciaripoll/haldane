"""Define the Hamiltonian class.

This class contains the hopping matrix, the on-site Hubbard energy and
the extended Hubbard interaction matrix.
"""

import numpy as np


def add_hopping(lattice, A, i, delta, amp):
    """Add a hopping to the hopping matrix A.

    Args:
        i (int): index of the initial position of the hopping.
        delta (1darray of ints): hopping direction in lattice
            coordinates.
        amp (complex of float): hopping amplitude.

    """
    # Initial and final coordinates.
    ic = lattice.lat_coords[i]
    jc = ic + delta
    j = lattice.position_to_index(jc)

    # If jc is outside the lattice, j == -1. We avoid these cases.
    if j >= 0:
        A[j, i] += amp

    return


def build_A(lattice, hopping_params, trap_potential):
    """Compute the hopping matrix of the Haldane Hamiltonian.

    Args:
        lattice (OpenLattice): lattice on which the Hamiltonian is
            defined.
        hopping_params (tuple of floats): hopping parameters:
            (t1, t2, dϕ, ϵ), where ϵ is the sublattice energy imbalance
            and dϕ is the ϕ phase divided by pi.
        trap_potential (function): assigns an on-site potential to
            every site in the lattice.

    """
    t1, t2, dϕ, ϵ = hopping_params

    A = np.zeros((lattice.L, lattice.L), np.complex128)

    # Harmonic trap.
    for i in range(lattice.L):
        i_coords = lattice.xy_coords[i]
        A[i, i] += trap_potential(i_coords)

    # Lattice imbalance.
    if not np.isclose(ϵ, 0):
        for i in range(lattice.L):
            A[i, i] += (ϵ if i%2 == 0 else -ϵ)

    # Hopping terms.
    t2 = t2*np.exp(1j*dϕ*np.pi)
    amp_first_neigh_B = np.full(3, t1, np.float64)
    amp_first_neigh_A = np.full(3, t1, np.float64)
    amp_second_neigh_B = np.array(
        [t2, np.conj(t2), t2, np.conj(t2), t2, np.conj(t2)]
        )
    amp_second_neigh_A = np.conj(amp_second_neigh_B)

    # Add the hoppings to A.
    for i in range(lattice.L):
        if lattice.lat_coords[i, 2] == 0:
            # Sublattice A.
            for (delta, amp) in zip(lattice.first_neigh_A,
                                    amp_first_neigh_A):
                add_hopping(lattice, A, i, delta, amp)
            for (delta, amp) in zip(lattice.second_neigh,
                                    amp_second_neigh_A):
                add_hopping(lattice, A, i, delta, amp)

        else:
            # Sublattice B.
            for (delta, amp) in zip(lattice.first_neigh_B,
                                    amp_first_neigh_B):
                add_hopping(lattice, A, i, delta, amp)
            for (delta, amp) in zip(lattice.second_neigh,
                                    amp_second_neigh_B):
                add_hopping(lattice, A, i, delta, amp)

    return A
