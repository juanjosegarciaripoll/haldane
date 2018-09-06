"""Define the Hamiltonian class.

This class contains the hopping matrix, the on-site Hubbard energy and
the extended Hubbard interaction matrix.
"""

import numpy as np


class OpenHamiltonian(object):
    """Matrices of a Haldane Hamiltonian with open boundaries.

    Attributes:
        lattice (OpenLattice): lattice on which the Hamiltonian is
            defined.
        A (2darray of floats): hopping matrix.

    """

    def __init__(self, lattice, hopping_params, hoppings_type,
                 trap_potential, lattice_imbalance, inhomogeneity=None):
        """Initialize the Hamiltonian class.

        Args:
            lattice (OpenLattice): lattice on which the Hamiltonian is
                defined.
            hopping_params (tuple of floats): hopping parameters,
                explained below.
            hoppings_type: string with the name of the hoppings we want
                to use:
                'juanjo': modified Haldane Hamiltonian:
                    hopping_param = (t, ta, tb, dpy).
                'haldane': Haldane's Hamiltonian.
                    hopping_param = (t1, t2, dphi).
            trap_potential (float): k coefficient of the harmonic trap.
                The potential is quadratic like: V(x) = k*x**2, with
                x the distance of a point in the lattice to the center.
            lattice_imbalance (float): energy difference between the
                two sublattices.
            inhomogeneity (tuple of floats): distance from the center
                of the origin of the inhomogeneity, angle of the origin
                of the inhomogeneity with respect to the xy axis,
                strength, and variance of the inhomogeneity.

        """
        self.lattice = lattice
        self.A = np.zeros((self.lattice.L, self.lattice.L), np.complex128)
        self.build_A(hopping_params, hoppings_type,
                     trap_potential, lattice_imbalance)

    def build_A(self, hopping_params, hoppings_type,
                trap_potential, lattice_imbalance, inhomogeneity=None):
        """Compute the hopping matrix A."""
        # Diagonal terms.
        # Harmonic trap.
        center = (self.lattice.xy_coords[0] + self.lattice.xy_coords[-1])/2
        for i in range(self.lattice.L):
            i_coords = self.lattice.xy_coords[i]
            self.A[i, i] += trap_potential*np.linalg.norm(i_coords - center)**2

        # Lattice imbalance.
        if not np.isclose(lattice_imbalance, 0):
            for i in range(self.lattice.L):
                self.A[i, i] += (lattice_imbalance if i%2 == 0
                                 else -lattice_imbalance)

        # Inhomogeneity.
        if inhomogeneity is not None:
            R_inh = inhomogeneity[0]
            theta_inh = inhomogeneity[1]
            xy_inh = np.array([R_inh*np.cos(theta_inh),
                               R_inh*np.sin(theta_inh)])
            s_inh = inhomogeneity[2]
            var_inh = inhomogeneity[3]

            for i in range(self.lattice.L):
                self.A[i, i] += s_inh*np.exp(np.linalg.norm(self.lattice.xy_coords[i] - xy_inh)**2/(2*var_inh**2))

        # Hopping terms.
        # Assert hoppings are valid.
        hoppings_type_list = ['juanjo', 'haldane']
        if hoppings_type not in hoppings_type_list:
            raise RuntimeError('{} is not a valid hopping.'
                               .format(hoppings_type))

        # Compute the hopping amplitudes.
        if hoppings_type == 'juanjo':
            t, ta, tb, dpy = hopping_params
            py = dpy*np.pi
            tv1 = t*(np.cos(py) + 1j*np.sin(py))
            tv2 = t*(np.cos(py) - 1j*np.sin(py))
            tv3 = t
            amp_first_neigh_B = np.array([tv1, tv2, tv3])
            amp_first_neigh_A = np.conj(amp_first_neigh_B)
            amp_second_neigh_B = np.full(6, tb, np.float64)
            amp_second_neigh_A = np.full(6, ta, np.float64)

        elif hoppings_type == 'haldane':
            t1, t2, dphi = hopping_params
            tmp = t2*np.exp(1j*dphi*np.pi)
            amp_first_neigh_B = np.full(3, t1, np.float64)
            amp_first_neigh_A = np.full(3, t1, np.float64)
            amp_second_neigh_B = np.array(
                [tmp, np.conj(tmp), tmp, np.conj(tmp), tmp, np.conj(tmp)]
                )
            amp_second_neigh_A = np.conj(amp_second_neigh_B)

        # Add the hoppings to A.
        for i in range(self.lattice.L):
            if self.lattice.lat_coords[i, 2] == 0:
                # Sublattice A.
                for (delta, amp) in zip(self.lattice.first_neigh_A,
                                        amp_first_neigh_A):
                    self.add_hopping(i, delta, amp)
                for (delta, amp) in zip(self.lattice.second_neigh,
                                        amp_second_neigh_A):
                    self.add_hopping(i, delta, amp)

            else:
                # Sublattice B.
                for (delta, amp) in zip(self.lattice.first_neigh_B,
                                        amp_first_neigh_B):
                    self.add_hopping(i, delta, amp)
                for (delta, amp) in zip(self.lattice.second_neigh,
                                        amp_second_neigh_B):
                    self.add_hopping(i, delta, amp)

        return

    def add_hopping(self, i, delta, amp):
        """Add a hopping to the hopping matrix A.

        Args:
            i (int): index of the initial position of the hopping.
            delta (1darray of ints): hopping direction in lattice
                coordinates.
            amp (complex of float): hopping amplitude.

        """
        # Initial and final coordinates.
        ic = self.lattice.lat_coords[i]
        jc = ic + delta
        j = self.lattice.position_to_index(jc)

        # If jc is outside the lattice, j == -1. We avoid these cases.
        if j >= 0:
            self.A[j, i] += amp

        return
