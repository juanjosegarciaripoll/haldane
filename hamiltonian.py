"""Define the Hamiltonian class.

This class contains the hopping matrix, the on-site Hubbard energy and
the extended Hubbard interaction matrix.
"""

import numpy as np

from lattice import OpenLattice
from circular_lattice import CircularLattice


class OpenHamiltonian(object):
    """Matrices of a Haldane Hamiltonian with open boundaries.

    Attributes:
        lattice (OpenLattice): lattice on which the Hamiltonian is
            defined.
        J (2darray of floats): hopping matrix.

    """

    def __init__(self, lattice, single_levels, hopping_params, hoppings_type):
        """Initialize the Hamiltonian class.

        Args:
            lattice (OpenLattice): lattice on which the Hamiltonian is
                defined.
            single_levels (1darray of floats): single particle levels.
            hopping_params (list of floats): hopping parameters, explained
                below.
            hoppings_type: string with the name of the hoppings we want
                to use:
                'juanjo': juanjo hopping hamiltonian:
                    hopping_param = [t, ta, tb, dpy].
                'haldane': haldane's hamiltonian.
                    hopping_param = [t1, t2, dphi].
                'graphene': similar to haldane but with real hoppings.
                    hopping_param = [t1, t2].
        """
        self.lattice = lattice
        self.J = np.diag(single_levels.astype(np.complex128))
        self.build_J(hopping_params, hoppings_type)

    def build_J(self, hopping_params, hoppings_type):
        """Add the proper hoppings to J.

        Args:
            hopping_params: list of hopping parameters. Explained below.
            hoppings_type: string with the name of the hoppings we want
                to use:
                'juanjo': juanjo hopping hamiltonian:
                    hopping_param = [t, ta, tb, dpy].
                'haldane': haldane's hamiltonian.
                    hopping_param = [t1, t2, dphi].
                'graphene': similar to haldane but with real hoppings.
                    hopping_param = [t1, t2].
                'hubbard_chain': spinful 1D chain with Hubbard
                    interactions.
                    hopping_param = [t].
        """
        # Assert hoppings are valid.
        hoppings_type_list = ['juanjo', 'haldane', 'graphene']
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
            ts = t2*np.exp(1j*dphi*np.pi)
            amp_first_neigh_B = np.full(3, t1, np.float64)
            amp_first_neigh_A = np.full(3, t1, np.float64)
            amp_second_neigh_B = np.array([ts, np.conj(ts), ts,
                                           np.conj(ts), ts, np.conj(ts)])
            amp_second_neigh_A = np.conj(amp_second_neigh_B)

        elif hoppings_type == 'graphene':
            t1, t2 = hopping_params
            amp_first_neigh_B = np.full(3, t1, np.float64)
            amp_first_neigh_A = amp_first_neigh_B
            amp_second_neigh_B = np.full(6, t2, np.float64)
            amp_second_neigh_A = amp_second_neigh_B

        # Add the hoppings to J.
        for i in range(self.lattice.L):
            if i%2 == 1:
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

        # Expand into two spin populations.
        self.J = np.kron(np.eye(2), self.J)
        return

    def add_hopping(self, i, delta, amp):
        """Add a hopping to the hopping matrix J.

        Args:
            i (int): index of the initial position of the hopping.
            delta (1darray of ints): hopping direction in lattice
                coordinates.
            amp (complex of float): hopping amplitude.

        """
        # Initial and final coordinates.
        ic = self.lattice.index_to_position(i)
        jc = ic + delta
        j = self.lattice.position_to_index(jc)

        not_crosses_boundary = True
        if isinstance(self.lattice, OpenLattice):
            Nx = self.lattice.Nx
            Ny = self.lattice.Ny
            not_crosses_boundary = ((0 <= jc[0] <= Nx) and (0 <= jc[1] <= Ny)
                                    and (0 <= j < self.lattice.L))
        elif isinstance(self.lattice, CircularLattice):
            if j < 0:
                not_crosses_boundary = False

        if not_crosses_boundary:
            self.J[j, i] += amp

        return
