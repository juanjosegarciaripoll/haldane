"""Test the building of the Haldane Hamiltonian."""

import unittest
import numpy as np

from lattice import OpenLattice
from hamiltonian import OpenHamiltonian


class HaldaneHamiltonianHoppingMatrixTestCase(unittest.TestCase):
    """Test the hopping matrix of Haldane's Hamiltonian."""

    def setUp(self):
        lattice = OpenLattice(4, 4, 0)
        # Hamiltonian parameters.
        t1 = 1
        t2 = 0.1
        dphi = 0.5
        hopping_params = (t1, t2, dphi)
        hoppings_type = 'haldane'
        trap_potential = 1
        lattice_imbalance = 1
        ham = OpenHamiltonian(lattice, hopping_params, hoppings_type,
                              trap_potential, lattice_imbalance)
        self.t1 = t1
        self.t2 = t2*np.exp(1j*dphi*np.pi)
        self.A = ham.A

    def test_on_site_terms(self):
        self.assertAlmostEqual(self.A[0, 0], 26)
        self.assertAlmostEqual(self.A[1, 1], 15)
        self.assertAlmostEqual(self.A[2, 2], 14)
        self.assertAlmostEqual(self.A[6, 6], 8)
        self.assertAlmostEqual(self.A[11, 11], 0)
        self.assertAlmostEqual(self.A[12, 12], 2)
        self.assertAlmostEqual(self.A[17, 17], 3)
        self.assertAlmostEqual(self.A[21, 21], 3)
        self.assertAlmostEqual(self.A[27, 27], 6)
        self.assertAlmostEqual(self.A[31, 31], 24)

    def test_first_neighbour_hoppings(self):
        self.assertAlmostEqual(self.A[0, 1], self.t1)
        self.assertAlmostEqual(self.A[1, 2], self.t1)
        self.assertAlmostEqual(self.A[10, 3], self.t1)
        self.assertAlmostEqual(self.A[10, 11], self.t1)
        self.assertAlmostEqual(self.A[9, 8], self.t1)
        self.assertAlmostEqual(self.A[8, 9], self.t1)
        self.assertAlmostEqual(self.A[19, 20], self.t1)
        self.assertAlmostEqual(self.A[26, 19], self.t1)
        self.assertAlmostEqual(self.A[30, 31], self.t1)

    def test_second_neighbour_hoppings(self):
        self.assertAlmostEqual(self.A[2, 0], np.conj(self.t2))
        self.assertAlmostEqual(self.A[0, 2], self.t2)
        self.assertAlmostEqual(self.A[14, 12], np.conj(self.t2))
        self.assertAlmostEqual(self.A[12, 14], self.t2)
        self.assertAlmostEqual(self.A[10, 18], np.conj(self.t2))
        self.assertAlmostEqual(self.A[18, 10], self.t2)
        self.assertAlmostEqual(self.A[19, 27], np.conj(self.t2))
        self.assertAlmostEqual(self.A[27, 19], self.t2)
        self.assertAlmostEqual(self.A[12, 6], np.conj(self.t2))
        self.assertAlmostEqual(self.A[6, 12], self.t2)
        self.assertAlmostEqual(self.A[19, 13], np.conj(self.t2))
        self.assertAlmostEqual(self.A[13, 19], self.t2)

    def test_zero_terms(self):
        """Test that the zeros in the hopping matrix are right."""
        self.assertAlmostEqual(
            np.linalg.norm(self.A[:, 0]),
            np.sqrt(26**2 + self.t1**2 + 2*np.abs(self.t2)**2)
            )
        self.assertAlmostEqual(
            np.linalg.norm(self.A[:, 1]),
            np.sqrt(15**2 + 3*self.t1**2 + 2*np.abs(self.t2)**2)
            )
        self.assertAlmostEqual(
            np.linalg.norm(self.A[:, 11]),
            np.sqrt(3*self.t1**2 + 6*np.abs(self.t2)**2)
            )
        self.assertAlmostEqual(
            np.linalg.norm(self.A[:, 17]),
            np.sqrt(3**2 + 3*self.t1**2 + 4*np.abs(self.t2)**2)
            )
        self.assertAlmostEqual(
            np.linalg.norm(self.A[:, 27]),
            np.sqrt(6**2 + 2*self.t1**2 + 4*np.abs(self.t2)**2)
            )
