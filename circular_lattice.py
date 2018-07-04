"""Define a lattice class with the shape of a circle."""

import numpy as np

from lattice import OpenLattice


class CircularLattice(object):
    """Circular honeycomb lattice with open boundary conditions.

    Attributes:
        r (int): maximum distance allowed to the center of the lattice.
        Np (int of float): number of particles in the lattice.
        L (int): total number of sites/lattice length.
        coords_pts (2darray of floats): coordinates of every point.
        xyn_coords_pts (2darray of ints): coordinates of every point in
            lattice units.
        first_neigh_A, first_neigh_B (list of 1darrays of ints):
            directions of the hopping amplitudes to first neighbors.
        second_neigh (list of 1darrays of ints): directions of the
            hopping amplitudes to second neighbors.

    """

    def __init__(self, r, Np):
        """Initialize class."""
        self.r = r
        self.Np = Np

        # Create a really big lattice.
        tmp_lattice = OpenLattice(50, 50, 0)
        # Center of the lattice.
        center = (tmp_lattice.coords_pts[0] + tmp_lattice.coords_pts[-1])/2
        # Indices of the points from tmp_lattice that will be in our
        # circular lattice.
        tmp_ix_pts = np.nonzero(
            np.linalg.norm(tmp_lattice.coords_pts - center, axis=1) < r
            )[0]

        self.L = len(tmp_ix_pts)
        self.coords_pts = tmp_lattice.coords_pts[tmp_ix_pts]
        tmp_xyn_coords = []
        for i in tmp_ix_pts:
            tmp_xyn_coords.append(tmp_lattice.index_to_position(i))
        self.xyn_coords_pts = np.array(tmp_xyn_coords)

        self.first_neigh_A = np.array([[0, -1, 1], [-1, 0, 1], [0, 0, 1]])
        self.first_neigh_B = np.array([[0, 1, -1], [1, 0, -1], [0, 0, -1]])
        self.second_neigh = np.array([[-1, 1, 0], [0, 1, 0], [1, 0, 0],
                                      [1, -1, 0], [0, -1, 0], [-1, 0, 0]])

    def position_to_index(self, pos):
        """Return the index of a position [x, y, z].

        Args:
            pos: 1darray with the coordinates: [x, y, n].
                x, y (int): coordinates of the unit cell.
                n (int): position within unit cell: 0 for A, 1 for B.

        Returns:
            (int): index of the position.

        """
        tmp = np.linalg.norm(self.xyn_coords_pts-pos, axis=1)
        if np.any(np.isclose(tmp, 0)):
            return np.nonzero(tmp == 0)[0][0]
        else:
            return -1

    def index_to_position(self, i):
        """Return a list with the coordinates of a position.

        Args:
            i (int): index.

        Returns:
            (1darray of ints): [x, y, n], where:
                x, y: coordinates of the unit cell.
                n: position within the unit cell: 0 for A, 1 for B.

        """
        return self.xyn_coords_pts[i]
