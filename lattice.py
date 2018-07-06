"""Define the Lattice class."""

import numpy as np


class OpenLattice(object):
    """Honeycomb lattice with open boundary conditions.

    Attributes:
        Nx, Ny (int): length and width in unit cells of the lattice.
        L (int): total number of sites/lattice length.
        lat_coords (2darray of ints): lattice coordinates of every
            point (x, y posititions and displacement inside the unit
            cell).
        xy_coords (2darray of floats): real space coordinates of every
            point.
        first_neigh_A, first_neigh_B (list of 1darrays of ints):
            directions of the hopping amplitudes to first neighbors.
        second_neigh (list of 1darrays of ints): directions of the
            hopping amplitudes to second neighbors.

    """

    def __init__(self, Nx, Ny):
        """Initialize class."""
        self.Nx = Nx
        self.Ny = Ny

        self.L = 2*Nx*Ny
        self.lat_coords = np.array(
            [np.kron(np.ones(self.Ny, np.int64),
                     np.kron(np.arange(self.Nx), np.ones(2, np.int64))),
             np.kron(np.arange(self.Ny), np.ones(2*self.Nx, np.int64)),
             np.kron(np.ones(self.Nx*self.Ny, np.int64),
                     np.array([0, 1], np.int64))
             ], np.int64).T
        self.xy_coords = np.array([
            (3/2*(self.lat_coords[:, 0] + self.lat_coords[:, 1])
             + self.lat_coords[:, 2]),
            -np.sqrt(3)/2*(-self.lat_coords[:, 0] + self.lat_coords[:, 1])
        ]).T

        self.first_neigh_A = np.array([[0, 0, 1], [-1, 0, 1], [0, -1, 1]])
        self.first_neigh_B = np.array([[0, 0, -1], [1, 0, -1], [0, 1, -1]])
        self.second_neigh = np.array([[0, 1, 0], [1, 0, 0], [1, -1, 0],
                                      [0, -1, 0], [-1, 0, 0], [-1, 1, 0]])

    def position_to_index(self, pos):
        """Return the index of a position [x, y, z].

        Args:
            pos: 1darray with the coordinates: [x, y, n].
                x, y (int): coordinates of the unit cell.
                n (int): position within unit cell: 0 for A, 1 for B.

        Returns:
            i (int): index of the position.

        Examples:
            >>> lat = OpenLattice(4, 4)
            >>> lat.position_to_index([0, 0, 0])
            0
            >>> lat.position_to_index([0, 0, 1])
            1
            >>> lat.position_to_index([1, 0, 0])
            2
            >>> lat.position_to_index([2, 0, 1])
            5
            >>> lat.position_to_index([3, 1, 0])
            14
            >>> lat.position_to_index([1, 2, 1])
            19
            >>> lat.position_to_index([1, 3, 0])
            26
            >>> lat.position_to_index([3, 3, 0])
            30
            >>> lat.position_to_index([3, 3, 1])
            31
            >>> lat.position_to_index([-3, 3, 1])
            -1
            >>> lat.position_to_index([5, 2, 1])
            -1
        """
        tmp = np.linalg.norm(self.lat_coords - pos, axis=1)
        if np.any(np.isclose(tmp, 0)):
            return np.argmin(tmp)
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

        Examples:
            >>> lat = OpenLattice(4, 4)
            >>> lat.index_to_position(1)
            array([0, 0, 1])
            >>> lat.index_to_position(2)
            array([1, 0, 0])
            >>> lat.index_to_position(5)
            array([2, 0, 1])
            >>> lat.index_to_position(14)
            array([3, 1, 0])
            >>> lat.index_to_position(19)
            array([1, 2, 1])
            >>> lat.index_to_position(26)
            array([1, 3, 0])
            >>> lat.index_to_position(30)
            array([3, 3, 0])
        """
        return self.lat_coords[i]
