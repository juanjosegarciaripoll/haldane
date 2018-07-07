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
            np.sqrt(3)/2*(-self.lat_coords[:, 0] + self.lat_coords[:, 1])
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
            i (int): index of the position, -1 if position is not in
                the lattice.

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


class CircularLattice(object):
    """Circular honeycomb lattice with open boundary conditions.

    Attributes:
        r (int): distance in lattice units from the center to the
            furthest lattice point.
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

    def __init__(self, r):
        """Initialize class."""
        self.r = r

        # Start with a very big diamond lattice to prune.
        big_lat = OpenLattice(40, 40)
        center = (big_lat.xy_coords[0] + big_lat.xy_coords[-1])/2
        ix_pts = np.nonzero(
            np.linalg.norm(big_lat.xy_coords - center, axis=1) < self.r
            )[0]
        self.L = np.size(ix_pts)
        self.xy_coords = big_lat.xy_coords[ix_pts]
        self.lat_coords = big_lat.lat_coords[ix_pts]

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
            i (int): index of the position, -1 if position is not in
                the lattice.

        """
        tmp = np.linalg.norm(self.lat_coords - pos, axis=1)
        if np.any(np.isclose(tmp, 0)):
            return np.argmin(tmp)
        else:
            return -1


class SquareLattice(object):
    """Square lattice.

    Attributes:
        Nx, Ny (int): length and width in unit cells of the lattice.
        L (int): total number of sites/lattice length.
        lat_coords (2darray of ints): lattice coordinates of every
            point (x, y posititions and displacement inside the unit
            cell).

    """

    def __init__(self, Nx, Ny):
        """Initialize class."""
        self.Nx = Nx
        self.Ny = Ny

        self.L = Nx*Ny
        self.lat_coords = np.array(
            [np.kron(np.ones(self.Ny, np.int64), np.arange(self.Nx)),
             np.kron(np.arange(self.Ny), np.ones(self.Nx, np.int64)),
             ]).T
        self.xy_coords = np.array([
            2*np.pi/3*(self.lat_coords[:, 0] + self.lat_coords[:, 1]),
            2*np.pi/np.sqrt(3)*(-self.lat_coords[:, 0] + self.lat_coords[:, 1])
        ]).T
