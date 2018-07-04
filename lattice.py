"""Define the Lattice class."""

import numpy as np


class OpenLattice(object):
    """Honeycomb lattice with open boundary conditions.

    Attributes:
        Nx, Ny (int): length and width in unit cells of the lattice.
        Np (int of float): number of particles in the lattice.
        L (int): total number of sites/lattice length.
        coords_pts (2darray of floats): coordinates of every point.
        first_neigh_A, first_neigh_B (list of 1darrays of ints):
            directions of the hopping amplitudes to first neighbors.
        second_neigh (list of 1darrays of ints): directions of the
            hopping amplitudes to second neighbors.

    """

    def __init__(self, Nx, Ny, Np):
        """Initialize class."""
        self.Nx = Nx
        self.Ny = Ny
        self.Np = Np

        self.L = 2*(Nx+1)*(Ny+1)-2
        self.coords_pts = np.zeros((self.L, 2), np.float64)
        for i in range(self.L):
            coords = self.index_to_position(i)
            self.coords_pts[i, 0] = 3/2*(coords[0] + coords[1]) + coords[2]
            self.coords_pts[i, 1] = np.sqrt(3)/2*(- coords[0] + coords[1])

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
            i (int): index of the position.

        Examples:
            >>> lat = OpenLattice(3, 3, 18)
            >>> lat.position_to_index([0, 0, 0])
            -1
            >>> lat.position_to_index([0, 0, 1])
            0
            >>> lat.position_to_index([1, 0, 0])
            1
            >>> lat.position_to_index([2, 0, 1])
            4
            >>> lat.position_to_index([3, 1, 0])
            13
            >>> lat.position_to_index([1, 2, 1])
            18
            >>> lat.position_to_index([1, 3, 0])
            25
            >>> lat.position_to_index([3, 3, 0])
            29
            >>> lat.position_to_index([3, 3, 1])
            30
        """
        x, y, n = pos
        return 2*x + n + 2*(self.Nx+1)*y-1

    def index_to_position(self, i):
        """Return a list with the coordinates of a position.

        Args:
            i (int): index.

        Returns:
            (1darray of ints): [x, y, n], where:
                x, y: coordinates of the unit cell.
                n: position within the unit cell: 0 for A, 1 for B.

        Examples:
            >>> lat = OpenLattice(3, 3, 18)
            >>> lat.index_to_position(0)
            array([0, 0, 1])
            >>> lat.index_to_position(1)
            array([1, 0, 0])
            >>> lat.index_to_position(4)
            array([2, 0, 1])
            >>> lat.index_to_position(13)
            array([3, 1, 0])
            >>> lat.index_to_position(18)
            array([1, 2, 1])
            >>> lat.index_to_position(25)
            array([1, 3, 0])
            >>> lat.index_to_position(29)
            array([3, 3, 0])
        """
        i += 1
        n = i%2
        i >>= 1
        x = i%(self.Ny+1)
        y = i//(self.Ny+1)

        return np.array([x, y, n])


class PeriodicLattice(object):
    """Define a honeycomb lattice with periodic boundary conditions.

    Attributes:
        Nx, Ny (int): length and width in unit cells of the lattice.
        Np (int of float): number of particles in the lattice.

        N (int): number of unit cells.
        L (int): total number of sites/lattice length.

        l_points (1darray of ints): indices of every spinless site.
        L_points (1darray of ints): indices of every point in the
            lattice with spin.
        unit_cells (1darray of ints): indices of every unit cell's
            starting point.
    """

    def __init__(self, Nx, Ny, Np, has_pbc=True):
        """Initialize class."""
        self.Nx = Nx
        self.Ny = Ny
        self.has_pbc = has_pbc
        self.N = Nx*Ny
        self.L = 4*self.N
        self.Np = Np

        self.L_points = [i for i in range(self.L)]
        self.unit_cells = [2*i for i in range(self.N)]

        # List with the vectors connecting the first and second neighbors.
        self.first_neigh_A = np.array([[1, -1, -1, 0], [0, 0, -1, 0],
                                       [1, 0, -1, 0]])
        self.first_neigh_B = np.array([[-1, 1, 1, 0], [0, 0, 1, 0],
                                       [-1, 0, 1, 0]])
        # The second neighbor vectors are equal for sublattices A and B.
        self.second_neigh = np.array([[-1, 1, 0, 0], [0, 1, 0, 0],
                                      [1, 0, 0, 0], [1, -1, 0, 0],
                                      [0, -1, 0, 0], [-1, 0, 0, 0]])

    def position_to_index(self, pos):
        """Return the index of a position [x, y, z, s].

        Args:
            pos: 1darray with the coordinates: [x, y, n, s]. The spin
                may be excluded from the list and defaults to s=0.
                x, y (int): coordinates of the unit cell.
                n (int): position within the unit cell: 0 for B, 1 for
                    A.
                s (int): spin of the position = {0, 1}.

        Returns:
            i (int): index of the position.
        """
        x = pos[0]%self.Nx
        y = pos[1]%self.Ny
        n = pos[2]%2
        s = pos[3]

        return n + 2*(x + self.Nx*y) + 2*self.N*s

    def index_to_position(self, i):
        """Return a list with the coordinates of a position.

        Args:
            i (int): index.

        Returns:
            (1darray of ints): [x, y, n, s], where:
                x, y: coordinates of the unit cell.
                n: position within the unit cell: 0 for B, 1 for A.
                s: spin of the position = {0, 1}.
        """
        # Spin.
        s = 0 if i < 2*self.N else 1
        i //= 2*self.N
        # Position within the unit cell.
        n = i%2
        i >>= 1
        # Coordinates of the unit cell.
        x = i%self.Nx
        y = (i-x)//self.Nx

        return np.array([x, y, n, s])
