"""Main script."""

import numpy as np
from scipy.linalg import eigh

from lattice import OpenLattice
# from circular_lattice import CircularLattice
from hamiltonian import OpenHamiltonian
from harmonic_trap import build_harmonic_trap
from currents import compute_currents
from plot_lattice import plot_voronoi

if __name__ == '__main__':

    # 1. Parameters.
    Nx = Ny = 25
    r = 25
    Np = int(Nx*Ny*0.8)
    lattice = OpenLattice(Nx, Ny, Np)
    # lattice = CircularLattice(r, Np)
    t = 1
    ta = 0.1
    tb = -ta
    dpy = 0.5
    hopping_params = [t, ta, tb, dpy]
    hoppings_type = 'juanjo'
    single_levels = build_harmonic_trap(lattice, 1/30)
    ham = OpenHamiltonian(lattice, single_levels,
                          hopping_params, hoppings_type)
    print('Number of sites = {}'.format(lattice.L))
    print('Number of particles = {}'.format(lattice.Np))

    # 2. Diagonalize Hamiltonian.
    w, v = eigh(ham.J)
    v = v[:, :Np]
    occupations = np.linalg.norm(v, axis=1)
    plot_voronoi(lattice, occupations, do_plot=False)

    # 3. Compute currents.
    currents = compute_currents(lattice, ham.J, v)
    plot_voronoi(lattice, currents, colormap='bwr')
