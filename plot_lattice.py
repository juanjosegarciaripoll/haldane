"""Different plot scripts for the Honeycomb lattice."""

import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt


def plot_voronoi(lattice, data, colormap='binary', figsize=(20, 12)):
    """Plot data on a Voronoi diagram of the lattice.

    Args:
        lattice (OpenLattice): lattice on which we plot the data.
        data (1darray of floats): data to plot on the lattice points.
        colormap (str, optional): colormap of the plot.
        figsize (tuple of ints, opt): figsize of the plot.

    """
    plt.figure(figsize=figsize)

    if np.any(np.iscomplex(data)):
        data = np.real(data)
    norm = mpl.colors.Normalize(vmin=np.min(data), vmax=np.max(data),
                                clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=colormap)

    fill_coords = np.zeros((lattice.L, 3, 2), np.float64)
    disp_A = np.array([[-1, 0], [1/2, np.sqrt(3)/2], [1/2, -np.sqrt(3)/2]])
    disp_B = np.array([[1, 0], [-1/2, np.sqrt(3)/2], [-1/2, -np.sqrt(3)/2]])
    for i in range(lattice.L):
        if i%2 == 0:
            fill_coords[i] = lattice.xy_coords[i] + disp_A
        else:
            fill_coords[i] = lattice.xy_coords[i] + disp_B
        plt.fill(fill_coords[i, :, 0], fill_coords[i, :, 1],
                 color=mapper.to_rgba(data[i]))
    return
