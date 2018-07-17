"""Different plot scripts for the Honeycomb lattice."""

import numpy as np
from scipy.spatial import Voronoi
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt


def plot_voronoi(lattice, data, colormap='binary', figsize=(20, 12),
                 alpha=1, binary_minmax_data=False):
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
    if binary_minmax_data:
        norm = mpl.colors.Normalize(vmin=0, vmax=1, clip=True)
    else:
        norm = mpl.colors.Normalize(vmin=np.min(data), vmax=np.max(data),
                                    clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=colormap)

    fill_coords = np.zeros((lattice.L, 3, 2), np.float64)
    disp_A = np.array([[-1, 0], [1/2, np.sqrt(3)/2], [1/2, -np.sqrt(3)/2]])
    disp_B = np.array([[1, 0], [-1/2, np.sqrt(3)/2], [-1/2, -np.sqrt(3)/2]])
    for i in range(lattice.L):
        if lattice.lat_coords[i, 2] == 0:
            fill_coords[i] = lattice.xy_coords[i] + disp_A
        else:
            fill_coords[i] = lattice.xy_coords[i] + disp_B
        plt.fill(fill_coords[i, :, 0], fill_coords[i, :, 1],
                 color=mapper.to_rgba(data[i]), alpha=alpha)
    return


def plot_voronoi_2(lattice, data, colormap='Blues', figsize=(20, 12),
                   alpha=1, binary_minmax_data=False):
    """Plot with the help of scipy's Voronoi."""
    vor = Voronoi(lattice.xy_coords)
    if binary_minmax_data:
        norm = mpl.colors.Normalize(vmin=0, vmax=1, clip=True)
    else:
        norm = mpl.colors.Normalize(vmin=np.min(data), vmax=np.max(data),
                                    clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=colormap)
    for r in range(len(vor.point_region)):
        region = vor.regions[vor.point_region[r]]
        if not np.any(np.isclose(-1, region)):
            polygon = [vor.vertices[i] for i in region]
            plt.fill(*zip(*polygon), color=mapper.to_rgba(data[r]),
                     alpha=alpha)
    return
