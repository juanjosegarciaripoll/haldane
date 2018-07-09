"""Different plot scripts for the Honeycomb lattice."""

import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt


def plot_voronoi(lattice, data, colormap='Blues'):
    """Plot data on a Voronoi diagram of the lattice.

    Args:
        lattice (OpenLattice): lattice on which we plot the data.
        data (1darray of floats): data to plot on the lattice points.

    """
    plt.figure(figsize=(20, 12))

    vor = Voronoi(lattice.xy_coords)
    norm = mpl.colors.Normalize(vmin=np.min(data), vmax=np.max(data),
                                clip=True)
    if colormap == 'Blues':
        mapper = cm.ScalarMappable(norm=norm, cmap=cm.Blues)
    elif colormap == 'bwr':
        mapper = cm.ScalarMappable(norm=norm, cmap=cm.bwr)
    for r in range(len(vor.point_region)):
        region = vor.regions[vor.point_region[r]]
        if not np.any(np.isclose(-1, region)):
            polygon = [vor.vertices[i] for i in region]
            plt.fill(*zip(*polygon), color=mapper.to_rgba(data[r]))

    # plt.show()
    return
