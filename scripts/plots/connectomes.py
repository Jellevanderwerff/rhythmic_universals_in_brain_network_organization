import pandas as pd
import os
import scipy.io
import numpy as np
from nilearn import plotting
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

# Load coordinates
coords = pd.read_csv(os.path.join("data", "brain", "mappings", "coordinates.csv"))
coords = list(zip(coords["X"], coords["Y"], coords["Z"]))

colors = {
    "G_weighted.mat": [(207/255, 153/255, 147/255, 0.5), (207/255, 153/255, 147/255, 0.5)],
    "binary_ternary_weighted.mat": [(33/255, 63/255, 82/255, 0.3), (33/255, 63/255, 82/255, 1)],
    "entropy_weighted.mat": [(119/255, 144/255, 160/255, 0.3), (119/255, 144/255, 160/255, 1)],
    "binary_ternary_struct_weighted.mat": [(33/255, 63/255, 82/255, 0.3), (33/255, 63/255, 82/255, 1)],
}


for file in (
    "G_weighted.mat",
    "binary_ternary_weighted.mat",
    "entropy_weighted.mat"
):
    print(file)
    # Copy coords
    current_coords = coords.copy()
    # Load data
    data = scipy.io.loadmat(os.path.join("data", "brain", "adjacency", file))
    # Get adjacency matrix
    adj = data[list(data.keys())[-1]]
    # Change diagonal to zeros
    np.fill_diagonal(adj, 0)
    # Get indices of columns that are not all zeros
    indices = np.where(~np.all(adj == 0, axis=0))[0]
    # Remove rows and columns that are all zeros
    adj = adj[indices][:, indices]
    # Remove coordinates that are not in the adjacency matrix
    current_coords = [current_coords[i] for i in indices]
    # node_sizes
    node_sizes = np.sum(np.abs(adj), axis=1)
    # normalize node_sizes
    node_sizes = (node_sizes - np.min(node_sizes) + 0.01) / (np.max(node_sizes) - np.min(node_sizes)) * 250

    # Make colourmap
    cmap = LinearSegmentedColormap.from_list('custom', colors[file])

    # Plot connectome
    for colorbar in (True, False):
        fig, ax = plt.subplots(figsize=(4, 4))
        plotting.plot_connectome(
            adj,
            current_coords,
            colorbar=colorbar,
            node_color=colors[file][1][:-1],
            edge_cmap=cmap,
            edge_vmin = None,
            edge_vmax = None,
            node_size=node_sizes,
            axes=ax,
            display_mode="z",
        )

        # Save
        filename = f'{file[:-4]}_withcolorbar.pdf' if colorbar else f'{file[:-4]}_nocolorbar.pdf'
        fig.savefig(os.path.join('plots', 'connectomes', filename))
