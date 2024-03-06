import pandas as pd
import os
import scipy.io
import numpy as np
from nilearn import plotting
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Load coordinates
subcortical_dk = scipy.io.loadmat(os.path.join("data", "brain", "mappings", "dk_subcortical_84.mat"))
node_names = subcortical_dk['namesNodes']


# Load data
data = scipy.io.loadmat(os.path.join("data", "brain", "fixel", "cortical_thalamal_striatal_network.mat"))
data = data[list(data.keys())[-1]]




"""
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
node_sizes = (node_sizes - np.min(node_sizes)) / (np.max(node_sizes) - np.min(node_sizes)) * 250

fig, ax = plt.subplots(figsize=(4, 4))

# Plot connectome
plotting.plot_connectome(
    adj,
    current_coords,
    colorbar=True,
    node_color="black",
    edge_cmap='viridis' if not 'struct' in file else 'cividis',
    edge_vmin=None if not 'struct' in file else 0,
    edge_vmax=None if not 'struct' in file else np.max(adj),
    node_size=node_sizes,
    axes=ax,
    display_mode="z",
)

# Save
fig.savefig(os.path.join('plots', 'connectomes', f'{file[:-4]}_withcolorbar.pdf'))
fig.savefig(os.path.join('plots', 'connectomes', f'{file[:-4]}_withcolorbar.png'), dpi=600)
"""