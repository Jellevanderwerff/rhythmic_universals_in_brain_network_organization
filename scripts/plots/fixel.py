import pandas as pd
import os
import scipy.io
import numpy as np
from nilearn import plotting, datasets, image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# Load coordinates
subcortical_dk = pd.read_csv(os.path.join("data", "brain", "mappings", "cortical_subcortical_coords.csv"))
coords = list(zip(subcortical_dk['X'], subcortical_dk['Y'], subcortical_dk['Z']))
labels = subcortical_dk['Label']

# Load data
data = scipy.io.loadmat(os.path.join("data", "brain", "fixel", "cortical_thalamus_striatum_network.mat"))['connectivityMatrix']

# Get indices of rows that are not all zeros
non_zero_rows = np.where(np.any(data, axis=1))[0]
data = data[non_zero_rows, :]
data = data[:, non_zero_rows]
coords = [coords[i] for i in non_zero_rows]
labels = [labels[i] for i in non_zero_rows]

cmap = LinearSegmentedColormap.from_list('mycmap', ['#B59B77', '#B59B77'])

fig, ax = plt.subplots(figsize=(4, 4))
plotting.plot_connectome(data, coords, edge_vmin=0, edge_vmax=1, node_color='#B59B77', node_size=100, display_mode='y', edge_cmap=cmap, axes=ax)
fig.savefig(os.path.join("plots", "fixel", "cortical_thalamus_striatum_network.pdf"))
