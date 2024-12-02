import numpy as np
import pandas as pd
import scipy.io
from nilearn import plotting
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

# Get the project root directory
project_root = Path(__file__).resolve().parents[2]
data_path = project_root / 'data' / 'network_assignment'

# Load the matrix data: Gresponse('AvAdjacencyMat.rsFC.negative.Gresponse.mat'); Entropy difference ('AvAdjacencyMat.rsFC.negative.EntropyDiff.mat'); binary or ternary ('AvAdjacencyMat.rsFC.negative.binary_or_ternary_introduced.mat')
mat_data = scipy.io.loadmat(data_path / 'AvAdjacencyMat.rsFC.negative.Gresponse.mat')
matrix = mat_data['neg_mask_true']

# Load the coordinates csv data
coords_data = pd.read_csv(project_root / 'scripts' / 'plots' / 'coordinates.csv')
coords = coords_data[['X', 'Y', 'Z']].values

# Load node RGB colors from the new CSV file and convert to hex format: nodes can be colored according to:
# 1) Yeo atlas in Freesurfer (sortedDataTable_colors_yeoNet.csv)
# 2) Lobe (sortedDataTable_colors_lobe.csv)
colors_data = pd.read_csv(data_path / 'sortedDataTable_colors_shirer.csv')
colors = colors_data[['R', 'G', 'B']].values

def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))

node_colors_hex = [rgb_to_hex(rgb) for rgb in colors]

# Calculate the degree for each node
node_degrees = np.sum(matrix, axis=1)

# For nodes with degree 0, set their size to 0
node_sizes = np.where(node_degrees == 0, 0, node_degrees)

# Normalize and scale the node sizes of remaining nodes
min_size = 20  # minimum node size
max_size = 100  # maximum node size

# Only scale nodes with a non-zero degree
non_zero_degrees = node_degrees[node_degrees > 0]
scaled_node_sizes = min_size + (non_zero_degrees - np.min(non_zero_degrees)) * (max_size - min_size) / (np.max(non_zero_degrees) - np.min(non_zero_degrees))
node_sizes[node_degrees > 0] = scaled_node_sizes

# Adjust colormap so that 1 is mapped to dark gray
colors = [(0.3, 0.3, 0.3), (0.3, 0.3, 0.3)]  # RGB for dark gray
cm = mcolors.LinearSegmentedColormap.from_list("custom", colors, N=2)

# Create a 3D plot with a transparent MNI brain
display = plotting.view_connectome(adjacency_matrix=matrix,
                                   node_coords=coords,
                                   edge_threshold="90%",
                                   node_size=node_sizes.tolist(),
                                   node_color=node_colors_hex,
                                   edge_cmap=cm)

# Display the visualization in a web browser
display.open_in_browser()




