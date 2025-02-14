import numpy as np
import pandas as pd
import scipy.io
import plotly.graph_objects as go
import webbrowser
from pathlib import Path
import os

# Get the project root directory
project_root = Path(__file__).resolve().parents[2]
data_path = project_root / 'data' / 'brain' / 'network_assignment'
plots_path = project_root / 'plots' / 'connectomes'

# Load the matrix data
# Entropy difference: AvAdjacencyMat.rsFC.negative.EntropyDiff.mat
# G_response: AvAdjacencyMat.rsFC.negative.Gresponse.mat
# Binary_or_ternary: AvAdjacencyMat.rsFC.negative.binary_or_ternary_introduced.mat

mat_data = scipy.io.loadmat(data_path / 'AvAdjacencyMat.rsFC.negative.binary_or_ternary_introduced.mat')
matrix = mat_data['neg_mask_true']

# Load the csv data
csv_data = pd.read_csv(project_root / 'data' / 'brain' / 'mappings' / 'coordinates.csv')
coords = csv_data[['X', 'Y', 'Z']].values
labels = csv_data['Label'].tolist()

# Extract pairs of connected nodes from the matrix
edges = []
connected_nodes = set()
for i in range(matrix.shape[0]):
    for j in range(i+1, matrix.shape[1]):
        if matrix[i, j] == 1:
            edges.append((i, j))
            connected_nodes.add(i)
            connected_nodes.add(j)

# Filter the coordinates and labels for nodes that are connected
filtered_coords = coords[list(connected_nodes)]
filtered_labels = [labels[i] for i in connected_nodes]

# Create traces for nodes and edges
node_trace = go.Scatter3d(x=filtered_coords[:, 0],
                          y=filtered_coords[:, 1],
                          z=filtered_coords[:, 2],
                          mode='markers+text',
                          marker=dict(size=5, color='blue'),
                          text=filtered_labels,
                          textposition="top center")

edges_x = []
edges_y = []
edges_z = []
for i, j in edges:
    edges_x += [coords[i, 0], coords[j, 0], None]
    edges_y += [coords[i, 1], coords[j, 1], None]
    edges_z += [coords[i, 2], coords[j, 2], None]

edge_trace = go.Scatter3d(x=edges_x, y=edges_y, z=edges_z, mode='lines', line=dict(color='red', width=2))

layout = go.Layout(scene=dict(aspectmode='cube',
                              xaxis=dict(title='X'),
                              yaxis=dict(title='Y'),
                              zaxis=dict(title='Z')))

# Combine traces and save to an HTML file
fig = go.Figure(data=[edge_trace, node_trace], layout=layout)
fig.write_html(os.path.join(plots_path, "connectome_plot.html"))

# Automatically open the saved HTML file in the default web browser
webbrowser.open(os.path.join(plots_path, 'connectome_plot.html'), new=2)
