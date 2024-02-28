import pandas as pd
import matplotlib.pyplot as plt
from mne.viz import plot_connectivity_circle
import numpy as np
import scipy.io
import os

# Load the data
conmat = scipy.io.loadmat(os.path.join('data', 'brain', 'AvAdjacencyMat.rsFC.negative.binary_or_ternary_introduced.weighted.mat'))['neg_averageValues']
# remove nan values from 2D array
conmat[np.isnan(conmat)] = 0

labels = [f'Node {i}' for i in range(1, conmat.shape[0] + 1)]

# Create a graph
G = plot_connectivity_circle(conmat, node_names=labels, colorbar=True, facecolor='white', textcolor='black')
plt.show()
