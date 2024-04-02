import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os

# load connectivity for one subject
conn = loadmat(os.path.join("scripts", "plots", "example_connectivity.mat"))['Z']
conn1 = conn[:,:,10]
conn2 = conn[:,:,13]
np.fill_diagonal(conn1, 0)
np.fill_diagonal(conn2, 0)

# plot the matrix
fig, ax = plt.subplots()
ax.imshow(conn1, cmap='cividis', interpolation='none')
# remove x and y axis
ax.set_xticks([])
ax.set_yticks([])
ax.yaxis.set_visible(False)
ax.xaxis.set_visible(False)
# remove all spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
# save
fig.savefig(os.path.join('illustrations', 'example_connectivity_matrix1.pdf'))

# plot the matrix
fig, ax = plt.subplots()
ax.imshow(conn2, cmap='cividis', interpolation='none')
# remove x and y axis
ax.set_xticks([])
ax.set_yticks([])
ax.yaxis.set_visible(False)
ax.xaxis.set_visible(False)
# remove all spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
# save
fig.savefig(os.path.join('illustrations', 'example_connectivity_matrix2.pdf'))
