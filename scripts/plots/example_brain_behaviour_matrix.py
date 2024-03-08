import os
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

cmap = LinearSegmentedColormap.from_list('mycmap', ['#0028D0', '#FE2E13'])

# generate random 6x6 matrix
matrix = np.random.default_rng(123).uniform(-1, 1, (6, 6))

# plot the matrix
fig, ax = plt.subplots()
ax.imshow(matrix, cmap=cmap, interpolation='none')
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
fig.savefig(os.path.join('illustrations', 'example_brain_behavior_matrix.pdf'))
