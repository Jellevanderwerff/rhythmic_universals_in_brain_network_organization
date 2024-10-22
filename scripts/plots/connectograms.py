import pandas as pd
import matplotlib.pyplot as plt
from mne.viz.circle import _plot_connectivity_circle as plot_connectivity_circle
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import scipy.io
import os

"""
Note that this script is not completely reproducible for the moment. In the source of the MNE package I change the con_thresh to 0.1 so it didn't
plot white connections. For some reason the function does not allow np.nan
"""

# Matplotlib parameters
plt.rcParams["font.family"] = "Helvetica"

# Define labels and colors
labels = [
    "Visual",
    "Motor",
    "Dorsal attention",
    "Ventral attention",
    "Limbic",
    "Auditory",
    "Default-mode",
]
colors = [
    (120.003, 18.003, 134.0025),
    (69.9975, 129.999, 180.00449999999998),
    (0.0, 117.9885, 13.9995),
    (230.01000000000002,148.002, 33.9915),
    (219.98850000000002, 247.9875, 163.9905),
    (195.993, 58.0125, 250.002),
    (204.9945, 61.990500000000004, 78.00450000000001),
]
colors = np.array(colors) / 255


for file in ('binary_ternary_ratios.mat', 'entropy_difference.mat', 'gramm_redundancy.mat'):
    # Load data
    data = scipy.io.loadmat(
        os.path.join(
            "data",
            "brain",
            "between_network_connections",
            file
        )
    )[file[:-4]]

    # Plot connectogram
    fig = plt.figure(figsize=(3, 3), tight_layout=True)
    ax = fig.add_subplot(111, polar=True)
    plot_connectivity_circle(
        data, labels, facecolor="white", textcolor="black", colorbar=False, node_colors=colors, node_edgecolor=None, show=False, ax=ax,
        linewidth = 5, fontsize_names=9, node_height=3, padding=10
    )

    # Fix upside down text
    n_nodes = len(labels)
    node_angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    angles_deg = 180 * node_angles / np.pi

    for angle_deg, text in zip(angles_deg, ax.texts):
        if angle_deg >= 270 or angle_deg < 90:
            ha = "left"
        else:
            # Flip the label, so text is always upright
            angle_deg += 180
            ha = "right"

        text.set_rotation(angle_deg)
        text.set_horizontalalignment(ha)


    # Save
    fig.savefig(os.path.join('plots', 'connectograms', f'{file[:-4]}.pdf'))
