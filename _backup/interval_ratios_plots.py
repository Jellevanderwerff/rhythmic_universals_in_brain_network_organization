import os

from thebeat import Sequence
from thebeat.visualization import plot_interval_ratios_density
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal

# Load df
df = pd.read_csv('ITIs.csv')

"""
PER PARTICIPANT
"""
# Create lists of Sequences for each pp
list_of_lists = []

# Make a list of Sequences for each pp
for pp_id in df.pp_id.unique():
    df_piece = df[df['pp_id'] == pp_id]
    pp_seqs = []
    for id in df_piece.sequence_id.unique():
        seq = Sequence(df_piece[df_piece['sequence_id'] == id].resp_iti.values)
        pp_seqs.append(seq)
    list_of_lists.append(pp_seqs)

# Make empty grid
n_pps = len(list_of_lists)
with plt.style.context('seaborn'):
    fig, axs = plt.subplots(ncols=4,
                            nrows=np.ceil(n_pps / 4).astype(int),
                            tight_layout=True,
                            dpi=600)

# Empty list for collecting the density peaks
pp_peaks = []

count = 0

# Plot, get peaks, and plot text for each participant
for pp in df.pp_id.unique():
    pp_list = list_of_lists[count]
    fig, ax = plot_interval_ratios_density(pp_list, ax=axs[count // 4, count % 4],
                                           x_axis_label=None, y_axis_label=None,
                                           title="pp " + str(pp))

    y = ax.get_lines()[0].get_ydata()

    # Get local maxima
    peaks = scipy.signal.find_peaks(y)[0]
    pp_peaks.append(list(peaks))

    # Add one to the count
    count += 1

# Remove empty axes if number of plots is odd
if n_pps % 2 == 1:
    fig.delaxes(axs[-1, -1])

fig.suptitle('Participant response interval ratios')
fig.supxlabel('Dyadic interval ratio')
fig.supylabel('Probability density')
fig.show()
fig.savefig(os.path.join('plots', 'interval_ratios_bypp.png'))

# Print peaks for each pp
for pp in range(len(pp_peaks)):
    pp_id = df.pp_id.unique()[pp]
    peaks_ratios = list(np.array(pp_peaks[pp]) / 100)
    print("pp " + str(pp_id) + ": " + str(peaks_ratios))

"""
ALL PARTICIPANTS
"""
# Create Sequences for response inter-tap intervals
seqs_all = [Sequence(df[df['sequence_id'] == id].resp_iti.values)
            for id in df.sequence_id.unique()]

# Plot all participants together and print peaks
fig, ax = plot_interval_ratios_density(seqs_all, dpi=600, title="Dyadic interval ratios (response)")
fig.savefig(os.path.join('plots', 'interval_ratios_all.png'))
fig.show()
y = ax.get_lines()[0].get_ydata()

# Get local maxima
peaks = scipy.signal.find_peaks(y)[0]

# Print peaks
print(f"Peaks across participants: {peaks}")

"""
Theoretical ratios
"""

theoretical_ratios = {
    "1:1": 1 / 2,
    "1:2": 1 / 3,
    "2:1": 2 / 3,
    "1:3": 1 / 4,
    "3:1": 3 / 4,
    "1:4": 1 / 5,
    "4:1": 4 / 5,
    "2:3": 2 / 5,
    "3:2": 3 / 5,
    "3:4": 3 / 7,
    "4:3": 4 / 7,
}

# Print theoretical ratios
print(f"Theoretical ratios: ")

for k, v in theoretical_ratios.items():
    print(f"{k}: {v}")

"""
Beat and metre. 
Ratios were taken to normalize with respect to tempo and to compare structures (rather than absolute durations) across patterns. 
For each ratio distribution, we found the location of the maxima by taking the second derivative of the kernel density estimation (KDE) function. We then tested whether these fixed IOI relationships (the peaks in Fig. 2) coincided beyond chance with those expected theoretically. The most parsimonious way of generating a musical duration from another is to multiply or divide it by two, three or four. Hence we predicted that we would find, with high frequency, ratios of 1:1 (equal duration IOIs), 1:2, 1:3, 1:4, 2:3, 3:4, and their reciprocals, giving a total of 11 expected theoretical ratios. As the predicted ratios spanned 11 possible values, we extracted the 11 most frequent ratios from our empirical distributions. We then matched
the expected with the empirical ratios (with a 0.01 tolerance on ratio differences) and quantified the match using the Jaccard index25. Given two sets, the Jaccard index is calculated as the ratio between their union and their intersection, that is, the number of elements in common divided by the number of overall elements. Finally, we performed a Monte Carlo simulation with 1 million iterations to test whether the matching of the predicted and found peaks was attributable to chance. This provided a P′ value, calculated as the proportion of randomizations with an average Jaccard index greater than or equal to the empirical Jaccard index; that is, the relative number of cases for which a list of 11 random ratios has equal number or more matches with predicted ratios than the 11 empirical ratios.
"""
