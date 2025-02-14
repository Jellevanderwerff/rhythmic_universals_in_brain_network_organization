"""
# Grammatical complexity
Here we calculate the redundancy statistic G (Jamieson & Mewhort, 2005).

The steps are:

1. Use Silhouette scores to find the optimal number k of clusters for each participant in each
   condition. So, 4 number k of clusters per participant.
2. Do K-means clustering to find bins (for each participant and condition separately)
3. Calculate G for each participant
    - Loop over different conditions
    - Loop over participants
    - Discretize all the ITIs into k bins, so we have (in the case of
    k = 3) the 'symbols' A, B, and C
    - The 'grammar' of the participant is made up of all the different combinations encountered
    of those symbols, e.g. AB, BC, CA, etc.
    - Calculate the frequencies and the probabilities for those bigrams
    - Calculate G

"""

# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import scipy.stats
import os
import warnings
import string

# variables
K_MIN = 2       # minimum number of clusters to test for
K_MAX = 8      # maximum number of clusters to test for

#suppress warnings
warnings.filterwarnings("ignore")

# Load data
ITIs = pd.read_csv(os.path.join('data', 'experiment', 'processed', 'ITIs.csv'))

# create df for k-means clusters
df_clusters = pd.DataFrame()
# create df for silhouette scores
df_silhouette = pd.DataFrame()

# Loop over whether we're doing stimulus or response
for stim_resp in ('resp_iti', 'stim_ioi'):
    # Loop over tempi
    for tempo, tempo_df in ITIs.groupby('stim_tempo_intended'):
        # Loop over two lengths
        for length, length_df in tempo_df.groupby('length'):
            # Loop over participants
            for pp_id, pp_df in length_df.groupby('pp_id_behav'):
                print(pp_id)
                # Get the data
                data = pp_df[stim_resp]
                # Clean, remove outliers outside of 2.5 SDs from the mean
                data = data[(data > np.mean(data) - 2.5 * np.std(data)) & (data < np.mean(data) + 2.5 * np.std(data))]
                # Reshape horizontal data to vertical
                data = data.values.reshape(-1, 1)

                silhouette_dict = {}
                k_means_dict = {}

                # Calculate optimal number of clusters using Silhouette score
                for k in range(K_MIN, K_MAX + 1):
                    kmeans = KMeans(n_clusters=k, max_iter=1000)
                    kmeans.fit(data)
                    k_means_dict[k] = kmeans
                    silhouette = silhouette_score(data, kmeans.labels_)
                    silhouette_dict[k] = silhouette

                # Get optimal number of clusters, i.e. the maximum Silhouette score
                k_clusters = max(silhouette_dict, key=silhouette_dict.get)

                # Get the clusters from the respective saved k-means object in the dict
                kmeans_clusters = sorted(k_means_dict[k_clusters].cluster_centers_)
                kmeans_clusters = [x for l in kmeans_clusters for x in l]  # unpack (for some reason the cluster_centers are all arrays themselves)

                # Calculate left boundaries for clusters; i.e. in between the cluster centers
                # So, the first boundary that we have is in between the center of cluster 1 and cluster 0
                # Everything to the left of that center is the first cluster
                bin_left_boundaries = [kmeans_clusters[i] + (kmeans_clusters[i + 1] - kmeans_clusters[i]) / 2 for i in range(len(kmeans_clusters) - 1)]

                # Make dataframe for k-means clusters and concatenate
                pp_df_output = pd.DataFrame({
                    'pp_id_behav': pp_id,
                    'stim_tempo_intended': tempo,
                    'length': length,
                    'stim_resp': stim_resp,
                    'k_clusters': k_clusters,
                    'bin_left_boundary_i': range(len(bin_left_boundaries)),
                    'bin_left_boundary': bin_left_boundaries
                })
                df_clusters = pd.concat([df_clusters, pp_df_output]).reset_index(drop=True)

                # Make dataframe for silhouette scores and concatenate
                pp_silhouette_output = pd.DataFrame({
                    'pp_id_behav': pp_id,
                    'stim_tempo_intended': tempo,
                    'length': length,
                    'k_clusters': int(k_clusters),
                    'silhouette': silhouette
                }, index=[0])

                df_silhouette = pd.concat([df_silhouette, pp_silhouette_output]).reset_index(drop=True)


"""
Now we calculate G.
"""

# Create empty dataframe for G measure
G_df = pd.DataFrame(columns=['pp_id_behav', 'stim_tempo_intended', 'length', 'stim_resp', 'k_clusters', 'G'])

for stim_resp, resp_type_df in df_clusters.groupby('stim_resp'):
    for pp_id, pp_df in resp_type_df.groupby('pp_id_behav'):
        for tempo, pp_clusters_bytempo in pp_df.groupby('stim_tempo_intended'):
            for length, pp_clusters_bytempoandlength in pp_clusters_bytempo.groupby('length'):
                # Get k and left boundaries of bins
                k = pp_clusters_bytempoandlength.k_clusters.values[0]
                pp_bins_left_boundaries = pp_clusters_bytempoandlength.bin_left_boundary.values

                # This list will hold all the different combinations of durations encountered in the data in this condition
                # (e.g. short-short, short-medium, etc., represented as a-z)
                bigrams_observations = []

                # These are the symbols we will use (i.e. the k first letters of the alphabet)):
                symbols = string.ascii_uppercase[:k]

                # Loop over sequences
                for sequence_id in ITIs[(ITIs.pp_id_behav == pp_id) & (ITIs.stim_tempo_intended == tempo) & (ITIs.length == length)].sequence_id.unique():
                    # Get the response inter-tap intervals
                    iois = ITIs[ITIs.sequence_id == sequence_id][stim_resp].values
                    # Digitize responses into bins (i.e. we get the indices of the bins to which each ITI belongs)
                    iois_symbol_indices = np.digitize(iois, pp_bins_left_boundaries)
                    # Convert indices to symbols
                    iois_symbols = [symbols[i] for i in iois_symbol_indices]
                    # Add the bigrams to the list
                    bigrams_observations.extend(list(zip(iois_symbols[:-1], iois_symbols[1:])))

                # Merge each bigram tuple into a string
                bigrams_observations = [''.join(bigram) for bigram in bigrams_observations]
                # Get unique bigrams
                bigrams_set = set(bigrams_observations)
                # Get the frequency of each bigram
                bigrams_frequencies = [bigrams_observations.count(bigram) for bigram in bigrams_set]
                # Get the probabilities
                bigrams_probabilities = [freq / sum(bigrams_frequencies) for freq in bigrams_frequencies]
                # Calculate entropies
                U_referencegrammar = scipy.stats.entropy([1/len(bigrams_set)] * len(bigrams_set))  # uniform probabilities
                U_targetgrammar = scipy.stats.entropy(bigrams_probabilities)

                # Galculate G
                G = 1 - (U_targetgrammar / U_referencegrammar)

                pp_df = pd.DataFrame({
                    'pp_id_behav': pp_id,
                    'stim_tempo_intended': tempo,
                    'length': length,
                    'stim_resp': stim_resp,
                    'k_clusters': k,
                    'G': G
                }, index=[0])

                G_df = pd.concat([G_df, pp_df], ignore_index=True)


G_df.sort_values(by=['pp_id_behav', 'stim_tempo_intended', 'length']).reset_index(drop=True)

# change data types
G_df.pp_id_behav = G_df.pp_id_behav.astype(int)
G_df.stim_tempo_intended = G_df.stim_tempo_intended.astype(int)
G_df.length = G_df.length.astype(int)
df_silhouette.pp_id_behav = df_silhouette.pp_id_behav.astype(int)
df_silhouette.stim_tempo_intended = df_silhouette.stim_tempo_intended.astype(int)
df_silhouette.length = df_silhouette.length.astype(int)

# open ITIs and ITIs_bytrial
ITIs = pd.read_csv(os.path.join('data', 'experiment', 'processed', 'ITIs.csv'))
ITIs_bytrial = pd.read_csv(os.path.join('data', 'experiment', 'processed', 'ITIs_bytrial.csv'))

# add G measure to ITIs and ITIs_bytrial
for stim_resp, stim_resp_df in G_df.groupby('stim_resp'):
    for pp_id, G_pp_df in stim_resp_df.groupby('pp_id_behav'):
        for tempo, G_pp_tempo_df in G_pp_df.groupby('stim_tempo_intended'):
            for length, G_pp_tempo_length_df in G_pp_tempo_df.groupby('length'):
                ITIs.loc[
                    (ITIs.pp_id_behav == pp_id) &
                    (ITIs.stim_tempo_intended == tempo) &
                    (ITIs.length == length),
                    f'G_{stim_resp[:-4]}'
                    ] = G_pp_tempo_length_df.G.values[0]
                ITIs.loc[
                    (ITIs.pp_id_behav == pp_id) &
                    (ITIs.stim_tempo_intended == tempo) &
                    (ITIs.length == length),
                    f'k_clusters_{stim_resp[:-4]}'
                    ] = G_pp_tempo_length_df.k_clusters.values[0]
                ITIs_bytrial.loc[
                    (ITIs_bytrial.pp_id_behav == pp_id) &
                    (ITIs_bytrial.stim_tempo_intended == tempo) &
                    (ITIs.length == length),
                    f'G_{stim_resp[:-4]}'
                    ] = G_pp_tempo_length_df.G.values[0]
                ITIs_bytrial.loc[
                    (ITIs_bytrial.pp_id_behav == pp_id) &
                    (ITIs_bytrial.stim_tempo_intended == tempo) &
                    (ITIs.length == length),
                    f'k_clusters_{stim_resp[:-4]}'
                    ] = G_pp_tempo_length_df.k_clusters.values[0]

# calculate diff
ITIs['G_diff'] = ITIs.G_resp - ITIs.G_stim
ITIs_bytrial['G_diff'] = ITIs_bytrial.G_resp - ITIs_bytrial.G_stim

# add silhouette score to ITIs and ITIs_bytrial
for pp_id, silhouette_pp_df in df_silhouette.groupby('pp_id_behav'):
    for tempo, silhouette_pp_tempo_df in silhouette_pp_df.groupby('stim_tempo_intended'):
        for length, silhouette_pp_tempo_length_df in silhouette_pp_tempo_df.groupby('length'):
            ITIs.loc[(ITIs.pp_id_behav == pp_id) & (ITIs.stim_tempo_intended == tempo) & (ITIs.length == length), 'silhouette'] = silhouette_pp_tempo_df.silhouette.values[0]
            ITIs_bytrial.loc[(ITIs_bytrial.pp_id_behav == pp_id) & (ITIs_bytrial.stim_tempo_intended == tempo) & (ITIs.length == length), 'silhouette'] = silhouette_pp_tempo_df.silhouette.values[0]

# sort
ITIs = ITIs.sort_values(by = ["pp_id_behav", "stim_id"]).reset_index(drop = True)
ITIs_bytrial = ITIs_bytrial.sort_values(by = ["pp_id_behav", "stim_id"]).reset_index(drop = True)

# save
ITIs.to_csv(os.path.join('data', 'experiment', 'processed', 'ITIs.csv'), index = False)
ITIs_bytrial.to_csv(os.path.join('data', 'experiment', 'processed', 'ITIs_bytrial.csv'), index = False)
