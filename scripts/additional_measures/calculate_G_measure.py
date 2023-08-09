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
    k = 3) the 'symbols' short, medium, and long
    - Make a transition matrix of frequencies of transitions between symbols, 
    so a matrix that looks something like this:

             short | medium | long |      
    short      0   |   1    |  2   |
    medium     3   |   4    |  5   |
    long       6   |   7    |  8   |
    
    Here, the number in the cell (i, j) denotes the number of times that symbol j follows symbol i.
    So the rows are i, and the columns are the symbol that follows i (i.e. j)

    - Turn this frequency matrix into probabilities by dividing each cell by the sum of the 
    row
    - Calculate entropy (U) on the basis of the probabilities
    - Calculate G = 1 - U(targetgrammar) / U(referencegrammar)

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

# variables
K_MIN = 2       # minimum number of clusters to test for
K_MAX = 5      # maximum number of clusters to test for

#suppress warnings
warnings.filterwarnings("ignore")

# Load data
ITIs = pd.read_csv(os.path.join('data', 'experiment', 'processed', 'ITIs.csv'))

# clean (this is to avoid impossible values in the data that lead to non-converging k-means)
clean_df = ITIs[ITIs.resp_iti < 1200]

# create df for k-means clusters
df_clusters = pd.DataFrame()
# create df for silhouette scores
df_silhouette = pd.DataFrame()

# Loop over two tempi
for tempo, tempo_df in clean_df.groupby('stim_tempo_intended'):
    # Loop over two lengths
    for length, length_df in tempo_df.groupby('length'):
        # Loop over participants
        for pp_id, pp_df in length_df.groupby('pp_id'):
            # Reshape horizontal data to vertical
            data = pp_df.resp_iti.values.reshape(-1, 1)

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
                'pp_id': pp_id,
                'stim_tempo_intended': tempo,
                'length': length,
                'k_clusters': k_clusters,
                'bin_left_boundary_i': range(len(bin_left_boundaries)),
                'bin_left_boundary': bin_left_boundaries
            })
            df_clusters = pd.concat([df_clusters, pp_df_output]).reset_index(drop=True)

            # Make dataframe for silhouette scores and concatenate
            pp_silhouette_output = pd.DataFrame({
                'pp_id': pp_id,
                'stim_tempo_intended': tempo,
                'length': length,
                'k_clusters': k_clusters,
                'silhouette': silhouette
            }, index=[0])

            df_silhouette = pd.concat([df_silhouette, pp_silhouette_output]).reset_index(drop=True)


"""
Now we calculate G.
"""

# Create empty dataframe for G measure
G_df = pd.DataFrame(columns=['pp_id', 'stim_tempo_intended', 'length', 'k_clusters', 'G'])

for pp_id, pp_df in df_clusters.groupby('pp_id'):
    for tempo, pp_clusters_bytempo in pp_df.groupby('stim_tempo_intended'):
        for length, pp_clusters_bytempoandlength in pp_clusters_bytempo.groupby('length'):
            # Get k and left boundaries of bins
            k = pp_clusters_bytempoandlength.k_clusters.values[0]
            pp_bins_left_boundaries = pp_clusters_bytempoandlength.bin_left_boundary.values

            # Create unconstrained grammar (uniform probabilities)
            probabilities_unconstrained = np.array([1/k] * (k*k))  # * k would result in the same entropy value
            U_unconstrained = scipy.stats.entropy(probabilities_unconstrained)

            # Make empty k by k array which will hold the frequencies
            freqs_pp_bytempoandlength = np.zeros((k, k))
            # Loop over sequences
            for sequence_id in ITIs[(ITIs.pp_id == pp_id) & (ITIs.stim_tempo_intended == tempo) & (ITIs.length == length)].sequence_id.unique():
                seq_itis = ITIs[ITIs.sequence_id == sequence_id].resp_iti.values
                # Digitize
                itis_digitized = np.digitize(seq_itis, bins=pp_bins_left_boundaries)
                # Increase frequency of relevant position in matrix by one
                for index in range(len(itis_digitized) - 1):
                    i = itis_digitized[index]
                    j = itis_digitized[index + 1]
                    freqs_pp_bytempoandlength[i, j] += 1

            # Turn frequencies into probabilities
            # We divide each datapoint by the row sum for this
            # The row sum in the resulting matrix should always sum to 1
            probabilities_pp_bytempoandlength = np.empty((k, k))
            for i in range(k):
                probabilities_pp_bytempoandlength[i] = freqs_pp_bytempoandlength[i] / np.sum(freqs_pp_bytempoandlength[i])

            # Finally, we calculate G
            U_pp_bytempoandlength = scipy.stats.entropy(probabilities_pp_bytempoandlength.flatten())
            G_pp_bytempoandlength = 1 - (U_pp_bytempoandlength / U_unconstrained)

            pp_df = pd.DataFrame({
                'pp_id': pp_id,
                'stim_tempo_intended': tempo,
                'length': length,
                'k_clusters': k,
                'G': G_pp_bytempoandlength
            }, index=[0])

            G_df = pd.concat([G_df, pp_df], ignore_index=True)


G_df.sort_values(by=['pp_id', 'stim_tempo_intended', 'length']).reset_index(drop=True)

# change data types
G_df.pp_id = G_df.pp_id.astype(int)
G_df.stim_tempo_intended = G_df.stim_tempo_intended.astype(int)
G_df.length = G_df.length.astype(int)
df_silhouette.pp_id = df_silhouette.pp_id.astype(int)
df_silhouette.stim_tempo_intended = df_silhouette.stim_tempo_intended.astype(int)
df_silhouette.length = df_silhouette.length.astype(int)

# open ITIs and ITIs_bytrial
ITIs = pd.read_csv(os.path.join('data', 'experiment', 'processed', 'ITIs.csv'))
ITIs_bytrial = pd.read_csv(os.path.join('data', 'experiment', 'processed', 'ITIs_bytrial.csv'))

# add G measure to ITIs and ITIs_bytrial
for pp_id, G_pp_df in G_df.groupby('pp_id'):
    for tempo, G_pp_tempo_df in G_pp_df.groupby('stim_tempo_intended'):
        for length, G_pp_tempo_length_df in G_pp_tempo_df.groupby('length'):
            ITIs.loc[(ITIs.pp_id == pp_id) & (ITIs.stim_tempo_intended == tempo) & (ITIs.length == length), 'G'] = G_pp_tempo_length_df.G.values[0]
            ITIs.loc[(ITIs.pp_id == pp_id) & (ITIs.stim_tempo_intended == tempo) & (ITIs.length == length), 'k_clusters'] = G_pp_tempo_length_df.k_clusters.values[0]
            ITIs_bytrial.loc[(ITIs_bytrial.pp_id == pp_id) & (ITIs_bytrial.stim_tempo_intended == tempo) & (ITIs.length == length), 'G'] = G_pp_tempo_length_df.G.values[0]
            ITIs_bytrial.loc[(ITIs_bytrial.pp_id == pp_id) & (ITIs_bytrial.stim_tempo_intended == tempo) & (ITIs.length == length), 'k_clusters'] = G_pp_tempo_length_df.k_clusters.values[0]
    
# add silhouette score to ITIs and ITIs_bytrial
for pp_id, silhouette_pp_df in df_silhouette.groupby('pp_id'):
    for tempo, silhouette_pp_tempo_df in silhouette_pp_df.groupby('stim_tempo_intended'):
        for length, silhouette_pp_tempo_length_df in silhouette_pp_tempo_df.groupby('length'):
            ITIs.loc[(ITIs.pp_id == pp_id) & (ITIs.stim_tempo_intended == tempo) & (ITIs.length == length), 'silhouette'] = silhouette_pp_tempo_df.silhouette.values[0]
            ITIs_bytrial.loc[(ITIs_bytrial.pp_id == pp_id) & (ITIs_bytrial.stim_tempo_intended == tempo) & (ITIs.length == length), 'silhouette'] = silhouette_pp_tempo_df.silhouette.values[0]

# sort
ITIs = ITIs.sort_values(by = ["pp_id", "stim_id"]).reset_index(drop = True)
ITIs_bytrial = ITIs_bytrial.sort_values(by = ["pp_id", "stim_id"]).reset_index(drop = True)

# save
ITIs.to_csv(os.path.join('data', 'experiment', 'processed', 'ITIs.csv'), index = False)
ITIs_bytrial.to_csv(os.path.join('data', 'experiment', 'processed', 'ITIs_bytrial.csv'), index = False)
