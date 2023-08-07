"""
# Grammatical complexity
Here we calculate the redundancy statistic G (Jamieson & Mewhort, 2005).

The steps are:

1. Use elbow method to find optimal number k of clusters
2. Do K-means clustering to find bins (for each participant and condition separately)
3. Calculate G for each participant
    - Loop over different conditions
    - Loop over participants
    - Discretize all the ITIs into k bins (checked using elbow method), so we have (in the case of
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
            data = pp_df.resp_iti.values.reshape(-1, 1)  # Reshape 1-D data
            kmeans = KMeans(n_clusters=3, max_iter=1000)
            kmeans.fit(data)
            silhouette = silhouette_score(data, kmeans.labels_)
            kmeans_clusters = sorted(kmeans.cluster_centers_)

            # Make dataframe for k-means clusters and concatenate
            pp_df_output = pd.DataFrame({
                'pp_id': pp_id,
                'stim_tempo_intended': tempo,
                'length': length,
                'cluster_center_0': kmeans_clusters[0],
                'cluster_center_1': kmeans_clusters[1],
                'cluster_center_2': kmeans_clusters[2],
                'bin_left_boundary_0': kmeans_clusters[0] + (kmeans_clusters[1] - kmeans_clusters[0]) / 2,  # get the middle between the centers
                'bin_left_boundary_1': kmeans_clusters[1] + (kmeans_clusters[2] - kmeans_clusters[1]) / 2,   # get the middle between the centers
                'silhouette': silhouette,
            })
            df_clusters = pd.concat([df_clusters, pp_df_output])

            # Make dataframe for silhouette scores and concatenate
            pp_silhouette_output = pd.DataFrame({
                'pp_id': pp_id,
                'stim_tempo_intended': tempo,
                'length': length,
                'silhouette': silhouette
            }, index=[0])

            df_silhouette = pd.concat([df_silhouette, pp_silhouette_output])

# Create unconstrained grammar (uniform probabilities)
probabilities_unconstrained = np.array([1/3] * 9)
U_unconstrained = scipy.stats.entropy(probabilities_unconstrained)

# Create empty dataframe for G measure
G_df = pd.DataFrame(columns=['pp_id', 'stim_tempo_intended', 'length', 'G'])

for pp_id, pp_df in df_clusters.groupby('pp_id'):
    for tempo, pp_clusters_bytempo in pp_df.groupby('stim_tempo_intended'):
        for length, pp_clusters_bytempoandlength in pp_clusters_bytempo.groupby('length'):
            pp_bin_left_boundaries = np.array([pp_clusters_bytempoandlength.bin_left_boundary_0.values[0],
                                               pp_clusters_bytempoandlength.bin_left_boundary_1.values[0]])
            # Make empty k by k array which will hold the frequencies
            freqs_pp_bytempoandlength = np.zeros((3, 3))
            # Loop over sequences
            for sequence_id in ITIs[(ITIs.pp_id == pp_id) & (ITIs.stim_tempo_intended == tempo) & (ITIs.length == length)].sequence_id.unique():
                seq_itis = ITIs[ITIs.sequence_id == sequence_id].resp_iti.values
                # Digitize
                itis_digitized = np.digitize(seq_itis, bins=pp_bin_left_boundaries)
                # Increase frequency of relevant position in matrix by one
                for index in range(len(itis_digitized) - 1):
                    i = itis_digitized[index]
                    j = itis_digitized[index + 1]
                    freqs_pp_bytempoandlength[i, j] += 1

            # Turn frequencies into probabilities
            probabilities_pp_bytempoandlength = np.empty((3, 3))
            probabilities_pp_bytempoandlength[0] = freqs_pp_bytempoandlength[0] / np.sum(freqs_pp_bytempoandlength[0])
            probabilities_pp_bytempoandlength[1] = freqs_pp_bytempoandlength[1] / np.sum(freqs_pp_bytempoandlength[1])
            probabilities_pp_bytempoandlength[2] = freqs_pp_bytempoandlength[2] / np.sum(freqs_pp_bytempoandlength[2])

            U_pp_bytempoandlength = scipy.stats.entropy(probabilities_pp_bytempoandlength.flatten())

            G_pp_bytempoandlength = 1 - (U_pp_bytempoandlength / U_unconstrained)

            pp_df = pd.DataFrame({
                'pp_id': pp_id,
                'stim_tempo_intended': tempo,
                'length': length,
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
            ITIs_bytrial.loc[(ITIs_bytrial.pp_id == pp_id) & (ITIs_bytrial.stim_tempo_intended == tempo) & (ITIs.length == length), 'G'] = G_pp_tempo_length_df.G.values[0]
    
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
