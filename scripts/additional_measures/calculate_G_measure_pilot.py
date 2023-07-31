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
raw_df = pd.read_csv(os.path.join('data', 'pilot', 'processed', 'ITIs.csv'))

# clean
clean_df = raw_df[raw_df.resp_iti < 1200]

# Split data
df_400 = clean_df[clean_df.stim_tempo_intended == 400]
df_600 = clean_df[clean_df.stim_tempo_intended == 600]
dfs = {'400': df_400, '600': df_600}


# calculate k-means clusters
df_clusters = pd.DataFrame()
df_silhouette = pd.DataFrame()

for tempo, df in dfs.items():
    for pp_id in df.pp_id.unique():
        pp_data = df[df.pp_id == pp_id]
        data = pp_data.resp_iti.values.reshape(-1, 1)  # Reshape 1-D data
        kmeans = KMeans(n_clusters=3, max_iter=1000)
        kmeans.fit(data)
        silhouette = silhouette_score(data, kmeans.labels_)
        kmeans_clusters = sorted(kmeans.cluster_centers_)
        pp_df = pd.DataFrame({
            'pp_id': pp_id,
            'stim_tempo_intended': tempo,
            'cluster_center_0': kmeans_clusters[0],
            'cluster_center_1': kmeans_clusters[1],
            'cluster_center_2': kmeans_clusters[2],
            'bin_left_boundary_0': kmeans_clusters[0] + (kmeans_clusters[1] - kmeans_clusters[0]) / 2,  # get the middle between the centers
            'bin_left_boundary_1': kmeans_clusters[1] + (kmeans_clusters[2] - kmeans_clusters[1]) / 2,   # get the middle between the centers
            'silhouette': silhouette,
        })
        df_clusters = pd.concat([df_clusters, pp_df])
        df_silhouette = pd.concat([df_silhouette, pd.DataFrame({
            'pp_id': pp_id,
            'stim_tempo_intended': tempo,
            'silhouette': silhouette
        }, index=[0])])

# make csv
freqs = {'400': {},
         '600': {}}

probabilities_unconstrained = np.array([1/9] * 9)
U_unconstrained = scipy.stats.entropy(probabilities_unconstrained)

G_df = pd.DataFrame(columns=['pp_id', 'stim_tempo_intended', 'G'])

# Loop over two tempi
for tempo in freqs.keys():
    df_tempo = dfs[tempo]
    # Loop over participants
    for pp_id in df_tempo.pp_id.unique():
        pp_bins_df = df_clusters[(df_clusters.pp_id == pp_id) & (df_clusters.stim_tempo_intended == tempo)]
        pp_bin_left_boundaries = np.array([pp_bins_df.bin_left_boundary_0.values[0],
                                           pp_bins_df.bin_left_boundary_1.values[0]])
        df_pp = df_tempo[df_tempo.pp_id == pp_id]
        # Make empty k by k array
        freqs_pp = np.zeros((3, 3))
        # Loop over sequences
        for sequence_id in df_pp.sequence_id.unique():
            seq_itis = df_pp[df_pp.sequence_id == sequence_id].resp_iti.values
            # Digitize
            itis_digitized = np.digitize(seq_itis, bins=pp_bin_left_boundaries)
            # Increase frequency of relevant position in matrix by one
            for index in range(len(itis_digitized) - 1):
                i = itis_digitized[index]
                j = itis_digitized[index + 1]
                freqs_pp[i, j] += 1

        probabilities_pp = freqs_pp / np.sum(freqs_pp)
        U_pp = scipy.stats.entropy(probabilities_pp.flatten())
        G_pp = 1 - (U_pp / U_unconstrained)
        pp_df = pd.DataFrame({
            'pp_id': pp_id,
            'stim_tempo_intended': tempo,
            'G': G_pp
        }, index=[0])

        G_df = pd.concat([G_df, pp_df], ignore_index=True)

G_df.sort_values(by=['pp_id', 'stim_tempo_intended']).reset_index(drop=True)

# change data types
G_df.pp_id = G_df.pp_id.astype(int)
G_df.stim_tempo_intended = G_df.stim_tempo_intended.astype(int)
df_silhouette.pp_id = df_silhouette.pp_id.astype(int)
df_silhouette.stim_tempo_intended = df_silhouette.stim_tempo_intended.astype(int)

# open ITIs and ITIs_bytrial
ITIs = pd.read_csv(os.path.join('data', 'pilot', 'processed', 'ITIs.csv'))
ITIs_bytrial = pd.read_csv(os.path.join('data', 'pilot', 'processed', 'ITIs_bytrial.csv'))

# add G measure to ITIs and ITIs_bytrial
for pp_id, G_pp_df in G_df.groupby('pp_id'):
    for tempo, G_pp_tempo_df in G_pp_df.groupby('stim_tempo_intended'):
        ITIs.loc[(ITIs.pp_id == pp_id) & (ITIs.stim_tempo_intended == tempo), 'G'] = G_pp_tempo_df.G.values[0]
        ITIs_bytrial.loc[(ITIs_bytrial.pp_id == pp_id) & (ITIs_bytrial.stim_tempo_intended == tempo), 'G'] = G_pp_tempo_df.G.values[0]
    
# add silhouette score to ITIs and ITIs_bytrial
for pp_id, silhouette_pp_df in df_silhouette.groupby('pp_id'):
    for tempo, silhouette_pp_tempo_df in silhouette_pp_df.groupby('stim_tempo_intended'):
        ITIs.loc[(ITIs.pp_id == pp_id) & (ITIs.stim_tempo_intended == tempo), 'silhouette'] = silhouette_pp_tempo_df.silhouette.values[0]
        ITIs_bytrial.loc[(ITIs_bytrial.pp_id == pp_id) & (ITIs_bytrial.stim_tempo_intended == tempo), 'silhouette'] = silhouette_pp_tempo_df.silhouette.values[0]

# sort
ITIs = ITIs.sort_values(by = ["pp_id", "stim_id"]).reset_index(drop = True)
ITIs_bytrial = ITIs_bytrial.sort_values(by = ["pp_id", "stim_id"]).reset_index(drop = True)

# save
ITIs.to_csv(os.path.join('data', 'pilot', 'processed', 'ITIs.csv'), index = False)
ITIs_bytrial.to_csv(os.path.join('data', 'pilot', 'processed', 'ITIs_bytrial.csv'), index = False)
