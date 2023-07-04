# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import scipy.stats
import os
import warnings

#suppress warnings
warnings.filterwarnings("ignore")

# Load data
raw_df = pd.read_csv(os.path.join('..', 'data', 'pilot', 'processed', 'ITIs.csv'))

# clean
clean_df = raw_df[raw_df.resp_iti < 1200]

# Split data
df_400 = clean_df[clean_df.stim_tempo_intended == 400]
df_600 = clean_df[clean_df.stim_tempo_intended == 600]
dfs = {'400': df_400, '600': df_600}


# calculate k-means clusters
df_clusters = pd.DataFrame()

for tempo, df in dfs.items():
    for pp_id in df.pp_id.unique():
        pp_data = df[df.pp_id == pp_id]
        data = pp_data.resp_iti.values.reshape(-1, 1)  # Reshape 1-D data
        kmeans = KMeans(n_clusters=3, max_iter=1000)
        kmeans.fit(data)
        kmeans_clusters = sorted(kmeans.cluster_centers_)
        pp_df = pd.DataFrame({
            'pp_id': pp_id,
            'stim_tempo_intended': tempo,
            'cluster_center_0': kmeans_clusters[0],
            'cluster_center_1': kmeans_clusters[1],
            'cluster_center_2': kmeans_clusters[2],
            'bin_left_boundary_0': kmeans_clusters[0] + (kmeans_clusters[1] - kmeans_clusters[0]) / 2,  # get the middle between the centers
            'bin_left_boundary_1': kmeans_clusters[1] + (kmeans_clusters[2] - kmeans_clusters[1]) / 2   # get the middle between the centers
        })
        df_clusters = pd.concat([df_clusters, pp_df])

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

# open ITIs and ITIs_bytrial
ITIs = pd.read_csv(os.path.join('..', 'data', 'pilot', 'processed', 'ITIs.csv'), index_col=0)
ITIs_bytrial = pd.read_csv(os.path.join('..', 'data', 'pilot', 'processed', 'ITIs_bytrial.csv'), index_col=0)

# add G measure to ITIs and ITIs_bytrial
ITIs = pd.merge(ITIs, G_df, on=['pp_id', 'stim_tempo_intended'])
ITIs_bytrial = pd.merge(ITIs_bytrial, G_df, on=['pp_id', 'stim_tempo_intended'])

# save
ITIs.to_csv(os.path.join('..', 'data', 'pilot', 'processed', 'ITIs.csv'))
ITIs_bytrial.to_csv(os.path.join('..', 'data', 'pilot', 'processed', 'ITIs_bytrial.csv'))