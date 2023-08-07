"""
These are measures specific to a participant. So, they can be averages of data from one of the other
data files, or something else.
"""

import pandas as pd
import numpy as np
import thebeat
import os

ITIs = pd.read_csv(os.path.join('data', 'experiment', 'processed', 'ITIs.csv'))
ratio_prefs_kstest = pd.read_csv(os.path.join('data', 'experiment', 'processed', 'ratio_preferences_kstest.csv'))

# Create empty output df
pp_measures = pd.DataFrame()

for tempo, tempo_df in ITIs.groupby('stim_tempo_intended'):
    for length, length_df in tempo_df.groupby('length'):
        for pp_id, pp_df in length_df.groupby('pp_id'):

            ratio_diffs = []

            # Calculate integer ratio diffs
            for sequence_id, sequence_df in pp_df.groupby('sequence_id'):
                stim_seq = thebeat.Sequence(sequence_df.resp_iti)
                resp_seq = thebeat.Sequence(sequence_df.stim_ioi)
                ratio_diff = stim_seq.interval_ratios_from_dyads - resp_seq.interval_ratios_from_dyads
                ratio_diff_avg_abs = np.mean(np.abs(ratio_diff))
                ratio_diffs.append(ratio_diff_avg_abs)

            # pp's dataframe:
            pp_output_df = pd.DataFrame({
                'pp_id': pp_id,
                'stim_tempo_intended': tempo,
                'length': length,
                'G': pp_df.G.values[0],
                'rhythmic_contours_edit_distance_sum': pp_df.rhythmic_contours_edit_distance.sum(),
                'silhouette': pp_df.silhouette.values[0],
                'tempo_diff_avg': np.mean(pp_df.resp_iti - pp_df.stim_ioi),
                'tempo_diff_sd': np.std(pp_df.resp_iti - pp_df.stim_ioi),
                'tempo_diff_avg_abs': np.mean(np.abs(pp_df.resp_iti - pp_df.stim_ioi)),
                'ratio_diff_avg_abs': np.mean(ratio_diffs),
                'entropy_diff_avg': pp_df.entropy_diff.mean(),
                'preference_binary_D': ratio_prefs_kstest[(ratio_prefs_kstest.pp_id == pp_id) & (ratio_prefs_kstest.stim_tempo_intended == tempo) & (ratio_prefs_kstest.length == length) & (ratio_prefs_kstest.hypo_distribution == 'binary')].ks_statistic.values[0],
                'preference_ternary_D': ratio_prefs_kstest[(ratio_prefs_kstest.pp_id == pp_id) & (ratio_prefs_kstest.stim_tempo_intended == tempo) & (ratio_prefs_kstest.length == length) & (ratio_prefs_kstest.hypo_distribution == 'ternary')].ks_statistic.values[0],
            }, index=[0])

            pp_measures = pd.concat([pp_measures, pp_output_df], ignore_index=True)

# change data types where relevant
pp_measures.rhythmic_contours_edit_distance_sum = pp_measures.rhythmic_contours_edit_distance_sum.astype(int)

# sort df
pp_measures = pp_measures.sort_values(by=['pp_id', 'stim_tempo_intended', 'length']).reset_index(drop=True)

pp_measures.to_csv(os.path.join('data', 'experiment', 'processed', 'by_pp.csv'), index=False)
