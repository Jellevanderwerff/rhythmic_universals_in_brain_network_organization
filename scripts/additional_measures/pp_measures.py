"""
These are measures specific to a participant. So, they can be averages of data from one of the other
data files, or something else.
"""

import pandas as pd
import numpy as np
import thebeat
import os
from itertools import permutations

ITIs = pd.read_csv(os.path.join('data', 'experiment', 'processed', 'ITIs.csv'))
ratio_prefs_kstest = pd.read_csv(os.path.join('data', 'experiment', 'processed', 'ratio_preferences_kstest.csv'))
interval_ratios = pd.read_csv(os.path.join('data', 'experiment', 'processed', 'interval_ratios.csv'))

# get simple integer ratios (< 5)
#small_integer_ratios = list(permutations(range(1, 6), 2))
#small_integer_ratios = [f'{ratio[0]}:{ratio[1]}' for ratio in small_integer_ratios]
small_integer_ratios = ['1:1', '1:2', '2:1', '1:3', '3:1']

# Create empty output df
pp_measures = pd.DataFrame()

for tempo, tempo_df in ITIs.groupby('stim_tempo_intended'):
    for length, length_df in tempo_df.groupby('length'):
        for pp_id, pp_df in length_df.groupby('pp_id'):

            # Get interval ratios for pp
            pp_interval_ratios = interval_ratios[(interval_ratios.pp_id == pp_id) & (interval_ratios.stim_tempo_intended == tempo) & (interval_ratios.length == length)]

            # Count number of isochronous, binary, and ternary ratios
            small_integers_n = pp_interval_ratios[pp_interval_ratios.quantized_ratio_str.isin(small_integer_ratios)].shape[0]
            small_integers_prop = small_integers_n / pp_interval_ratios.shape[0]

            # pp's dataframe:
            pp_output_df = pd.DataFrame({
                'pp_id': pp_id,
                'stim_tempo_intended': tempo,
                'length': length,
                'G_resp': pp_df.G_resp.values[0],
                'entropy_diff_avg': pp_df.entropy_diff.mean(),
                'entropy_resp_avg': pp_df.resp_entropy.mean(),
                'edit_distance_norm_avg': pp_df.edit_distance_normalized.mean(),
                'rhythmic_contours_edit_distance_quantized_avg': pp_df.rhythmic_contours_edit_distance.mean(),
                'rhythmic_contours_entropy_diff_quantized': pp_df.rhythmic_contours_entropy_diff_quantized.mean(),
                'small_integers_vs_total_prop': small_integers_prop

            }, index=[0])

            pp_measures = pd.concat([pp_measures, pp_output_df], ignore_index=True)

# sort df
pp_measures = pp_measures.sort_values(by=['pp_id', 'stim_tempo_intended', 'length']).reset_index(drop=True)

pp_measures.to_csv(os.path.join('data', 'experiment', 'processed', 'pp_measures.csv'), index=False)
