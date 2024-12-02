"""
These are measures specific to a participant. So, they can be averages of data from one of the other
data files, or something else.
"""

import pandas as pd
import numpy as np
import thebeat
import os
from itertools import permutations
import scipy.io

ITIs = pd.read_csv(os.path.join('data', 'experiment', 'processed', 'ITIs.csv'))
ITIs_bytrial = pd.read_csv(os.path.join('data', 'experiment', 'processed', 'ITIs_bytrial.csv'))
ratios_introduced = pd.read_csv(os.path.join('data', 'experiment', 'processed', 'ratios_introduced.csv'))
pp_id_behav_scan_mapping = pd.read_csv(os.path.join('data', 'experiment', 'raw', 'pp_id_behav_scan_mapping.csv'))

# Create empty output df
pp_measures = pd.DataFrame()

for tempo, tempo_df in ITIs.groupby('stim_tempo_intended'):
    for length, length_df in tempo_df.groupby('length'):
        for pp_id, pp_df in length_df.groupby('pp_id_behav'):
            print(pp_id)
            # pp's dataframe:
            pp_output_df = pd.DataFrame({
                'pp_id_behav': pp_id,
                'pp_id_scan': pp_id_behav_scan_mapping[pp_id_behav_scan_mapping.pp_id_behav == pp_id].pp_id_scan.values[0],
                'stim_tempo_intended': tempo,
                'length': length,
                'condition': f"{tempo}_{length}",
                'n_trials': len(pp_df.sequence_id.unique()),
                'G_resp': pp_df.G_resp.values[0],
                'entropy_diff_norm_q_avg': pp_df.entropy_diff_norm_q.mean(),
                'asynchrony_norm_abs_avg': pp_df.asynchrony_norm_abs.mean(),
                'iti_ioi_cov_diff_avg': pp_df.iti_ioi_cov_diff.mean(),
                'isochrony_introduced': ratios_introduced[(ratios_introduced.pp_id == pp_id) & (ratios_introduced.stim_tempo_intended == tempo) & (ratios_introduced.length == length)].isochrony_introduced.values[0],
                'binary_or_ternary_introduced': ratios_introduced[(ratios_introduced.pp_id == pp_id) & (ratios_introduced.stim_tempo_intended == tempo) & (ratios_introduced.length == length)].binary_or_ternary_introduced.values[0],
                'simple_ratios_introduced': ratios_introduced[(ratios_introduced.pp_id == pp_id) & (ratios_introduced.stim_tempo_intended == tempo) & (ratios_introduced.length == length)].simple_ratios_introduced.values[0],
                'tempo_deviation_abs_avg': np.mean(np.abs(1 - pp_df.tempo_ratio_resp_to_stim)),
                'edit_distance_norm_q_avg': pp_df.edit_distance_norm_q.mean(),

            }, index=[0])

            # Add to output df
            pp_measures = pd.concat([pp_measures, pp_output_df], ignore_index=True)


# sort df
pp_measures = pp_measures.sort_values(by=['stim_tempo_intended', 'length', 'pp_id_scan']).reset_index(drop=True)
pp_measures.to_csv(os.path.join('data', 'experiment', 'processed', 'pp_measures.csv'), index=False)

# split by condition
pp_measures_400_4 = pp_measures[pp_measures.condition == '400_4'].reset_index(drop=True)
pp_measures_400_5 = pp_measures[pp_measures.condition == '400_5'].reset_index(drop=True)
pp_measures_600_4 = pp_measures[pp_measures.condition == '600_4'].reset_index(drop=True)
pp_measures_600_5 = pp_measures[pp_measures.condition == '600_5'].reset_index(drop=True)

# save
pp_measures_400_4.to_csv(os.path.join('data', 'experiment', 'processed', 'pp_measures_4004.csv'), index=False)
pp_measures_400_5.to_csv(os.path.join('data', 'experiment', 'processed', 'pp_measures_4005.csv'), index=False)
pp_measures_600_4.to_csv(os.path.join('data', 'experiment', 'processed', 'pp_measures_6004.csv'), index=False)
pp_measures_600_5.to_csv(os.path.join('data', 'experiment', 'processed', 'pp_measures_6005.csv'), index=False)
