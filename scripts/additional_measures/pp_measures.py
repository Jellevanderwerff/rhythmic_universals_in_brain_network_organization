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
pp_id_behav_scan_mapping = pd.read_csv(os.path.join('data', 'experiment', 'raw', 'pp_id_behav_scan_mapping.csv'))

small_integer_ratios = ['1:1', '1:2', '2:1', '1:3', '3:1']

# Create empty output df
pp_measures = pd.DataFrame()

for tempo, tempo_df in ITIs.groupby('stim_tempo_intended'):
    for length, length_df in tempo_df.groupby('length'):
        for pp_id, pp_df in length_df.groupby('pp_id_behav'):

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
                'simple_ratio_introduced_avg': pp_df.simple_ratio_introduced.mean(),
                'tempo_deviation_abs_avg': np.mean(np.abs(1 - pp_df.tempo_ratio_resp_to_stim)),


            }, index=[0])

            # Add to output df
            pp_measures = pd.concat([pp_measures, pp_output_df], ignore_index=True)


# sort df
pp_measures = pp_measures.sort_values(by=['pp_id_scan', 'stim_tempo_intended', 'length']).reset_index(drop=True)
pp_measures.to_csv(os.path.join('data', 'experiment', 'processed', 'pp_measures.csv'), index=False)
ITIs.to_csv(os.path.join('data', 'experiment', 'processed', 'ITIs.csv'), index=False)
ITIs_bytrial.to_csv(os.path.join('data', 'experiment', 'processed', 'ITIs_bytrial.csv'), index=False)
