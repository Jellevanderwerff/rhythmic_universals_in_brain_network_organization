"""
These are measures specific to a participant. So, they can be averages of data from one of the other
data files, or something else.
"""

import pandas as pd
import numpy as np
import thebeat
import os

ITIs = pd.read_csv(os.path.join('data', 'pilot', 'processed', 'ITIs.csv'))
pp_measures = pd.DataFrame()
pp_measures['pp_id'] = ITIs.pp_id.unique()


for pp_id, pp_df in ITIs.groupby('pp_id'):
    pp_measures.loc[pp_measures.pp_id == pp_id, 'tempo_400_diff_avg'] = np.mean(pp_df[pp_df.stim_tempo_intended == 400].stim_ioi - pp_df[pp_df.stim_tempo_intended == 400].resp_iti)
    pp_measures.loc[pp_measures.pp_id == pp_id, 'tempo_400_diff_sd'] = np.std(pp_df[pp_df.stim_tempo_intended == 400].stim_ioi - pp_df[pp_df.stim_tempo_intended == 400].resp_iti)
    pp_measures.loc[pp_measures.pp_id == pp_id, 'tempo_600_diff_avg'] = np.mean(pp_df[pp_df.stim_tempo_intended == 600].stim_ioi - pp_df[pp_df.stim_tempo_intended == 600].resp_iti)
    pp_measures.loc[pp_measures.pp_id == pp_id, 'tempo_600_diff_sd'] = np.std(pp_df[pp_df.stim_tempo_intended == 600].stim_ioi - pp_df[pp_df.stim_tempo_intended == 600].resp_iti)
    pp_measures.loc[pp_measures.pp_id == pp_id, 'entropy_diff_avg'] = pp_df.entropy_diff.mean()
    pp_measures.loc[pp_measures.pp_id == pp_id, 'G_400'] = pp_df[pp_df.stim_tempo_intended == 400].G.values[0]
    pp_measures.loc[pp_measures.pp_id == pp_id, 'G_600'] = pp_df[pp_df.stim_tempo_intended == 600].G.values[0]
    pp_measures.loc[pp_measures.pp_id == pp_id, 'G_avg_of_both_tempi'] = pp_df.G.mean()
    pp_measures.loc[pp_measures.pp_id == pp_id, 'rhythmic_contours_edit_distance_sum'] = pp_df.rhythmic_contours_edit_distance.sum()
    pp_measures.loc[pp_measures.pp_id == pp_id, 'silhouette_avg'] = pp_df.silhouette.mean()

    ratio_diffs = []

    # Calculate integer ratio diffs
    for stim_id, stim_df in pp_df.groupby('stim_id'):
        stim_seq = thebeat.Sequence(stim_df.resp_iti)
        resp_seq = thebeat.Sequence(stim_df.stim_ioi)
        ratio_diff = stim_seq.interval_ratios_from_dyads - resp_seq.interval_ratios_from_dyads
        ratio_diff_avg_abs = np.mean(np.abs(ratio_diff))
        ratio_diffs.append(ratio_diff_avg_abs)
    
    pp_measures.loc[pp_measures.pp_id == pp_id, 'ratio_diff_pp_avg_abs'] = np.mean(ratio_diffs)

# change data types where relevant
pp_measures.rhythmic_contours_edit_distance_sum = pp_measures.rhythmic_contours_edit_distance_sum.astype(int)

pp_measures.to_csv(os.path.join('data', 'pilot', 'processed', 'by_pp.csv'), index = False)