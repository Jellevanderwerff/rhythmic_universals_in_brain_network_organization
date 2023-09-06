"""
This measure is the sum of the absolute differences between the stimulus and response for tempo normalized
ITIs/IOIs.
"""
import pandas as pd
import numpy as np
import os
import thebeat

df = pd.read_csv(os.path.join('data', 'experiment', 'processed', 'ITIs.csv'))
df_bytrial = pd.read_csv(os.path.join('data', 'experiment', 'processed', 'ITIs_bytrial.csv'))

for sequence_id in df.sequence_id.unique():
    # Get the stimulus IOIs and response ITIs
    stim_iois = df.loc[df.sequence_id == sequence_id, 'stim_ioi'].values
    resp_itis = df.loc[df.sequence_id == sequence_id, 'resp_iti'].values

    # Get total durations and their ratio
    total_duration_stim = np.sum(stim_iois)
    total_duration_resp = np.sum(resp_itis)
    tempo_ratio_resp_to_stim = total_duration_resp / total_duration_stim

    # Normalize the response ITIs to the stimulus
    resp_itis_normalized = resp_itis / tempo_ratio_resp_to_stim

    # Normalize the resp_itis so that the total sum is 1
    resp_itis_normalized_to_one = resp_itis_normalized / np.sum(resp_itis_normalized)
    stim_iois_normalized_to_one = stim_iois / np.sum(stim_iois)

    # Get the absolute difference between the stimulus and response
    abs_diffs = np.abs(resp_itis_normalized_to_one - stim_iois_normalized_to_one)

    # Save the measure
    df.loc[df.sequence_id == sequence_id, 'stim_resp_error_abs'] = abs_diffs
    df.loc[df.sequence_id == sequence_id, 'stim_resp_error_abs_sum'] = np.sum(abs_diffs)
    df_bytrial.loc[df.sequence_id == sequence_id, 'stim_resp_error_abs_sum'] = np.sum(abs_diffs)

# Save the data
df.to_csv(os.path.join('data', 'experiment', 'processed', 'ITIs.csv'), index=False)
df_bytrial.to_csv(os.path.join('data', 'experiment', 'processed', 'ITIs_bytrial.csv'), index=False)
