"""
This measure is the sum of the absolute differences between the stimulus and response for tempo normalized
ITIs/IOIs.
"""
import pandas as pd
import numpy as np
import os
import thebeat

df = pd.read_csv(os.path.join('data', 'behavioural', 'processed', 'ITIs.csv'))
df_bytrial = pd.read_csv(os.path.join('data', 'behavioural', 'processed', 'ITIs_bytrial.csv'))

for sequence_id in df.sequence_id.unique():
    # Get the stimulus IOIs and response ITIs
    stim_iois = df.loc[df.sequence_id == sequence_id, 'stim_ioi'].values
    resp_itis = df.loc[df.sequence_id == sequence_id, 'resp_iti_norm'].values

    # Normalize the resp_itis so that the total sum is 1
    resp_itis_normalized_to_one = resp_itis / np.sum(resp_itis)
    stim_iois_normalized_to_one = stim_iois / np.sum(stim_iois)

    # Get the absolute difference between the stimulus and response
    abs_diffs = np.abs(resp_itis_normalized_to_one - stim_iois_normalized_to_one)

    # Save the measure
    df.loc[df.sequence_id == sequence_id, 'asynchrony_norm_abs'] = abs_diffs
    df.loc[df.sequence_id == sequence_id, 'iti_ioi_cov_diff'] = (np.std(resp_itis) / np.mean(resp_itis)) - (np.std(stim_iois) / np.mean(stim_iois))
    df_bytrial.loc[df_bytrial.sequence_id == sequence_id, 'asynchrony_norm_abs_trialsum'] = np.sum(abs_diffs)
    df_bytrial.loc[df_bytrial.sequence_id == sequence_id, 'iti_ioi_cov_diff'] = (np.std(resp_itis) / np.mean(resp_itis)) - (np.std(stim_iois) / np.mean(stim_iois))


# Save the data
df.to_csv(os.path.join('data', 'behavioural', 'processed', 'ITIs.csv'), index=False)
df_bytrial.to_csv(os.path.join('data', 'behavioural', 'processed', 'ITIs_bytrial.csv'), index=False)
