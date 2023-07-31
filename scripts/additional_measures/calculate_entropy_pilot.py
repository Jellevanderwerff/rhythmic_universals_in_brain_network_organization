import thebeat
import os
import pandas as pd

# Load the data
ITIs = pd.read_csv(os.path.join('data', 'pilot', 'processed', 'ITIs.csv'))
ITIs_bytrial = pd.read_csv(os.path.join('data', 'pilot', 'processed', 'ITIs_bytrial.csv'))

# Calculate stim-resp entropy diff
for sequence, sequence_df in ITIs.groupby('sequence_id'):
    stim = thebeat.Sequence(sequence_df['stim_ioi'].values)
    resp = thebeat.Sequence(sequence_df['resp_iti'].values)
    stim_entropy = thebeat.stats.get_rhythmic_entropy(stim, bin_fraction=1/16)
    resp_entropy = thebeat.stats.get_rhythmic_entropy(resp, bin_fraction=1/16)
    ITIs.loc[ITIs['sequence_id'] == sequence, 'stim_entropy'] = stim_entropy
    ITIs.loc[ITIs['sequence_id'] == sequence, 'resp_entropy'] = resp_entropy
    ITIs.loc[ITIs['sequence_id'] == sequence, 'entropy_diff'] = stim_entropy - resp_entropy
    ITIs_bytrial.loc[ITIs['sequence_id'] == sequence, 'stim_entropy'] = stim_entropy
    ITIs_bytrial.loc[ITIs['sequence_id'] == sequence, 'resp_entropy'] = resp_entropy
    ITIs_bytrial.loc[ITIs['sequence_id'] == sequence, 'entropy_diff'] = stim_entropy - resp_entropy

# Save the data
ITIs.to_csv(os.path.join('data', 'pilot', 'processed', 'ITIs.csv'), index=False)
ITIs_bytrial.to_csv(os.path.join('data', 'pilot', 'processed', 'ITIs_bytrial.csv'), index=False)
