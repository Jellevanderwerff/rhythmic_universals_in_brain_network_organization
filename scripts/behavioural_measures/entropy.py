import thebeat
import os
import pandas as pd

# Load the data
ITIs = pd.read_csv(os.path.join('data', 'experiment', 'processed', 'ITIs.csv'))
ITIs_bytrial = pd.read_csv(os.path.join('data', 'experiment', 'processed', 'ITIs_bytrial.csv'))
fourier_df = pd.read_csv(os.path.join('data', 'experiment', 'processed', 'fourier.csv'))

# Calculate stim-resp entropy diff
for sequence, sequence_df in ITIs.groupby('sequence_id'):
    stim = thebeat.Sequence(sequence_df['stim_ioi'].values)
    resp = thebeat.Sequence(sequence_df['resp_iti_norm'].values)
    stim_fourier_tempo = fourier_df.loc[fourier_df['sequence_id'] == sequence, 'fourier_16th_duration_stim'].values[0]
    resp_fourier_tempo = fourier_df.loc[fourier_df['sequence_id'] == sequence, 'fourier_16th_duration_resp'].values[0]
    stim = stim.quantize_iois(stim_fourier_tempo)
    resp = resp.quantize_iois(resp_fourier_tempo)
    stim_entropy = thebeat.stats.get_rhythmic_entropy(stim, resolution=stim_fourier_tempo)
    resp_entropy = thebeat.stats.get_rhythmic_entropy(resp, resolution=resp_fourier_tempo)
    ITIs.loc[ITIs['sequence_id'] == sequence, 'stim_entropy_norm_q'] = stim_entropy
    ITIs.loc[ITIs['sequence_id'] == sequence, 'resp_entropy_norm_q'] = resp_entropy
    ITIs.loc[ITIs['sequence_id'] == sequence, 'entropy_diff_norm_q'] = resp_entropy - stim_entropy
    ITIs_bytrial.loc[ITIs_bytrial['sequence_id'] == sequence, 'stim_entropy_norm_q'] = stim_entropy
    ITIs_bytrial.loc[ITIs_bytrial['sequence_id'] == sequence, 'resp_entropy_norm_q'] = resp_entropy
    ITIs_bytrial.loc[ITIs_bytrial['sequence_id'] == sequence, 'entropy_diff_norm_q'] = resp_entropy - stim_entropy

# Save the data
ITIs.to_csv(os.path.join('data', 'experiment', 'processed', 'ITIs.csv'), index=False)
ITIs_bytrial.to_csv(os.path.join('data', 'experiment', 'processed', 'ITIs_bytrial.csv'), index=False)
