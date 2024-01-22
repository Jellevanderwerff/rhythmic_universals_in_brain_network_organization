import pandas as pd
import numpy as np
import os
import thebeat

"""
Here we follow the method from Roeske et al. (2020) in Current Biology, to calculate interval ratios.
"""

boundaries = [1/4.75, 1/4.25, 1/3.75, 1/3.25, 1/2.75, 1/2.25, 1/2]
isochrony_bin_index = 5
binary_bin_index = 3
ternary_bin_index = 1

# Load the data
df = pd.read_csv(os.path.join('data', 'experiment', 'processed', 'ITIs.csv'))
df_bytrial = pd.read_csv(os.path.join('data', 'experiment', 'processed', 'ITIs_bytrial.csv'))
ratios_df = pd.DataFrame()

for sequence_id in df.sequence_id.unique():
    print(sequence_id)
    # Get stimulus IOIs and response ITIs
    stim_iois = df[df.sequence_id == sequence_id].stim_ioi.values
    resp_itis = df[df.sequence_id == sequence_id].resp_iti.values
    # Make thebeat Sequences
    stim = thebeat.Sequence(stim_iois)
    resp = thebeat.Sequence(resp_itis)
    # Get interval ratios from dyads
    stim_interval_ratios = stim.interval_ratios_from_dyads
    resp_interval_ratios = resp.interval_ratios_from_dyads
    # 'Aggregate'. If any of the interval ratios is larger than 0.5, we take the inverse
    stim_interval_ratios = [ratio if ratio <= 0.5 else 1 - ratio for ratio in stim_interval_ratios]
    resp_interval_ratios = [ratio if ratio <= 0.5 else 1 - ratio for ratio in resp_interval_ratios]
    # Discretize (and normalize)
    stim_ratios_freqs = np.histogram(stim_interval_ratios, bins=boundaries, normalize=True)
    resp_ratios_freqs = np.histogram(resp_interval_ratios, bins=boundaries, normalize=True)
    # Count isochrony
    isochrony_freq_stim = stim_ratios_freqs[0][isochrony_bin_index]
    isochrony_freq_resp = resp_ratios_freqs[0][isochrony_bin_index]
    # Count binary
    binary_freq_stim = stim_ratios_freqs[0][binary_bin_index]
    binary_freq_resp = resp_ratios_freqs[0][binary_bin_index]
    # Count ternary
    ternary_freq_stim = stim_ratios_freqs[0][ternary_bin_index]
    ternary_freq_resp = resp_ratios_freqs[0][ternary_bin_index]
    # Count additions/subtractions
    isochrony_introduced = isochrony_freq_resp - isochrony_freq_stim
    binary_introduced = binary_freq_resp - binary_freq_stim
    ternary_introduced = ternary_freq_resp - ternary_freq_stim
    binary_or_ternary_introduced = binary_introduced + ternary_introduced

    # Make little dataframe
    df_piece = pd.DataFrame({
        'pp_id': df[df.sequence_id == sequence_id].pp_id_behav.values[0],
        'sequence_id': sequence_id,
        'isochrony_introduced': isochrony_introduced,
        'binary_introduced': binary_introduced,
        'ternary_introduced': ternary_introduced,
        'binary_or_ternary_introduced': binary_or_ternary_introduced
    }, index=[0])

    # Add to ratios_df
    ratios_df = pd.concat([ratios_df, df_piece], ignore_index=True)

    # Also add to ITIs df the important bits
    df.loc[df.sequence_id == sequence_id, 'isochrony_introduced'] = isochrony_introduced
    df.loc[df.sequence_id == sequence_id, 'binary_introduced'] = binary_introduced
    df.loc[df.sequence_id == sequence_id, 'ternary_introduced'] = ternary_introduced
    df.loc[df.sequence_id == sequence_id, 'binary_or_ternary_introduced'] = binary_or_ternary_introduced

    # Also add to ITIs_bytrial df the important bits
    df_bytrial.loc[df_bytrial.sequence_id == sequence_id, 'isochrony_introduced'] = isochrony_introduced
    df_bytrial.loc[df_bytrial.sequence_id == sequence_id, 'binary_introduced'] = binary_introduced
    df_bytrial.loc[df_bytrial.sequence_id == sequence_id, 'ternary_introduced'] = ternary_introduced
    df_bytrial.loc[df_bytrial.sequence_id == sequence_id, 'binary_or_ternary_introduced'] = binary_or_ternary_introduced


# Save the data
df.to_csv(os.path.join('data', 'experiment', 'processed', 'ITIs.csv'), index=False)
df_bytrial.to_csv(os.path.join('data', 'experiment', 'processed', 'ITIs_bytrial.csv'), index=False)
ratios_df.to_csv(os.path.join('data', 'experiment', 'processed', 'ratios_introduced.csv'), index=False)
