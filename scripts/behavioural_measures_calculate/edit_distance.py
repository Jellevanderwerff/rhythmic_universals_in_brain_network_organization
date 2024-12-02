import pandas as pd
import numpy as np
import thebeat
import os

# Load the data
df = pd.read_csv(os.path.join('data', 'experiment', 'processed', 'ITIs.csv'))
df_bytrial = pd.read_csv(os.path.join('data', 'experiment', 'processed', 'ITIs_bytrial.csv'))
fourier = pd.read_csv(os.path.join('data', 'experiment', 'processed', 'fourier.csv'))

edit_distances = pd.DataFrame()

for sequence_id, sequence_df in df.groupby("sequence_id"):
    duration_sixteenth_note = fourier[fourier.sequence_id == sequence_id].fourier_16th_duration_combined.values[0]

    # stim
    stim_iois = sequence_df[sequence_df.sequence_id == sequence_id].stim_ioi.values
    stim = thebeat.Sequence(stim_iois)
    stim.quantize_iois(to=duration_sixteenth_note)

    # resp
    resp_iois = sequence_df[sequence_df.sequence_id == sequence_id].resp_iti_norm.values
    resp = thebeat.Sequence(resp_iois)
    resp.quantize_iois(to=duration_sixteenth_note)

    # edit distance
    edit_distance = thebeat.stats.edit_distance_sequence(stim, resp, resolution=duration_sixteenth_note)

    # Add to output df
    df.loc[df.sequence_id == sequence_id, 'edit_distance_norm_q'] = int(edit_distance)
    df_bytrial.loc[df_bytrial.sequence_id == sequence_id, 'edit_distance_norm_q'] = int(edit_distance)

df.to_csv(os.path.join('data', 'experiment', 'processed', 'ITIs.csv'), index=False)
df_bytrial.to_csv(os.path.join('data', 'experiment', 'processed', 'ITIs_bytrial.csv'), index=False)
