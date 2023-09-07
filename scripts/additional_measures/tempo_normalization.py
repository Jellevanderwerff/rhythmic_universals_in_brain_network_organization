import pandas as pd
import numpy as np
import thebeat
import os


# Load the data
df = pd.read_csv(os.path.join('data', 'experiment', 'processed', 'ITIs.csv'))
df_bytrial = pd.read_csv(os.path.join('data', 'experiment', 'processed', 'ITIs_bytrial.csv'))

for sequence_id in df.sequence_id.unique():
    # Get stimulus IOIs and response ITIs
    stim_iois = df[df.sequence_id == sequence_id].stim_ioi.values
    resp_itis = df[df.sequence_id == sequence_id].resp_iti.values

    # Get total durations and their ratio
    total_duration_stim = np.sum(stim_iois)
    total_duration_resp = np.sum(resp_itis)
    tempo_ratio_resp_to_stim = total_duration_resp / total_duration_stim

    # Normalize the response ITIs
    resp_itis_normalized = resp_itis / tempo_ratio_resp_to_stim
    assert np.round(np.sum(resp_itis_normalized)) == np.round(total_duration_stim)

    # Save the normalized ITIs
    df.loc[df.sequence_id == sequence_id, 'resp_iti_norm'] = resp_itis_normalized

    # Save the ratio as well
    df.loc[df.sequence_id == sequence_id, 'tempo_ratio_resp_to_stim'] = tempo_ratio_resp_to_stim
    df_bytrial.loc[df.sequence_id == sequence_id, 'tempo_ratio_resp_to_stim'] = tempo_ratio_resp_to_stim

# Save the data
df.to_csv(os.path.join('data', 'experiment', 'processed', 'ITIs.csv'), index=False)
df_bytrial.to_csv(os.path.join('data', 'experiment', 'processed', 'ITIs_bytrial.csv'), index=False)
