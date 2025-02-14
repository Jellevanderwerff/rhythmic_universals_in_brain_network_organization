import thebeat
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import fractions

# load data
ITIs = pd.read_csv(os.path.join('data', 'experiment', 'processed', 'ITIs.csv'))
fourier_df = pd.read_csv(os.path.join('data', 'experiment', 'processed', 'fourier.csv'))

for pp_id, pp_df in ITIs.groupby('pp_id_behav'):
    print(pp_id)
    for tempo, tempo_df in pp_df.groupby('stim_tempo_intended'):
        for length, length_df in tempo_df.groupby('length'):
            for sequence_id, sequence_df in length_df.groupby('sequence_id'):
                for stim_resp in ('stim_ioi', 'resp_iti_norm'):
                    # Make thebeat sequence
                    seq = thebeat.Sequence(sequence_df[stim_resp].values)
                    # Get the fourier tempo
                    if stim_resp == 'stim_ioi':
                        sixteenth_duration = fourier_df[fourier_df.sequence_id == sequence_id]['fourier_16th_duration_stim'].values[0]
                        sixteenth_duration_stim = sixteenth_duration
                    else:
                        sixteenth_duration = fourier_df[fourier_df.sequence_id == sequence_id]['fourier_16th_duration_resp'].values[0]
                        sixteenth_duration_resp = sixteenth_duration
                    # Do the quantization
                    seq_q = seq.quantize_iois(to=sixteenth_duration)
                    # Save quantized IOIs
                    if stim_resp == 'stim_ioi':
                        ITIs.loc[ITIs.sequence_id == sequence_id, 'stim_ioi_q'] = seq_q.iois
                    else:
                        ITIs.loc[ITIs.sequence_id == sequence_id, 'resp_iti_norm_q'] = seq_q.iois

# Save
ITIs.to_csv(os.path.join('data', 'experiment', 'processed', 'ITIs.csv'), index=False)
