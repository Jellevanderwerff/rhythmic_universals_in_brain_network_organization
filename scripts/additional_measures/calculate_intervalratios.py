import thebeat
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import fractions

# load data
ITIs = pd.read_csv(os.path.join('data', 'experiment', 'processed', 'ITIs.csv'))
fourier_df = pd.read_csv(os.path.join('data', 'experiment', 'processed', 'fourier.csv'))

ratios_df = pd.DataFrame()


for pp_id, pp_df in ITIs.groupby('pp_id'):
    print(pp_id)
    for tempo, tempo_df in pp_df.groupby('stim_tempo_intended'):
        for length, length_df in tempo_df.groupby('length'):
            for sequence_id, sequence_df in length_df.groupby('sequence_id'):
                for stim_resp in ('stim_ioi', 'resp_iti'):
                    seq = thebeat.Sequence(sequence_df[stim_resp].values)
                    interval_ratios = seq.interval_ratios_from_dyads

                    if stim_resp == 'stim_ioi':
                        sixteenth_duration = fourier_df[fourier_df.sequence_id == sequence_id]['fourier_16th_duration_stim'].values[0]
                        sixteenth_duration_stim = sixteenth_duration
                    else:
                        sixteenth_duration = fourier_df[fourier_df.sequence_id == sequence_id]['fourier_16th_duration_resp'].values[0]
                        sixteenth_duration_resp = sixteenth_duration
                    seq_q = seq.quantize_iois(to=sixteenth_duration)
                    quantized_ratios = seq_q.interval_ratios_from_dyads

                df_piece = pd.DataFrame({
                    'pp_id': pp_id,
                    'stim_tempo_intended': tempo,
                    'length': length,
                    'sequence_id': sequence_id,
                    'interval_ratio_i': list(range(1, len(interval_ratios)+1)),
                    'interval_ratio': interval_ratios,
                    'quantized_ratio_i': list(range(1, len(quantized_ratios)+1)),
                    'quantized_ratio': quantized_ratios,
                    'fourier_16th_duration_stim': sixteenth_duration_stim,
                    'fourier_16th_duration_resp': sixteenth_duration_resp
                })
                ratios_df = pd.concat([ratios_df, df_piece])


# Below is to add a string such as 1:2, to avoid confusion:
ratios = ratios_df['quantized_ratio']
ratios_other = 1 - ratios

output = []

for ratio, ratio_other in zip(ratios, ratios_other):
    print(ratio)
    fr = fractions.Fraction(ratio / ratio_other).limit_denominator(100)
    output.append(f"{fr.numerator}:{fr.denominator}")

ratios_df['quantized_ratio_str'] = output
ratios_df.to_csv('data/experiment/processed/interval_ratios.csv', index=False)
