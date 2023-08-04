import thebeat
import pandas as pd
import os

# load data
ITIs = pd.read_csv(os.path.join('data', 'pilot', 'processed', 'ITIs.csv'))

ratios_df = pd.DataFrame()

for pp_id, pp_df in ITIs.groupby('pp_id'):
    for tempo, tempo_df in pp_df.groupby('stim_tempo_intended'):
        for length, length_df in tempo_df.groupby('length'):
            for sequence_id, sequence_df in length_df.groupby('sequence_id'):
                seq = thebeat.Sequence(sequence_df.resp_iti.values)
                interval_ratios = seq.interval_ratios_from_dyads
                df_piece = pd.DataFrame({
                    'pp_id': pp_id,
                    'stim_tempo_intended': tempo,
                    'length': length,
                    'sequence_id': sequence_id,
                    'interval_ratio_i': list(range(1, len(interval_ratios)+1)),
                    'interval_ratio': interval_ratios
                })
                ratios_df = pd.concat([ratios_df, df_piece])

# write out
ratios_df.to_csv(os.path.join('data', 'pilot', 'processed', 'interval_ratios.csv'), index=False)