import pandas as pd
import numpy as np
import os
import thebeat

"""
Here we follow the method from Roeske et al. (2020) in Current Biology, to calculate interval ratios.
"""

"""
Bin index    Bin center  Bin width    Range          Meaning
0            1/4.5       0.0247678    1/4.75—1/4.25  off ternary
1            1/4         0.03137255   1/4.25—1/3.75  on ternary
2            1/3.5       0.04102564   1/3.75—1/3.25  off binary
3            1/3         0.05594406   1/3.25—1/2.75  on binary
4            1/2.5       0.08080808   1/2.75—1/2.25  off isochrony
5            1/2         0.05555556   1/2.25—1/2     on isochrony
"""

boundaries = [1/4.75, 1/4.25, 1/3.75, 1/3.25, 1/2.75, 1/2.25, 1/2]
bin_widths = np.diff(boundaries)
isochrony_bin_index = 5
binary_bin_index = 3
ternary_bin_index = 1

# Load the data
df = pd.read_csv(os.path.join('data', 'experiment', 'processed', 'ITIs.csv'))
pp_measures = pd.read_csv(os.path.join('data', 'experiment', 'processed', 'pp_measures.csv'))
ratios_df = pd.DataFrame()

for pp_id_behav, pp_df in df.groupby('pp_id_behav'):
    print(pp_id_behav)
    for tempo, tempo_df in pp_df.groupby('stim_tempo_intended'):
        for length, length_df in tempo_df.groupby('length'):
            pp_stim_ratios = []
            pp_resp_ratios = []
            for sequence_id, sequence_df in length_df.groupby('sequence_id'):
                stim_iois = sequence_df.stim_ioi.values
                resp_itis = sequence_df.resp_iti.values
                # Make thebeat Sequences
                stim = thebeat.Sequence(stim_iois)
                resp = thebeat.Sequence(resp_itis)
                # Get interval ratios from dyads
                stim_interval_ratios = stim.interval_ratios_from_dyads
                resp_interval_ratios = resp.interval_ratios_from_dyads
                # 'Aggregate'. If any of the interval ratios is larger than 0.5, we take the inverse
                stim_interval_ratios = [ratio if ratio <= 0.5 else 1 - ratio for ratio in stim_interval_ratios]
                resp_interval_ratios = [ratio if ratio <= 0.5 else 1 - ratio for ratio in resp_interval_ratios]
                # Add to list
                pp_stim_ratios.extend(stim_interval_ratios)
                pp_resp_ratios.extend(resp_interval_ratios)

            # Discretize
            stim_ratios_freqs = np.histogram(pp_stim_ratios, bins=boundaries)
            resp_ratios_freqs = np.histogram(pp_resp_ratios, bins=boundaries)
            # Normalize according to bin width
            stim_ratios_freqs_norm = stim_ratios_freqs[0] / (len(pp_stim_ratios) * bin_widths)
            resp_ratios_freqs_norm = resp_ratios_freqs[0] / (len(pp_resp_ratios) * bin_widths)
            # Count isochrony
            isochrony_freq_stim = stim_ratios_freqs_norm[isochrony_bin_index]
            isochrony_freq_resp = resp_ratios_freqs_norm[isochrony_bin_index]
            # Count binary
            binary_freq_stim = stim_ratios_freqs_norm[binary_bin_index]
            binary_freq_resp = resp_ratios_freqs_norm[binary_bin_index]
            # Count ternary
            ternary_freq_stim = stim_ratios_freqs_norm[ternary_bin_index]
            ternary_freq_resp = resp_ratios_freqs_norm[ternary_bin_index]
            # Count additions/subtractions
            isochrony_introduced = isochrony_freq_resp - isochrony_freq_stim
            binary_introduced = binary_freq_resp - binary_freq_stim
            ternary_introduced = ternary_freq_resp - ternary_freq_stim
            binary_or_ternary_introduced = binary_introduced + ternary_introduced
            simple_ratios_introduced = binary_or_ternary_introduced + isochrony_introduced

            # Make little dataframe
            df_piece = pd.DataFrame({
                'pp_id': df[df.sequence_id == sequence_id].pp_id_behav.values[0],
                'stim_tempo_intended': tempo,
                'length': length,
                'isochrony_introduced': isochrony_introduced,
                'binary_introduced': binary_introduced,
                'ternary_introduced': ternary_introduced,
                'binary_or_ternary_introduced': binary_or_ternary_introduced,
                'simple_ratios_introduced': simple_ratios_introduced,
            }, index=[0])

            # Add to ratios_df
            ratios_df = pd.concat([ratios_df, df_piece], ignore_index=True)

# Save
ratios_df.to_csv(os.path.join('data', 'experiment', 'processed', 'ratios_introduced.csv'), index=False)
