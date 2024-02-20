import pandas as pd
import thebeat
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')

ITIs = pd.read_csv(os.path.join('data', 'experiment', 'processed', 'ITIs.csv'))

fourier_df = pd.DataFrame()

for pp_id, pp_df in ITIs.groupby('pp_id_behav'):
    print(pp_id)
    for tempo, tempo_df in pp_df.groupby('stim_tempo_intended'):
        for length, length_df in tempo_df.groupby('length'):
            for sequence_id, sequence_df in length_df.groupby('sequence_id'):
                stim = sequence_df.stim_ioi.values
                resp = sequence_df.resp_iti_norm.values
                combined = np.concatenate((stim, resp))
                for stim_resp in ("stim", "resp", "combined"):
                    if stim_resp == "stim":
                        seq = thebeat.Sequence(stim)
                    elif stim_resp == "resp":
                        seq = thebeat.Sequence(resp)
                    elif stim_resp == "combined":
                        seq = thebeat.Sequence(combined)
                    else:
                        raise ValueError("Invalid stim_resp value")

                    # plot fft
                    fig, ax = thebeat.stats.fft_plot(seq, unit_size=1000, x_max=10, suppress_display=True)

                    # Get the data
                    x_data, y_data = ax.lines[0].get_data()
                    x_data, y_data = x_data[1:], y_data[1:]

                    # Close the figure
                    plt.close('all')

                    # Get the index of the highest value for y, and get its corresponding x value
                    max_y_index = np.argmax(y_data)
                    max_x = x_data[max_y_index]
                    peak = 1000 / max_x

                    # The code below is a bit cookie, but we need to find what value the found peak represents.
                    # It can e.g. be a quarternote (when someone taps isochronously), or it can be an eighth or sixteenth note
                    seq_tempo = seq.mean_ioi

                    note_values = [seq_tempo / x for x in (1, 2, 4)]
                    peak_closest_to = min(note_values, key=lambda x: abs(x - peak))
                    possibilities = (4, 2, 1)

                    sixteenth_duration = round(peak / possibilities[note_values.index(peak_closest_to)])
                    seq_q = seq.quantize_iois(to=sixteenth_duration)

                    if stim_resp == 'stim':
                        fourier_16th_duration_stim = sixteenth_duration
                    elif stim_resp == "resp":
                        fourier_16th_duration_resp = sixteenth_duration
                    elif stim_resp == "combined":
                        fourier_16th_duration_combined = sixteenth_duration

                df_piece = pd.DataFrame({
                    'pp_id': pp_id,
                    'stim_tempo_intended': tempo,
                    'length': length,
                    'sequence_id': sequence_id,
                    'fourier_16th_duration_stim': fourier_16th_duration_stim,
                    'fourier_16th_duration_resp': fourier_16th_duration_resp,
                    'fourier_16th_duration_combined': fourier_16th_duration_combined
                }, index=[0])
                fourier_df = pd.concat([fourier_df, df_piece], ignore_index=True)

fourier_df.to_csv(os.path.join('data', 'experiment', 'processed', 'fourier.csv'), index=False)
