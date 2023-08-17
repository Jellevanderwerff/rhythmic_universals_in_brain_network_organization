import thebeat
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import fractions

# load data
ITIs = pd.read_csv(os.path.join('data', 'experiment', 'processed', 'ITIs.csv'))

ratios_df = pd.DataFrame()

for pp_id, pp_df in ITIs.groupby('pp_id'):
    print(pp_id)
    for tempo, tempo_df in pp_df.groupby('stim_tempo_intended'):
        for length, length_df in tempo_df.groupby('length'):
            for sequence_id, sequence_df in length_df.groupby('sequence_id'):
                seq = thebeat.Sequence(sequence_df.resp_iti.values)
                interval_ratios = seq.interval_ratios_from_dyads

                # quantized interval ratios
                # plot fft
                fig, ax = thebeat.stats.fft_plot(seq, unit_size=1000, x_max=10, suppress_display=True)

                # Get the data
                x_data, y_data = ax.lines[0].get_data()

                # Close the figure
                plt.close()

                # Get the index of the highest value for y, and get its corresponding x value
                max_y_index = np.argmax(y_data)
                max_x = x_data[max_y_index]
                peak = 1000 / max_x

                # The code below is a bit cookie, but we need to find what value the found peak represents.
                # It can e.g. be a quarternote (when someone taps isochronously), or it can be an eighth or sixteenth note
                resp_tempo = seq.mean_ioi

                note_values = [resp_tempo / x for x in (1, 2, 4)]
                peak_closest_to = min(note_values, key=lambda x: abs(x - peak))
                possibilities = (4, 2, 1)

                sixteenth_duration = round(peak / possibilities[note_values.index(peak_closest_to)])
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
                    'quantized_ratio': quantized_ratios
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