import thebeat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Rhythmic universals

df = pd.read_csv('ITIs.csv')

for length in (4, 5):

    for tempo in (400, 600):

        seq_ids = np.unique(df['sequence_id'].loc[df['stim_tempo_intended'] == tempo].loc[df['length'] == length])

        seqs = []

        for i in seq_ids:
            df_piece = df.loc[df['sequence_id'] == i]
            iois = df_piece['resp_iti'].values
            seq = thebeat.Sequence(iois=iois)
            seqs.append(seq)

        interval_ratios = np.concatenate([seq.interval_ratios_from_dyads for seq in seqs])

        with plt.style.context('seaborn'):
            fig, ax = plt.subplots(dpi=600)
            ax.hist(interval_ratios, bins=100)
            fig.suptitle('Dyadic interval ratios (response)')
            ax.set_title(f'{tempo} ms, {length} events')
            ax.axvline(0.33, linestyle='--')
            ax.axvline(0.5, linestyle='--')
            ax.axvline(0.66, linestyle='--')
            fig.show()
            fig.savefig(f'itis_intervalratios_{tempo}_{length}.png')
