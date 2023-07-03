import thebeat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Rhythmic universals

df = pd.read_csv('ITIs.csv')
print(df.head(10))


seq_ids = np.unique(df['sequence_id'].loc[df['stim_tempo_intended'] == 400])

seqs = []

for id in seq_ids:
    df_piece = df.loc[df['sequence_id'] == id]
    iois = df_piece['stim_ioi'].values
    seq = thebeat.Sequence(iois=iois)
    seqs.append(seq)

interval_ratios = np.concatenate([seq.interval_ratios_from_dyads for seq in seqs])

with plt.style.context('seaborn'):
    fig, ax = plt.subplots(dpi=600)
    ax.hist(interval_ratios, bins=100)
    ax.set_title('Dyadic interval ratios (stimulus)')
    ax.axvline(0.33, linestyle='--')
    ax.axvline(0.5, linestyle='--')
    ax.axvline(0.66, linestyle='--')
    fig.show()
    fig.savefig('iois_intervalratios_histogram.png')
