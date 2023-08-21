import thebeat
import matplotlib.pyplot as plt
import os
import pandas as pd


s = thebeat.SoundStimulus.from_wav(os.path.join('illustrations', '14.wav'))
fig, ax = s.plot_waveform()
fig.savefig(os.path.join('illustrations', '14.eps'))

s = thebeat.SoundStimulus.from_wav(os.path.join('illustrations', '14-stim.wav'))
fig, ax = s.plot_waveform()
fig.savefig(os.path.join('illustrations', '14-stim.eps'))

df = pd.read_csv(os.path.join('data', 'experiment', 'processed', 'ITIs.csv'))
itis = df[df.sequence_id == '20_14'].resp_iti.values
seq = thebeat.Sequence(itis)

fig, ax = thebeat.stats.fft_plot(seq, 1000, x_max=10)

fig.savefig(os.path.join('illustrations', '14-fft.eps'))
