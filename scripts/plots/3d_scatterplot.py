import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the data
df = pd.read_csv(os.path.join("data", "experiment", "processed", "pp_measures_6005.csv"))

# Make plot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

for pp_id_behav, pp_df in df.groupby("pp_id_behav"):
    G = pp_df["G_resp"].values[0]
    entropy = pp_df["entropy_diff_norm_q_avg"].values[0]
    binary_ternary = pp_df["binary_or_ternary_introduced"].values[0]
    ax.scatter(G, entropy, binary_ternary, label=pp_id_behav)

ax.set_xlabel('G')
ax.set_ylabel('Entropy')
ax.set_zlabel('Binary or Ternary')

plt.show()
