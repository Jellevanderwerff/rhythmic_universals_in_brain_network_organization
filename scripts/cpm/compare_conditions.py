import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from statsmodels.stats.anova import AnovaRM

# -------------------------------
# Set the Home Directory and File Paths
# -------------------------------
# Set HOME_DIR as the directory where this script is located.
# If your files are located elsewhere, update HOME_DIR accordingly.
HOME_DIR = os.path.dirname(os.path.abspath(__file__))

files = {
    "4004": os.path.join(HOME_DIR, "pp_measures_4004.csv"),
    "4005": os.path.join(HOME_DIR, "pp_measures_4005.csv"),
    "6004": os.path.join(HOME_DIR, "pp_measures_6004.csv"),
    "6005": os.path.join(HOME_DIR, "pp_measures_6005.csv")
}

# -------------------------------
# Load the Data
# -------------------------------
# Read each file into a DataFrame and store them in a dictionary.
dfs = {}
for label, filepath in files.items():
    dfs[label] = pd.read_csv(filepath)

# Ensure that all files have the same number of rows (i.e. subjects).
n_subjects = None
for label, df in dfs.items():
    if n_subjects is None:
        n_subjects = df.shape[0]
    else:
        assert df.shape[0] == n_subjects, "All files must have the same number of subjects!"

# -------------------------------
# Define Variables and Colors
# -------------------------------
# List of behavioural variables to plot and analyze.
behaviours = [
    "G_resp",
    "entropy_diff_norm_q_avg",
    "isochrony_introduced",
    "binary_or_ternary_introduced"
]

# Define colors for each condition (file).
colors = {
    "4004": "blue",
    "4005": "green",
    "6004": "red",
    "6005": "purple"
}

# -------------------------------
# Function to compute KDE
# -------------------------------
def compute_kde(data, x_grid):
    kde = gaussian_kde(data.dropna())
    return kde(x_grid)

# -------------------------------
# Density Plots for Each Behaviour
# -------------------------------
plt.style.use('default')
n_points = 200  # number of points for density estimation

for behavior in behaviours:
    plt.figure(figsize=(8, 6))
    
    # Determine a common x-axis range across all datasets for the given behaviour.
    global_min = np.inf
    global_max = -np.inf
    for label, df in dfs.items():
        if behavior in df.columns:
            current_data = df[behavior].dropna()
            global_min = min(global_min, current_data.min())
            global_max = max(global_max, current_data.max())
    
    # Expand the x-range a little bit.
    x_min = global_min - 0.1 * abs(global_min)
    x_max = global_max + 0.1 * abs(global_max)
    x_grid = np.linspace(x_min, x_max, n_points)
    
    # Plot KDE for each condition for the current behaviour.
    for label, df in dfs.items():
        if behavior not in df.columns:
            continue
        
        data = df[behavior].dropna()
        if len(data) == 0:
            continue
        
        y = compute_kde(data, x_grid)
        plt.fill_between(x_grid, y, alpha=0.4, color=colors[label], label=f"{label}")
        plt.plot(x_grid, y, color=colors[label])
    
    plt.title(f"Density Distributions for {behavior}")
    plt.xlabel(behavior)
    plt.ylabel("Density")
    plt.legend(title="Condition")
    plt.tight_layout()
    plt.show()

# -------------------------------
# Descriptive Statistics and Repeated Measures ANOVA
# -------------------------------
# For each behaviour, compute descriptive statistics (mean, std, median) by condition
# and run a repeated-measures ANOVA (rm-ANOVA) to test for main effects of tempo and length.
for behavior in behaviours:
    print("="*80)
    print(f"\nBehavior: {behavior}")
    
    # Compute descriptive statistics for each condition.
    desc_stats = []
    for label, df in dfs.items():
        if behavior in df.columns:
            data = df[behavior].dropna()
            # Extract tempo and length from the file label, e.g., "6005" means tempo = 600 and length = 5.
            tempo = int(label[:-1])
            length = int(label[-1])
            desc_stats.append({
                "Condition": label,
                "Tempo": tempo,
                "Length": length,
                "Mean": data.mean(),
                "Std": data.std(),
                "Median": data.median(),
                "N": len(data)
            })
    desc_df = pd.DataFrame(desc_stats)
    print("\nDescriptive Statistics by Condition:")
    print(desc_df)
    
    # -------------------------------
    # Prepare Data for Repeated Measures ANOVA (Long Format)
    # -------------------------------
    # We assume that the same subjects (by row order) are in all files.
    records = []
    for label, df in dfs.items():
        if behavior not in df.columns:
            continue
        tempo = int(label[:-1])
        length = int(label[-1])
        for i in range(df.shape[0]):
            value = df.loc[i, behavior]
            records.append({
                "subject": i,
                "Condition": label,
                "Tempo": tempo,
                "Length": length,
                "measure": value
            })
    df_long = pd.DataFrame(records)
    
    # Show a preview of the long-format data for rm-ANOVA.
    print("\nPreview of long-format data for rm-ANOVA:")
    print(df_long.head())
    
    # -------------------------------
    # Run Repeated Measures ANOVA using statsmodels' AnovaRM.
    # -------------------------------
    try:
        aovrm = AnovaRM(df_long, depvar="measure", subject="subject", within=["Tempo", "Length"])
        res = aovrm.fit()
        print("\nRepeated Measures ANOVA results:")
        print(res)
    except Exception as e:
        print("Error running rm-ANOVA:", e)
    
    print("\n" + "="*80 + "\n")