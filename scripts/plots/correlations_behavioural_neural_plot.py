import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io
import os
from pathlib import Path

# Get the project root directory
project_root = Path(__file__).resolve().parents[2]
data_directory_path = project_root / 'data' / 'plots'

# Construct full paths to the .mat files
entropyDiff_all_behav_path = os.path.join(data_directory_path, 'entropyDiff_all_behav.mat')
entropyDiff_NegConnStrength_path = os.path.join(data_directory_path, 'entropyDiff_NegConnStrength.mat')
Gresponse_all_behav_path = os.path.join(data_directory_path, 'Gresponse_all_behav.mat')
Gresponse_NegConnStrength_path = os.path.join(data_directory_path, 'Gresponse_NegConnStrength.mat')
Binary_Ternary_all_behav_path = os.path.join(data_directory_path, 'binary_or_ternary_introduced_all_behav.mat')
Binary_Ternary_NegConnStrength_path = os.path.join(data_directory_path, 'binary_or_ternary_introduced_NegConnStrength.mat')

# Load data using the constructed paths
entropyDiff_all_behav = scipy.io.loadmat(entropyDiff_all_behav_path)
entropyDiff_NegConnStrength = scipy.io.loadmat(entropyDiff_NegConnStrength_path)
Gresponse_all_behav = scipy.io.loadmat(Gresponse_all_behav_path)
Gresponse_NegConnStrength = scipy.io.loadmat(Gresponse_NegConnStrength_path)
Binary_Ternary_all_behav = scipy.io.loadmat(Binary_Ternary_all_behav_path)
Binary_Ternary_NegConnStrength = scipy.io.loadmat(Binary_Ternary_NegConnStrength_path)

# Extract values
entropyDiff_all_behav_values = entropyDiff_all_behav[list(entropyDiff_all_behav.keys())[-1]]
entropyDiff_NegConnStrength_values = entropyDiff_NegConnStrength[list(entropyDiff_NegConnStrength.keys())[-1]]
Gresponse_all_behav_values = Gresponse_all_behav[list(Gresponse_all_behav.keys())[-1]]
Gresponse_NegConnStrength_values = Gresponse_NegConnStrength[list(Gresponse_NegConnStrength.keys())[-1]]
Binary_Ternary_all_behav_values = Binary_Ternary_all_behav[list(Binary_Ternary_all_behav.keys())[-1]]
Binary_Ternary_NegConnStrength_values = Binary_Ternary_NegConnStrength[list(Binary_Ternary_NegConnStrength.keys())[-1]]

# Plot Entropy difference vs Sum connection strength
plt.figure(figsize=(10,6))
sns.regplot(x=entropyDiff_NegConnStrength_values.squeeze(), y=entropyDiff_all_behav_values.squeeze(), 
            scatter_kws={'color':'black', 'alpha':0.7}, line_kws={'color':'blue'}, ci=95)
plt.xlabel("Sum connection strength")
plt.ylabel("Entropy difference")
plt.title("Entropy difference vs Sum connection strength")
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

# Plot G-response vs Sum connection strength
plt.figure(figsize=(10,6))
sns.regplot(x=Gresponse_NegConnStrength_values.squeeze(), y=Gresponse_all_behav_values.squeeze(), 
            scatter_kws={'color':'black', 'alpha':0.7}, line_kws={'color':'blue'}, ci=95)
plt.xlabel("Sum connection strength")
plt.ylabel("G-response")
plt.title("G-response vs Sum connection strength")
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

# Plot Binary-Ternary ratios vs Sum connection strength
plt.figure(figsize=(10,6))
sns.regplot(x=Binary_Ternary_NegConnStrength_values.squeeze(), y=Binary_Ternary_all_behav_values.squeeze(), 
            scatter_kws={'color':'black', 'alpha':0.7}, line_kws={'color':'blue'}, ci=95)
plt.xlabel("Sum connection strength")
plt.ylabel("Binary and Ternary Ratios")
plt.title("Binary and Ternary Ratios vs Sum connection strength")
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()
