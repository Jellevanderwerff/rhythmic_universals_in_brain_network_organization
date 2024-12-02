import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from pathlib import Path


# Get the project root directory
project_root = Path(__file__).resolve().parents[2]
data_path = project_root / 'data' / 'CPM' / 'behaviour'

# Load the data from the Excel file

# G_response: G-responsePredictedActualplot.xlsx
# Entropy diff: EntropyDiffPredictedActualplot.xlsx
# binary or ternary introduced: Binary_or_TernaryPredictedActualplot.xlsx

data = pd.read_excel(data_path / 'Binary_or_TernaryPredictedActualplot.xlsx')

# Calculate the Pearson correlation coefficient and the p-value
r_value, p_value = pearsonr(data["Predicted"], data["Actual"])

# Set up the figure and axis
plt.figure(figsize=(10, 6))

# Plot a scatter plot with regression line and 95% CI
sns.regplot(x="Predicted", y="Actual", data=data, ci=95, scatter_kws={"color": "black"}, line_kws={"color": "grey"})

# Increase the size of title, x-label, and y-label
plt.title("Binary or Ternary Ratios", fontsize=18, fontweight='bold')
plt.xlabel("Predicted", fontsize=15)
plt.ylabel("Actual", fontsize=15)

# Add the R and p-values to the bottom right of the figure with increased font size
plt.text(0.95*data["Predicted"].max(), 0.20*data["Actual"].max(), 
         f'R = {r_value:.2f}\np = {p_value:.4f}', 
         horizontalalignment='right', fontsize=16)

plt.tight_layout()
plt.show()
