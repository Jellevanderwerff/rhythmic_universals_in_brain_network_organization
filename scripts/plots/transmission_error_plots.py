import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

# Get the project root directory
project_root = Path(__file__).resolve().parents[2]
data_directory_path = project_root / 'data' / 'CPM' / 'behaviour'

# Load the CSV files
network_strength_df = pd.read_csv(data_directory_path / 'network_strength.csv')
pp_measures_df = pd.read_csv(data_directory_path / 'pp_measures_6005.csv')

# Z-score scaling function
def z_score_scaling(series):
    return (series - series.mean()) / series.std()

# Apply the z-score scaling to the behavioral columns excluding 'condition'
pp_measures_scaled_df = pp_measures_df.copy()
for column in pp_measures_df.columns:
    if pp_measures_df[column].dtype in ['float64', 'int64']:  # only scale numeric columns
        pp_measures_scaled_df[column] = z_score_scaling(pp_measures_df[column])

# Set the second behavior that will be used in all plots
behavior_1_col = "edit_distance_norm_q_avg"

# Generate the plots with a second y-axis for edit distance, save to PDF
for column in network_strength_df.columns:
    # Check if the column follows the naming pattern and extract the corresponding behavior name
    if column.startswith("train_sumneg_") and column.endswith("_6005"):
        behavior_name = column.split("train_sumneg_")[-1].split("_6005")[0]
        
        if behavior_name in pp_measures_scaled_df.columns:
            # Extract data for network strength and the corresponding behavior from pp_measures_scaled_df
            network_strength = network_strength_df[column]
            behavior_2 = pp_measures_scaled_df[behavior_name]
            behavior_1 = pp_measures_scaled_df[behavior_1_col]
            
            # Set plot size and initialize figure and axis
            fig, ax1 = plt.subplots(figsize=(8, 6))
            
            # Plot network strength vs behavior 2 on the primary y-axis (semi-transparent)
            ax1.scatter(network_strength, behavior_2, color='orange', alpha=0.2)
            coef_2 = np.polyfit(network_strength, behavior_2, 1)
            poly_2 = np.poly1d(coef_2)
            ax1.plot(network_strength, poly_2(network_strength), color='orange', linestyle='--', alpha=0.5)
            ax1.set_xlabel(column)
            ax1.set_ylabel(f"Scaled {behavior_name}", color='orange')
            ax1.tick_params(axis='y', labelcolor='black')
            
            # Create a second y-axis for edit distance (behavior_1)
            ax2 = ax1.twinx()
            ax2.scatter(network_strength, behavior_1, color='blue')
            coef_1 = np.polyfit(network_strength, behavior_1, 1)
            poly_1 = np.poly1d(coef_1)
            ax2.plot(network_strength, poly_1(network_strength), color='blue', linestyle='--')
            ax2.set_ylabel("Scaled edit distance (edit_distance_norm_q_avg)", color='blue')
            ax2.tick_params(axis='y', labelcolor='black')
            
            # Ensure both y-axes are aligned by setting the same limits
            ax1.set_ylim(min(min(behavior_2), min(behavior_1)), max(max(behavior_2), max(behavior_1)))
            ax2.set_ylim(min(min(behavior_2), min(behavior_1)), max(max(behavior_2), max(behavior_1)))
            
            # Title, no grid, and save the plot to a PDF
            plt.title(f"Scatter plot of {column} with scaled {behavior_name} and edit distance")
            plt.grid(False)
            
            # Save the plot to a PDF in the specified directory
            pdf_name = data_directory_path / f"edit_distance_{behavior_name}_6005.pdf"
            plt.savefig(pdf_name, format='pdf')
            
            # Show the plot
            plt.show()