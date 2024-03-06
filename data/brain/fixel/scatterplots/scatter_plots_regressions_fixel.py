import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Paths to the data files
residuals_file = '/Users/au540169/Dropbox/Progetto Aarhus/project_fMRI2022:23PredictorRhythm/analysis/fixelbased/analysis/partial_residuals_regression_X_subcortical-cortical.csv'
response_file = '/Users/au540169/Dropbox/Progetto Aarhus/project_fMRI2022:23PredictorRhythm/analysis/fixelbased/analysis/by_pp_updated_27_01_2024.csv'

# Load the data
residuals_df = pd.read_csv(residuals_file)
response_df = pd.read_csv(response_file)

# variables:
# simple_ratio_introduced_avg = simple ratios
# G_resp = G response
# asynchrony_norm_abs_avg
# entropy_diff_norm_q_avg = entropy difference
# iti_ioi_cov_diff_avg = coefficient of variation
# tempo_deviation_abs_avg = tempo deviation

# Specific variable names
x_variable = 'isochrony_introduced_600_5_ST_PREC_fd_X_partial_resid'
y_variable = 'isochrony_introduced_600_5'

# Extracting the relevant data
x_data = residuals_df[x_variable]
y_data = response_df[y_variable]

# Define labels
y_label = 'Isochrony (600 ms 5 events)'
x_label = 'ST_PREC fd (partial residuals)'

# Label font size
label_font_size = 18  # Adjust this value as needed

# Create the scatter plot with regression line and 95% confidence interval
plt.figure(figsize=(10, 6))
sns.regplot(x=x_data, y=y_data, ci=95, 
            scatter_kws={'alpha':0.5, 'color': 'black'}, 
            line_kws={'color': 'grey'})
plt.xlabel(x_label, fontsize=label_font_size)
plt.ylabel(y_label, fontsize=label_font_size)
plt.show()
