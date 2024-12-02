import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.stats.multitest as smm
from pathlib import Path

# Get the project root directory
project_root = Path(__file__).resolve().parents[2]
data_path = project_root / 'data' / 'fixel_based'
results_path = project_root / 'results' / 'fixel_based'

# Create the results directory if it doesn't exist
results_path.mkdir(parents=True, exist_ok=True)

# Load your datasets
design_matrix = pd.read_csv(data_path / "design_matrix.csv")
by_pp_updated = pd.read_csv(data_path / "by_pp_updated_27_01_2024.csv")
all_bundle_mask_average = pd.read_csv(data_path / "all_bundle_mask_average_subcortical-cortical.csv")

# Ensure 'subject_id' is of the same type in all dataframes
design_matrix['subject_id'] = design_matrix['subject_id'].astype(str)
by_pp_updated['subject_id'] = by_pp_updated['subject_id'].astype(str)
all_bundle_mask_average['subject_id'] = all_bundle_mask_average['subject_id'].astype(str)

# Merge your datasets
merged_data = pd.merge(design_matrix, by_pp_updated, on='subject_id')
merged_data = pd.merge(merged_data, all_bundle_mask_average, on='subject_id')

# List of behavioral variables and conditions to analyze
behavioral_vars = ["G_resp", "entropy_diff_norm_q_avg", "binary_or_ternary_introduced", "isochrony_introduced"]
conditions = ["600_5"]

# Metrics types
metrics_types = ["fd", "log_fc", "fdc"]

# Base metric names (without type suffix)
base_metrics = ["T_PREM", "T_PREC", "ST_PREM", "ST_PREC", "CP"]

# Specify covariates for 'log_fc' and 'fdc' metrics
covariates = ['log_eITV', 'rel_head_motion']
covariates_fd = ['rel_head_motion']

# Function to perform regression for a given variable and metric, returning the results and residuals
def perform_regression(variable, metric, metric_type):
    # Select covariates based on metric type
    if metric_type == 'fd':
        X = merged_data[[metric] + covariates_fd]
    else:  # 'log_fc' and 'fdc'
        X = merged_data[[metric] + covariates]

    y = merged_data[variable]

    # Adding a constant to the model (for intercept)
    X = sm.add_constant(X)

    # Fitting the model
    model = sm.OLS(y, X).fit()

    # Extracting additional metrics
    coef = model.params[metric]
    se = model.bse[metric]
    t_stat = model.tvalues[metric]
    conf_int = model.conf_int().loc[metric].tolist()  # Convert to list
    f_stat = model.fvalue
    f_pvalue = model.f_pvalue
    adj_rsq = model.rsquared_adj

    # Get residuals
    y_residuals = model.resid

    return {
        "P-Value": model.pvalues[metric],
        "Coefficient": coef,
        "Standard Error": se,
        "T-Statistic": t_stat,
        "Confidence Interval": conf_int,
        "F-Statistic": f_stat,
        "Adjusted R-Squared": adj_rsq
    }, y_residuals

# Initialize a list to store all regression results
results_list = []

# Initialize lists to store partial residuals
partial_residuals_list = []

# Running the analysis
for behav_var in behavioral_vars:
    for metric_type in metrics_types:
        for condition in conditions:
            for base_metric in base_metrics:
                adjusted_metric = base_metric + "_" + metric_type
                if adjusted_metric not in merged_data.columns:
                    print(f"Column {adjusted_metric} not found in merged_data")
                    continue

                try:
                    results, y_residuals = perform_regression(f"{behav_var}_{condition}", adjusted_metric, metric_type)
                except Exception as e:
                    print(f"Error in regression for {behav_var}_{condition} with {adjusted_metric}: {e}")
                    continue

                # Print Behavioral Variable if P-Value is less than 0.05
                if results["P-Value"] < 0.05:
                    print(f"Significant result: Behavioral Variable: {behav_var}, Condition: {condition}, Metric Type: {metric_type}, Base Metric: {base_metric}, P-Value: {results['P-Value']}")

                # Append results to list
                results_list.append({
                    "Behavioral Variable": behav_var,
                    "Condition": condition,
                    "Metric Type": metric_type,
                    "Base Metric": base_metric,
                    "P-Value": results["P-Value"],
                    "Coefficient": results["Coefficient"],
                    "Standard Error": results["Standard Error"],
                    "T-Statistic": results["T-Statistic"],
                    "Confidence Interval": results["Confidence Interval"],
                    "F-Statistic": results["F-Statistic"],
                    "Adjusted R-Squared": results["Adjusted R-Squared"]
                })

                # Compute residuals for X_j
                if metric_type == 'fd':
                    current_covariates = covariates_fd
                else:
                    current_covariates = covariates

                # Fit model for X_j with covariates: X_j ~ covariates
                X_covariates = sm.add_constant(merged_data[current_covariates])
                model_covariates = sm.OLS(merged_data[adjusted_metric], X_covariates).fit()

                # Calculate residuals for X_j
                residual_X_j = merged_data[adjusted_metric] - model_covariates.predict(X_covariates)

                # Calculate partial residuals
                partial_residuals_X = y_residuals + results["Coefficient"] * residual_X_j

                # Store partial residuals
                partial_residuals_col_name = f"{behav_var}_{condition}_{base_metric}_{metric_type}_partial_X_resid"
                partial_residuals_series = pd.Series(partial_residuals_X, name=partial_residuals_col_name)
                partial_residuals_list.append(partial_residuals_series)

# Convert results list to DataFrame once after loop
results_df = pd.DataFrame(results_list)

# Assuming 'conditions' is a list with one element
selected_condition = conditions[0]

# Save the results DataFrame to a CSV file
results_df.to_csv(results_path / f"{selected_condition}_regression_results.csv", index=False)

# Concatenate all partial residuals and save to CSV file
partial_residuals_df = pd.concat(partial_residuals_list, axis=1)
partial_residuals_df.to_csv(results_path / f"{selected_condition}_partial_residuals_regression_X.csv", index=False)

# Display the first few rows of the results DataFrame
print(results_df.head())

# Load the 'regression_results.csv' just created
data = pd.read_csv(results_path / f"{selected_condition}_regression_results.csv")

# Pivot the table to get the desired format
pivot_data_full = data.pivot_table(
    index='Base Metric',
    columns=['Behavioral Variable', 'Condition', 'Metric Type'],
    values='P-Value'
)

# Separate tables for fd, fdc, and log_fc
fd_table_full = pivot_data_full.xs('fd', level='Metric Type', axis=1)
fdc_table_full = pivot_data_full.xs('fdc', level='Metric Type', axis=1)
log_fc_table_full = pivot_data_full.xs('log_fc', level='Metric Type', axis=1)

# Function to adjust p-values using FDR correction
def adjust_p_values(df):
    pvals = df.values.flatten()
    adjusted_pvals = smm.multipletests(pvals, method='fdr_bh')[1]
    adjusted_df = pd.DataFrame(adjusted_pvals.reshape(df.shape), index=df.index, columns=df.columns)
    return adjusted_df

# Adjust p-values for each table
fd_table_adj_full = adjust_p_values(fd_table_full)
fdc_table_adj_full = adjust_p_values(fdc_table_full)
log_fc_table_adj_full = adjust_p_values(log_fc_table_full)

# Save the original and adjusted tables to CSV files
#fd_table_full.to_csv(results_path / f"{selected_condition}_fd_p_values.csv", index=True)
#fdc_table_full.to_csv(results_path / f"{selected_condition}_fdc_p_values.csv", index=True)
#log_fc_table_full.to_csv(results_path / f"{selected_condition}_log_fc_p_values.csv", index=True)

fd_table_adj_full.to_csv(results_path / f"{selected_condition}_fd_p_values_adjusted.csv", index=True)
fdc_table_adj_full.to_csv(results_path / f"{selected_condition}_fdc_p_values_adjusted.csv", index=True)
log_fc_table_adj_full.to_csv(results_path / f"{selected_condition}_log_fc_p_values_adjusted.csv", index=True)