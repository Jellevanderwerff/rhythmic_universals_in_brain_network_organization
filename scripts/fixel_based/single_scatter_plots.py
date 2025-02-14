import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pathlib import Path

# Get the project root directory
project_root = Path(__file__).resolve().parents[2]
results_path = project_root / 'results' / 'fixel_based'
data_path = project_root / 'data' / 'brain' / 'fixel_based'
plots_path = project_root / 'plots' / 'fixel_based'

# load condition
conditions = ["600_5"]
selected_condition = conditions[0]

# Paths to the data files
partial_residuals_file = results_path / f"{selected_condition}_partial_residuals_regression_X.csv"
observed_response_file = data_path / "by_pp_updated_27_01_2024.csv"  # Update this path as needed

# Load data
residuals_df = pd.read_csv(partial_residuals_file)
all_bundle_mask_average = pd.read_csv(data_path / "all_bundle_mask_average_subcortical-cortical.csv")
observed_response_df = pd.read_csv(observed_response_file)  # Load observed response data

# Specific variable names
x_variable = 'ST_PREC_fd'
y_partial_residual = 'isochrony_introduced_600_5_ST_PREC_fd_partial_X_resid'
y_observed_response = 'isochrony_introduced_600_5'

# Define labels
labels = {
    'partial_residuals': {
        'y_label': 'Isochrony Introduced (Partial Residuals)',
        'filename_suffix': '_part_res_scatter.pdf'
    },
    'observed_response': {
        'y_label': 'Isochrony Introduced (Observed)',
        'filename_suffix': '_scatter.pdf'
    }
}

# Common plot settings
label_font_size = 30
brown_color = '#A67C52'

def create_scatter_plot(x, y, y_label, x_label, filename):
    plt.figure(figsize=(12, 10))
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)

    sns.regplot(
        x=x, y=y, ci=95,
        scatter_kws={'alpha': 0.5, 'color': brown_color, 's': 120},
        line_kws={'color': brown_color}
    )

    plt.xlabel(x_label, fontsize=label_font_size, weight='bold', labelpad=20)
    plt.ylabel(y_label, fontsize=label_font_size, weight='bold', labelpad=20)

    # Remove the top and right frames
    sns.despine()

    # Remove the x and y axis lines, leaving only the values
    ax = plt.gca()
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_visible(False)

    # Limit the number of ticks on each axis
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

    # Increase the size of the tick labels
    plt.tick_params(axis='both', which='major', labelsize=16)

    # Save the figure as PDF in results_path with tight layout
    plt.savefig(plots_path / filename, format='pdf', bbox_inches='tight', pad_inches=0.3)
    plt.close()

# Extract data for partial residuals plot
x_data_partial = all_bundle_mask_average[x_variable]
y_data_partial = residuals_df[y_partial_residual]

# Extract data for observed response plot
x_data_observed = all_bundle_mask_average[x_variable]
y_data_observed = observed_response_df[y_observed_response]

# Define x label
x_label = 'Striatal-Precentral Fiber Density'

# Create and save partial residuals scatter plot
create_scatter_plot(
    x=x_data_partial,
    y=y_data_partial,
    y_label=labels['partial_residuals']['y_label'],
    x_label=x_label,
    filename=f"{selected_condition}_isochrony_{x_variable}{labels['partial_residuals']['filename_suffix']}"
)

# Create and save observed response scatter plot
create_scatter_plot(
    x=x_data_observed,
    y=y_data_observed,
    y_label=labels['observed_response']['y_label'],
    x_label=x_label,
    filename=f"{selected_condition}_isochrony_{x_variable}{labels['observed_response']['filename_suffix']}"
)