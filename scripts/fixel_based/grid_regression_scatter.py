import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import logging
from matplotlib.ticker import MaxNLocator  # **Added Import**

# Configure logging for better debugging and information
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the condition and metric type for the plot
condition = '600_5'
metric_type = 'fdc'

# Get the project root directory
project_root = Path(__file__).resolve().parents[2]
data_path = project_root / 'data' / 'brain' / 'fixel_based'        # Ensure this matches your actual directory name
results_path = project_root / 'results' / 'fixel_based'  # Ensure this matches your actual directory name
plots_path = project_root / 'plots' / 'fixel_based'

# Ensure that the results_path exists
results_path.mkdir(parents=True, exist_ok=True)
logging.info(f"Results directory set to: {results_path}")

# Define the file paths
data_file = data_path / 'all_bundle_mask_average_subcortical-cortical.csv'
residuals_file = results_path / f'{condition}_partial_residuals_regression_X.csv'

# Check if data files exist
if not data_file.exists():
    raise FileNotFoundError(f"Data file does not exist: {data_file}")

if not residuals_file.exists():
    raise FileNotFoundError(f"Residuals file does not exist: {residuals_file}")

# Load your data
df_all_bundle_mask = pd.read_csv(data_file)
df_residuals = pd.read_csv(residuals_file)

logging.info(f"Loaded data file with shape: {df_all_bundle_mask.shape}")
logging.info(f"Loaded residuals file with shape: {df_residuals.shape}")

# Define white-matter tracts and behaviors
tracts = ['CP', 'ST_PREC', 'ST_PREM', 'T_PREC', 'T_PREM']
behaviors = ['G_resp', 'entropy_diff_norm_q_avg', 'binary_or_ternary_introduced', 'isochrony_introduced']

# Calculate r values and p-values
r_values = []
p_values = []

for tract in tracts:
    for behavior in behaviors:
        tract_column = f'{tract}_{metric_type}'
        residuals_column = f'{behavior}_{condition}_{tract}_{metric_type}_partial_X_resid'

        # Check if columns exist
        if tract_column not in df_all_bundle_mask.columns:
            logging.error(f"Column '{tract_column}' not found in 'df_all_bundle_mask'")
            raise KeyError(f"Column '{tract_column}' not found in 'df_all_bundle_mask'")

        if residuals_column not in df_residuals.columns:
            logging.error(f"Column '{residuals_column}' not found in 'df_residuals'")
            raise KeyError(f"Column '{residuals_column}' not found in 'df_residuals'")

        x_values = df_all_bundle_mask[tract_column]
        y_values = df_residuals[residuals_column]

        # Calculate Pearson correlation
        corr, p_value = pearsonr(x_values, y_values)
        r_values.append(corr)
        p_values.append(p_value)

logging.info(f"Calculated {len(r_values)} correlation values and p-values.")

# Apply FDR correction (Benjamini-Hochberg method)
_, p_fdr_adjusted, _, _ = multipletests(p_values, method='fdr_bh')
logging.info("Applied FDR correction to p-values.")

# Define the file name for the PDF output within results_path
pdf_filename = plots_path / f'{condition}_{metric_type}_grid_scatter.pdf'
logging.info(f"PDF will be saved as: {pdf_filename}")

# Create a PdfPages object to save the plots in a single PDF
with PdfPages(pdf_filename) as pdf:
    # Initialize index for accessing precomputed r-values and pFDR-adjusted values
    index = 0

    # Plot the grid with scatterplots and regression lines
    fig, axes = plt.subplots(len(tracts), len(behaviors), figsize=(20, 25))
    fig.suptitle(f'Scatterplot with Regression Lines, r, and pFDR for {condition} and {metric_type}', fontsize=16)

    for i, tract in enumerate(tracts):
        for j, behavior in enumerate(behaviors):
            tract_column = f'{tract}_{metric_type}'
            residuals_column = f'{behavior}_{condition}_{tract}_{metric_type}_partial_X_resid'

            x_values = df_all_bundle_mask[tract_column]
            y_values = df_residuals[residuals_column]

            corr = r_values[index]
            p_fdr = p_fdr_adjusted[index]
            index += 1

            line_color = 'blue' if corr < 0 else 'red'

            ax = axes[i, j]
            sns.regplot(
                x=x_values,
                y=y_values,
                ax=ax,
                line_kws={'color': line_color},
                scatter_kws={'s': 50, 'alpha': 0.6}
            )

            ax.set_title(f'{tract} vs {behavior}', fontsize=10)
            ax.set_xlabel(f'{tract}_{metric_type}')
            ax.set_ylabel(f'{behavior} Residuals')

            #ax.annotate(
            #    f'r = {corr:.2f}, pFDR = {p_fdr:.3f}',
            #     xy=(0.7, 0.9),
            #    xycoords='axes fraction',
            #    fontsize=10,
            #    color='blue'
            #)

            # **Remove the Grid**
            ax.grid(False)

            # **Limit the Number of Ticks**
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

            # **Increase the Size of the Tick Labels**
            ax.tick_params(axis='both', which='major', labelsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save the figure to the PDF
    pdf.savefig(fig)
    plt.close(fig)
    logging.info(f'PDF saved successfully at: {pdf_filename}')

print(f'PDF saved as {pdf_filename}')