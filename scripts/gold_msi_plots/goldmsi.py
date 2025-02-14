import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from pathlib import Path

# Function to validate and handle unmapped values
def validate_mappings(data, behavior, mappings):
    unique_values = data[behavior].unique()
    mapped_values = mappings.get(behavior, {})
    unmapped = [val for val in unique_values if val not in mapped_values]
    if unmapped:
        print(f"Warning: The following values in '{behavior}' do not have mappings and will be set to 'Unknown': {unmapped}")
        data[f'{behavior}_mapped'] = data[behavior].replace(mappings[behavior])
        data[f'{behavior}_mapped'].fillna('Unknown', inplace=True)
    else:
        data[f'{behavior}_mapped'] = data[behavior].replace(mappings[behavior])

    # Ensure all mapped values are strings
    data[f'{behavior}_mapped'] = data[f'{behavior}_mapped'].astype(str)
    return data

# Get the project root directory
project_root = Path(__file__).resolve().parents[2]
data_path = project_root / 'data' / 'brain' / 'gold-msi'
plots_path = project_root / 'plots' / 'gold-msi'

# Load the data from Excel
file_path = data_path / "goldmsi-training.xlsx"

# Error handling for file reading
try:
    data = pd.read_excel(file_path)
except FileNotFoundError:
    sys.exit(f"Error: The file {file_path} does not exist.")
except Exception as e:
    sys.exit(f"An unexpected error occurred while reading the Excel file: {e}")

# Define mappings for various behaviors
mappings = {
    'msi32': {1: '0', 2: '1', 3: '2', 4: '3', 5: '4-5', 6: '6-9', 7: '10 or more'},  # Duration of practice
    'msi33': {1: '0', 2: '0.5', 3: '1', 4: '1.5', 5: '2', 6: '3-4', 7: '5 or more'},  # Hours of Daily Practice
    'msi35': {1: '0', 2: '0.5', 3: '1', 4: '2', 5: '3', 6: '4-6', 7: '7 or more'},  # Music theory
    'msi36': {1: '0', 2: '0.5', 3: '1', 4: '2', 5: '3-5', 6: '6-9', 7: '10 or more'},  # Years of formal training
    'msi37': {1: '0', 2: '1', 3: '2', 4: '3', 5: '4', 6: '5', 7: '6 or more'}  # Instruments Played
}

# Specify the behavior to analyze (e.g., 'msi36')
behavior = 'msi35'  # Change this variable to switch behaviors

# Validate mappings and handle unmapped values
if behavior not in mappings:
    sys.exit(f"Error: No mapping defined for behavior '{behavior}'.")

data = validate_mappings(data, behavior, mappings)

# Calculate the percentage distribution for the mapped data
distribution = data[f'{behavior}_mapped'].value_counts(normalize=True)

# Create a DataFrame for the visualization
distribution_table = pd.DataFrame({
    'Category': distribution.index,
    'Percentage': (distribution.values * 100).round(2)
})

# Define a custom sort order
# Extract numerical parts for sorting, handle 'or more' and ranges
def sort_key(category):
    if 'or more' in category:
        return float('inf')
    elif '-' in category:
        start = category.split('-')[0]
        try:
            return float(start)
        except ValueError:
            return float('inf')
    else:
        try:
            return float(category)
        except ValueError:
            return float('inf')

distribution_table['SortOrder'] = distribution_table['Category'].apply(sort_key)

# Sort the DataFrame based on the custom sort order
distribution_table.sort_values('SortOrder', inplace=True)
distribution_table.drop('SortOrder', axis=1, inplace=True)

# Reset index after sorting
distribution_table.reset_index(drop=True, inplace=True)

# Optional: Move 'Unknown' to the end if it exists
unknown_mask = distribution_table['Category'] == 'Unknown'
if unknown_mask.any():
    unknown_row = distribution_table[unknown_mask]
    distribution_table = distribution_table[~unknown_mask]
    distribution_table = pd.concat([distribution_table, unknown_row], ignore_index=True)

# Create a pie chart with the ordered categories
fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the figure size here
wedges, texts = ax.pie(
    distribution_table['Percentage'],
    startangle=90,
    colors=sns.color_palette("Blues", n_colors=len(distribution_table)),
    textprops={'fontsize': 8}
)

# Create the legend with percentage values
labels = [
    f'{label} : {pct:.1f}%'
    for label, pct in zip(distribution_table['Category'], distribution_table['Percentage'])
]
ax.legend(
    wedges, labels,
    title="Category and Percentage",
    loc="center left",
    bbox_to_anchor=(1, 0, 0.5, 1)
)

ax.set_ylabel('')  # Remove the y-label
ax.set_title(f'Pie Chart of {behavior}')

# Define the output PDF file path using the script directory
output_pdf_path = plots_path / f'Pie_Chart_{behavior}.pdf'

# Save the figure to PDF in the specified path, making sure to include the whole figure
try:
    fig.savefig(output_pdf_path, format='pdf', bbox_inches='tight')
    print(f"Pie chart saved successfully at {output_pdf_path}")
except Exception as e:
    sys.exit(f"An error occurred while saving the pie chart: {e}")

plt.close(fig)  # Close the figure to free memory