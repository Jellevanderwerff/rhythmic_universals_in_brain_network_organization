import pandas as pd
from statsmodels.multivariate.manova import MANOVA
from statsmodels.formula.api import ols
import statsmodels.api as sm
from pathlib import Path

# Get the project root directory
project_root = Path(__file__).resolve().parents[2]
data_path = project_root / 'data' / 'brain' / 'CPM'

# Load the data
data = pd.read_csv(data_path / "manova_data.csv")

# List of behavior grouping variables
behaviors = [
    'g_response_group',
    'entropydiff_group',
    'isochrony_introduced_group',
    'binary_ternary_group'
]

# Display available behaviors
print("Select a behavioral grouping variable for MANOVA and ANOVA analysis:")
for idx, behavior in enumerate(behaviors, start=1):
    print(f"{idx}. {behavior}")

# Prompt user to select a behavior
selection = int(input("Enter the number corresponding to the behavior variable: "))
selected_behavior = behaviors[selection - 1]
print(f"\nYou have selected: {selected_behavior}\n")

# Ensure the grouping variable is categorical
data[selected_behavior] = data[selected_behavior].astype('category')

# Define dependent variables
dependent_vars = ['MET', 'WAIS_IV', 'Gold_MSI']

# Drop rows with missing data in the selected behavior or dependent variables
data_clean = data.dropna(subset=[selected_behavior] + dependent_vars)

# Ensure dependent variables are numeric using .loc to avoid SettingWithCopyWarning
for var in dependent_vars:
    data_clean.loc[:, var] = pd.to_numeric(data_clean[var], errors='coerce')

# Drop any rows where dependent variables could not be converted to numeric
data_clean = data_clean.dropna(subset=dependent_vars)

# Construct the formula for MANOVA
dependent_str = ' + '.join(dependent_vars)
formula = f'{dependent_str} ~ {selected_behavior}'

# Perform MANOVA
maov = MANOVA.from_formula(formula, data=data_clean)
result = maov.mv_test()

# Extract the 'stat' DataFrame
stat_df = result.results[selected_behavior]['stat']

# Print the 'stat' DataFrame to inspect it
print("stat_df:")
print(stat_df)

# Print available indices (test names) to inspect them
print("Available indices in 'stat':", stat_df.index)

# Find the index for Wilks' Lambda
wilks_key = [key for key in stat_df.index if 'Wilks' in key][0]

# Access the Wilks' Lambda statistics
wilks = stat_df.loc[wilks_key]

wilks_lambda = wilks['Value']
f_value = wilks['F Value']
df_num = wilks['Num DF']
df_den = wilks['Den DF']
p_value = wilks['Pr > F']

print("\nMANOVA Results (Wilks' Lambda):")
print(f"Wilks' Lambda: {wilks_lambda}")
print(f"F-value: {f_value}")
print(f"Degrees of freedom: ({df_num}, {df_den})")
print(f"p-value: {p_value}")

# Compute Eta Squared (η²)
eta_squared = 1 - (wilks_lambda) ** (1 / len(dependent_vars))
print(f"Eta Squared (η²): {eta_squared:.4f}")

# Compute group statistics
print("\nGroup Statistics:")
for group in data_clean[selected_behavior].cat.categories:
    print(f"\nGroup: {group}")
    group_data = data_clean[data_clean[selected_behavior] == group]
    for var in dependent_vars:
        mean = group_data[var].mean()
        std = group_data[var].std()
        min_val = group_data[var].min()
        max_val = group_data[var].max()
        print(f"{var}: Mean = {mean:.2f}, SD = {std:.2f}, Range = ({min_val}, {max_val})")

# ----------------------------
# Perform Univariate ANOVAs
# ----------------------------
print("\nUnivariate ANOVA Results:")

for var in dependent_vars:
    print(f"\nANOVA for {var}:")
    formula_anova = f'{var} ~ C({selected_behavior})'
    model = ols(formula_anova, data=data_clean).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)