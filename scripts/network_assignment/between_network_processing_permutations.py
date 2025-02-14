import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from scipy.io import loadmat
from statsmodels.stats.multitest import multipletests
from scipy.stats import mannwhitneyu  # Import the Mann-Whitney U test function
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Get the directory of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Set the data directory relative to the script location
DATA_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'data', 'network_assignment')

def print_permutation_info(shuffled_mapping, count_shuffled, unique_networks):
    print("\nPermutation Information:")
    
    # Count nodes in each network after shuffling
    network_counts = {network: 0 for network in unique_networks}
    for node, network in shuffled_mapping.items():
        network_counts[network] += 1
    
    # Print network sizes
    print("Network sizes after shuffling:")
    for network, count in network_counts.items():
        print(f"{network}: {count} nodes")
    
    # Print new connection matrix
    print("\nNew connection matrix:")
    print(pd.DataFrame(count_shuffled, index=unique_networks, columns=unique_networks))
    print("\n" + "-"*50)

def permutation_connections_count(adjacency_matrix, node_to_rsn_mapping, dkas_to_networks, unique_networks):
    # Create a shuffled mapping of nodes to networks
    nodes = list(node_to_rsn_mapping.keys())
    shuffled_nodes = np.random.permutation(nodes)
    shuffled_mapping = {old: node_to_rsn_mapping[new] for old, new in zip(nodes, shuffled_nodes)}
    
    num_networks = len(unique_networks)
    count_shuffled = np.zeros((num_networks, num_networks))
    
    # Find between- and within- (diagonal) network connections
    for i in range(num_networks):
        for j in range(num_networks):
            count = 0
            for node1 in range(adjacency_matrix.shape[0]):
                for node2 in range(node1+1, adjacency_matrix.shape[1]):
                    if adjacency_matrix[node1, node2] != 0:
                        if shuffled_mapping[node1+1] == unique_networks[i] and shuffled_mapping[node2+1] == unique_networks[j]:
                            count += 1
                        elif shuffled_mapping[node1+1] == unique_networks[j] and shuffled_mapping[node2+1] == unique_networks[i]:
                            count += 1
            count_shuffled[i, j] = count_shuffled[j, i] = count
    return count_shuffled

def analyze_network_connectivity():
    behav = 'G_resp'  # Options: 'G_resp', 'entropy_diff_norm_q_avg', 'binary_or_ternary_introduced'
    
    # Load the adjacency matrix
    if behav == 'G_resp':
        data = loadmat(os.path.join(DATA_DIR, 'AvAdjacencyMat.rsFC.negative.Gresponse.mat'))
        maskedAverageMatrixLow = loadmat(os.path.join(DATA_DIR, 'maskedAverageMatrixLow_G-resp.mat'))['maskedAverageMatrixLow']
        maskedAverageMatrixHigh = loadmat(os.path.join(DATA_DIR, 'maskedAverageMatrixHigh_G-resp.mat'))['maskedAverageMatrixHigh']
        # Load new matrices
        newMatrixLow = loadmat(os.path.join(DATA_DIR, 'newMatrixLow_median_split_G-resp.mat'))['newMatrixLow']
        newMatrixHigh = loadmat(os.path.join(DATA_DIR, 'newMatrixHigh_median_split_G-resp.mat'))['newMatrixHigh']
    elif behav == 'entropy_diff_norm_q_avg':
        data = loadmat(os.path.join(DATA_DIR, 'AvAdjacencyMat.rsFC.negative.EntropyDiff.mat'))
        maskedAverageMatrixHigh = loadmat(os.path.join(DATA_DIR, 'maskedAverageMatrixHigh_entropy-diff-norm-q-avg.mat'))['maskedAverageMatrixHigh']
        maskedAverageMatrixLow = loadmat(os.path.join(DATA_DIR, 'maskedAverageMatrixLow_entropy-diff-norm-q-avg.mat'))['maskedAverageMatrixLow']
        # Load new matrices
        newMatrixLow = loadmat(os.path.join(DATA_DIR, 'newMatrixLow_median_split_entropy-diff-norm-q-avg.mat'))['newMatrixLow']
        newMatrixHigh = loadmat(os.path.join(DATA_DIR, 'newMatrixHigh_median_split_entropy-diff-norm-q-avg.mat'))['newMatrixHigh']
    elif behav == 'binary_or_ternary_introduced':
        data = loadmat(os.path.join(DATA_DIR, 'AvAdjacencyMat.rsFC.negative.binary_or_ternary_introduced.mat'))
        maskedAverageMatrixHigh = loadmat(os.path.join(DATA_DIR, 'maskedAverageMatrixHigh_binary-or-ternary-introduced.mat'))['maskedAverageMatrixHigh']
        maskedAverageMatrixLow = loadmat(os.path.join(DATA_DIR, 'maskedAverageMatrixLow_binary-or-ternary-introduced.mat'))['maskedAverageMatrixLow']
        # Load new matrices
        newMatrixLow = loadmat(os.path.join(DATA_DIR, 'newMatrixLow_median_split_binary-or-ternary-introduced.mat'))['newMatrixLow']
        newMatrixHigh = loadmat(os.path.join(DATA_DIR, 'newMatrixHigh_median_split_binary-or-ternary-introduced.mat'))['newMatrixHigh']


    # Initialize lists and counters for plotting
    network_pairs = []
    mean_connectivity_high_list = []
    mean_connectivity_low_list = []
    num_plots = 0
    
    adjacency_matrix = data[list(data.keys())[-1]]
    
    # Load the DKA-to-network mapping
    dkas_to_networks = pd.read_csv(os.path.join(DATA_DIR, 'altenrnative_mapping_ShirerKabbara.matlab.csv'))
    
    # Create a mapping of node index to RSN
    node_to_rsn_mapping = {}
    unique_networks = dkas_to_networks.columns
    for network in unique_networks:
        nodes = dkas_to_networks[network].dropna().values
        for node in nodes:
            node_to_rsn_mapping[int(node)] = network
    
    # Find the indexes of non-zero values in the lower triangle (excluding the diagonal)
    row, col = np.where(np.tril(adjacency_matrix, -1))
    node_to_node_connections = np.column_stack((row, col))
    
    # Compute the total number of connections between networks
    num_networks = len(unique_networks)
    between_network_connections_count = np.zeros((num_networks, num_networks))
    between_network_connection_strength_high = np.zeros((num_networks, num_networks))
    between_network_connection_strength_low = np.zeros((num_networks, num_networks))
    
    # Initialize the p-value matrix with NaN values
    p_value_matrix = np.full((num_networks, num_networks), np.nan)


    for i in range(num_networks):
        for j in range(i, num_networks):  # Start from i to avoid double counting
            nodes_network1 = dkas_to_networks[unique_networks[i]].dropna().values.astype(int)
            nodes_network2 = dkas_to_networks[unique_networks[j]].dropna().values.astype(int)

            print(f"\nNetwork pair: {unique_networks[i]} - {unique_networks[j]}")
            print(f"Nodes in {unique_networks[i]}: {nodes_network1}")
            print(f"Nodes in {unique_networks[j]}: {nodes_network2}")
            
            node_pairs = set()
            connected_pairs = []
            for k in nodes_network1:
                for l in nodes_network2:
                    if i == j and k >= l:  # For within-network, avoid double counting
                        continue
                    if adjacency_matrix[k-1, l-1] != 0:
                        node_pairs.add((min(k-1, l-1), max(k-1, l-1)))
                        connected_pairs.append((k, l))  # Store the original node numbers

            
            count = len(node_pairs)
            between_network_connections_count[i, j] = between_network_connections_count[j, i] = count
            
            print(f"Number of connections: {count}")
            print("Connected node pairs:")
            for pair in connected_pairs:
                print(f"{pair[0]} {pair[1]}")

            # Calculate connection strengths
            conn_values_high = [maskedAverageMatrixHigh[k, l] for k, l in node_pairs]
            conn_values_low = [maskedAverageMatrixLow[k, l] for k, l in node_pairs]
            
            between_network_connection_strength_low[i, j] = between_network_connection_strength_low[j, i] = np.mean(conn_values_low) if conn_values_low else 0
            between_network_connection_strength_high[i, j] = between_network_connection_strength_high[j, i] = np.mean(conn_values_high) if conn_values_high else 0

            # For newMatrixHigh
            n_high = newMatrixHigh.shape[2]  # Number of participants in High
            connectivity_values_high = np.zeros((len(node_pairs), n_high))
            for idx, (k, l) in enumerate(node_pairs):
                connectivity_values_high[idx, :] = newMatrixHigh[k, l, :]
            # Compute mean over node pairs for each participant
            mean_connectivity_high = np.mean(connectivity_values_high, axis=0)  # size n_high

            # For newMatrixLow
            n_low = newMatrixLow.shape[2]  # Number of participants in Low
            connectivity_values_low = np.zeros((len(node_pairs), n_low))
            for idx, (k, l) in enumerate(node_pairs):
                connectivity_values_low[idx, :] = newMatrixLow[k, l, :]
            # Compute mean over node pairs for each participant
            mean_connectivity_low = np.mean(connectivity_values_low, axis=0)  # size n_low

            network_pair_name = f"{unique_networks[i]} - {unique_networks[j]}"
            network_pairs.append(network_pair_name)
            # Store data for this network pair
            mean_connectivity_high_list.append(mean_connectivity_high)
            mean_connectivity_low_list.append(mean_connectivity_low)
            # Increment plot count
            num_plots += 1

            # Print size and vectors
            print(f"\nNetwork pair: {unique_networks[i]} - {unique_networks[j]}")
            print(f"Number of participants in High: {n_high}")
            print(f"Mean connectivity values in High (size {mean_connectivity_high.shape}):")
            print(mean_connectivity_high)
            print(f"Number of participants in Low: {n_low}")
            print(f"Mean connectivity values in Low (size {mean_connectivity_low.shape}):")
            print(mean_connectivity_low)
        
            # Perform the Mann-Whitney U test
            stat, p_value = mannwhitneyu(mean_connectivity_high, mean_connectivity_low, alternative='two-sided')

            # Store the p-value in the matrix (ensuring symmetry)
            p_value_matrix[i, j] = p_value_matrix[j, i] = p_value

     # Mask the p-value matrix so that only the lower triangular part has values
    upper_triangle_indices = np.triu_indices(num_networks, k=0)
    p_value_matrix[upper_triangle_indices] = np.nan       

    # Extract p-values from the lower triangle (excluding the diagonal)
    i_lower = np.tril_indices(num_networks, k=-1)
    p_values_flat = p_value_matrix[i_lower]
    p_values_non_nan = p_values_flat[~np.isnan(p_values_flat)]

    # Perform FDR correction
    rejected, p_values_corrected, _, _ = multipletests(p_values_non_nan, method='fdr_bh')

    # Create a new matrix for corrected p-values, initialized with NaNs
    p_value_matrix_corrected = np.full_like(p_value_matrix, np.nan)

    # Place the corrected p-values back into the corresponding positions in the matrix
    i_indices = i_lower[0][~np.isnan(p_values_flat)]
    j_indices = i_lower[1][~np.isnan(p_values_flat)]
    p_value_matrix_corrected[i_indices, j_indices] = p_values_corrected

    print("\nFDR-corrected p-value matrix (only lower triangle):")
    print(pd.DataFrame(p_value_matrix_corrected, index=unique_networks, columns=unique_networks))
    

    valid_network_pairs = []
    valid_mean_connectivity_high = []
    valid_mean_connectivity_low = []

    # Iterate through all network pairs to filter out those with NaN values
    for idx in range(num_plots):
        data_high = mean_connectivity_high_list[idx]
        data_low = mean_connectivity_low_list[idx]
        
        # Check for NaN values
        if np.isnan(data_high).any() or np.isnan(data_low).any():
            print(f"Skipping plot for network pair '{network_pairs[idx]}' due to NaN values.")
            continue  # Skip to the next network pair
        
        # If data is valid, append to the new lists
        valid_network_pairs.append(network_pairs[idx])
        valid_mean_connectivity_high.append(data_high)
        valid_mean_connectivity_low.append(data_low)

    # Update the count of plots to be generated
    filtered_num_plots = len(valid_network_pairs)

    print(f"Total network pairs: {num_plots}")
    print(f"Valid network pairs to plot: {filtered_num_plots}")
    print(f"Skipped plots due to NaN values: {num_plots - filtered_num_plots}")

    # Define the number of columns you want in your subplot grid
    ncols = 3  # Adjust based on your preference and the number of plots

    # Calculate the number of rows needed
    nrows = (filtered_num_plots + ncols - 1) // ncols  # Ceiling division

    # Create subplots with the calculated grid size
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*6, nrows*5))

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Iterate through the valid network pairs and plot
    for idx, ax in enumerate(axes):
        if idx < filtered_num_plots:
            # Data for this network pair
            data_high = valid_mean_connectivity_high[idx]
            data_low = valid_mean_connectivity_low[idx]
            network_name = valid_network_pairs[idx]
            
            # Prepare DataFrame for seaborn
            df = pd.DataFrame({
                'Connectivity': np.concatenate([data_low, data_high]),
                'Group': ['Low']*len(data_low) + ['High']*len(data_high)
            })
            
            # Create violin plot
            sns.violinplot(
                x='Group', 
                y='Connectivity', 
                hue='Group',
                data=df, 
                ax=ax, 
                inner=None, 
                palette='Set2', 
                cut=0,
                legend=False
            )
            
            # Add individual data points
            sns.stripplot(
                x='Group', 
                y='Connectivity', 
                data=df, 
                ax=ax, 
                color='black', 
                alpha=0.5
            )

            # Calculate medians
            median_low = np.median(data_low)
            median_high = np.median(data_high)
            
            # Plot median points with red diamond ('D') symbols
            ax.scatter(0, median_low, color='red', marker='D', s=100, label='Median Low')
            ax.scatter(1, median_high, color='red', marker='D', s=100, label='Median High')

            # Set title and labels
            ax.set_title(network_name, fontsize=12)
            ax.set_xlabel('Group')
            ax.set_ylabel('Mean Connectivity')
        else:
            # Hide any unused subplots
            ax.axis('off')

    # Adjust layout
    plt.tight_layout()

    # Ensure the output directory exists
    output_dir = os.path.join(SCRIPT_DIR, '..', '..', 'results', 'network_assignment')
    os.makedirs(output_dir, exist_ok=True)

    # Save the figure
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    fig3_filename = f"{behav}_{script_name}_fig3.pdf"
    fig3_path = os.path.join(output_dir, fig3_filename)
    plt.savefig(fig3_path, format='pdf', bbox_inches='tight')
    plt.close(fig)

    # Permutations
    N_PERMUTATIONS = 1000
    permutation_counts = np.zeros((num_networks, num_networks, N_PERMUTATIONS))
    
    for p in range(N_PERMUTATIONS):
        print(f'\n Performing iteration {p+1} out of {N_PERMUTATIONS}')
        perm_count = permutation_connections_count(adjacency_matrix, node_to_rsn_mapping, dkas_to_networks, unique_networks)
        permutation_counts[:, :, p] = perm_count

    # Initialize data structures for storing statistics
    network_pair_stats = []
    p_values_list = []
    p_values_indices = []  # indices into network_pair_stats where p_value is not NaN

    # Calculate p-values and other statistics
    p_value_matrix = np.zeros((num_networks, num_networks))

    for i in range(1, num_networks):
        for j in range(i):
            observed = between_network_connections_count[i, j]
            perm_values = permutation_counts[i, j, :]
            network_pair_name = f"{unique_networks[i]} - {unique_networks[j]}"
            
            if observed == 0:
                p_value = np.nan
            else:
                # Count how many permutation values are greater than or equal to the observed value
                p_value = (np.sum(perm_values > observed)) / N_PERMUTATIONS
                p_values_list.append(p_value)
                p_values_indices.append(len(network_pair_stats))  # index into network_pair_stats

            # Store p_value in p_value_matrix
            p_value_matrix[i, j] = p_value_matrix[j, i] = p_value

            # Compute mean, std, Z-score, and confidence intervals
            mean_perm = np.mean(perm_values)
            std_perm = np.std(perm_values, ddof=1)
            Z = (observed - mean_perm) / std_perm if std_perm != 0 else np.nan
            ci_lower = np.percentile(perm_values, 2.5)
            ci_upper = np.percentile(perm_values, 97.5)

            # Create the data dictionary
            data_dict = {
                'Network_Pair': network_pair_name,
                'Observed_Count': observed,
                'Raw_P_Value': p_value,
                'Z_Score': Z,
                'CI_Lower': ci_lower,
                'CI_Upper': ci_upper,
                'Index_i': i,
                'Index_j': j
            }

            # Append to network_pair_stats
            network_pair_stats.append(data_dict)

    # Perform FDR correction
    p_values_list = np.array(p_values_list)
    rejected, p_values_corrected, _, _ = multipletests(p_values_list, method='fdr_bh')

    # Assign adjusted p-values back to network_pair_stats
    for idx_in_p_values_list, idx_in_network_pair_stats in enumerate(p_values_indices):
        network_pair_stats[idx_in_network_pair_stats]['FDR_Adjusted_P_Value'] = p_values_corrected[idx_in_p_values_list]
        network_pair_stats[idx_in_network_pair_stats]['Significant'] = rejected[idx_in_p_values_list]

    # Build p_value_matrix and p_value_matrix_corrected from network_pair_stats
    p_value_matrix = np.full((num_networks, num_networks), np.nan)
    p_value_matrix_corrected = np.full((num_networks, num_networks), np.nan)

    for entry in network_pair_stats:
        i = entry['Index_i']
        j = entry['Index_j']
        p_value = entry['Raw_P_Value']
        adjusted_p_value = entry.get('FDR_Adjusted_P_Value', np.nan)

        p_value_matrix[i, j] = p_value_matrix[j, i] = p_value
        p_value_matrix_corrected[i, j] = p_value_matrix_corrected[j, i] = adjusted_p_value

    # Set the title based on behav
    if behav == 'G_resp':
        title = 'Grammatical Redundancy'
    elif behav == 'entropy_diff_norm_q_avg':
        title = 'Entropy Difference'
    elif behav == 'binary_or_ternary_introduced':
        title = 'Binary or Ternary Ratios Introduced'
    else:
        title = 'Between-Network Connectivity Analysis'

    # Create a custom colormap with gray for NaN values
    cmap = mcolors.ListedColormap(plt.colormaps['viridis'](np.linspace(0, 1, 256)))
    cmap.set_bad(color='lightgray')

    # Create figure 1
    fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
    fig1.suptitle(title, fontsize=16, fontweight='bold')

    # Function to set upper triangle to NaN
    def set_upper_triangle_nan(matrix):
        matrix_copy = matrix.copy()
        matrix_copy[np.triu_indices(matrix.shape[0], k=0)] = np.nan
        return matrix_copy

    # Replace 0 values with NaN in between_network_connections_count for Subplot 1
    between_network_connections_count[between_network_connections_count == 0] = np.nan

    # Create a custom colormap with lighter gray for NaN values in subplot 1
    cmap_viridis_lighter = mcolors.ListedColormap(plt.colormaps['viridis'].colors)
    cmap_viridis_lighter.set_bad(color='lightgray')  # Lighter gray for NaN in subplot 1

    # Create a custom colormap with darker gray for NaN values in subplots 2 and 3
    cmap_viridis_darker = mcolors.ListedColormap(plt.colormaps['viridis'].colors)
    cmap_viridis_darker.set_bad(color='gray')  # Darker gray for NaN in subplots 2 and 3

    # Define color limits for each subplot
    vmin1, vmax1 = 1, 10  # For connection counts in subplot 1
    vmin2, vmax2 = 0, 1   # For p-values and FDR-adjusted p-values in subplots 2 and 3

    # Subplot 1: Original heatmap with connection counts (lower triangle only, lighter gray for NaN)
    sns.heatmap(set_upper_triangle_nan(between_network_connections_count), ax=ax1, cmap=cmap_viridis_lighter, 
                xticklabels=unique_networks, yticklabels=unique_networks,
                annot=True, fmt='.2f', cbar=True, vmin=vmin1, vmax=vmax1)
    ax1.set_xlabel('Total Number of Connections')

    # Subplot 2: Heatmap with raw p-values (lower triangle only, darker gray for NaN)
    sns.heatmap(set_upper_triangle_nan(p_value_matrix), ax=ax2, cmap=cmap_viridis_darker, vmin=vmin2, vmax=vmax2, 
                xticklabels=unique_networks, yticklabels=unique_networks,
                annot=True, fmt='.3f', cbar=True)
    ax2.set_xlabel('Raw P-values')

    # Subplot 3: Heatmap with FDR-adjusted p-values (lower triangle only, darker gray for NaN)
    sns.heatmap(set_upper_triangle_nan(p_value_matrix_corrected), ax=ax3, cmap=cmap_viridis_darker, vmin=vmin2, vmax=vmax2, 
                xticklabels=unique_networks, yticklabels=unique_networks,
                annot=True, fmt='.3f', cbar=True)
    ax3.set_xlabel('FDR-adjusted P-values')

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    # Save figure 1
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    fig1_filename = f"{behav}_{script_name}_fig1.pdf"
    fig1_path = os.path.join(SCRIPT_DIR, '..', '..', 'results', 'network_assignment', fig1_filename)
    plt.savefig(fig1_path, format='pdf', bbox_inches='tight')
    plt.close(fig1)

    # Create figure 2
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    fig2.suptitle(title, fontsize=16, fontweight='bold')

    # Find the global min and max values across both matrices
    vmin = min(np.min(between_network_connection_strength_low), np.min(between_network_connection_strength_high))
    vmax = max(np.max(between_network_connection_strength_low), np.max(between_network_connection_strength_high))

    # Create a custom colormap with gray for NaN values
    cmap = mcolors.ListedColormap(plt.colormaps['viridis'].colors)
    cmap.set_bad(color='gray')  

    # Replace 0 values with NaN to avoid color coding them
    between_network_connection_strength_low[between_network_connection_strength_low == 0] = np.nan
    between_network_connection_strength_high[between_network_connection_strength_high == 0] = np.nan

    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(between_network_connection_strength_low), k=0)

    # Subplot 1: Low Behavior
    sns.heatmap(between_network_connection_strength_low, ax=ax1, cmap=cmap, 
                xticklabels=unique_networks, yticklabels=unique_networks,
                vmin=vmin, vmax=vmax,
                mask=mask,
                annot=True, fmt='.4f', cbar=True,
                square=True, cbar_kws={'shrink': .8},
                linewidths=0.5, linecolor='white')
    ax1.set_xlabel('Low Behavior')

    # Subplot 2: High Behavior
    sns.heatmap(between_network_connection_strength_high, ax=ax2, cmap=cmap, 
                xticklabels=unique_networks, yticklabels=unique_networks,
                vmin=vmin, vmax=vmax,
                mask=mask,
                annot=True, fmt='.4f', cbar=True,
                square=True, cbar_kws={'shrink': .8},
                linewidths=0.5, linecolor='white')
    ax2.set_xlabel('High Behavior')

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    # Save figure 2
    fig2_filename = f"{behav}_{script_name}_fig2.pdf"
    fig2_path = os.path.join(SCRIPT_DIR, '..', '..', 'results', 'network_assignment', fig2_filename)
    plt.savefig(fig2_path, format='pdf', bbox_inches='tight')
    plt.close(fig2)

    # Create a DataFrame from network_pair_stats and save to CSV
    df_permutation_stats = pd.DataFrame(network_pair_stats)
    # Drop the 'Index_i' and 'Index_j' columns if not needed
    df_permutation_stats = df_permutation_stats.drop(columns=['Index_i', 'Index_j'])

    # Save the DataFrame to CSV
    csv_filename = f"{behav}_{script_name}_permutation_stats.csv"
    csv_path = os.path.join(SCRIPT_DIR, '..', '..', 'results', 'network_assignment', csv_filename)
    df_permutation_stats.to_csv(csv_path, index=False)

# Run the analysis
analyze_network_connectivity()