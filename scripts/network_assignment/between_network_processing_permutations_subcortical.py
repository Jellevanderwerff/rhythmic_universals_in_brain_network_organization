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
    behav = 'entropy_diff_norm_q_avg'  # Options: 'entropy_diff_norm_q_avg', 'binary_or_ternary_introduced'
    
    # Load the adjacency matrix
        
    if behav == 'entropy_diff_norm_q_avg':
        data = loadmat(os.path.join(DATA_DIR, 'consensus_neg_composite_entropy_diff_norm_q_avg_k5.mat')) #swith between *k5 and k10 (k-fold used)
        
    elif behav == 'binary_or_ternary_introduced':
        data = loadmat(os.path.join(DATA_DIR, 'consensus_neg_composite_binary_or_ternary_introduced.mat'))
        


    # Initialize lists and counters for plotting
    num_plots = 0
    
    adjacency_matrix = data[list(data.keys())[-1]]
    
    # Load the DKA-to-network mapping
    dkas_to_networks = pd.read_csv(os.path.join(DATA_DIR, 'mapping_ShirerKabbara.matlab_subcortical.csv'))
    
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
                p_value = (np.sum(perm_values >= observed)) / N_PERMUTATIONS
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
        
    # Print comprehensive summary for network pairs with observed connections (count > 0)
    print("\n" + "="*80)
    print("BETWEEN-NETWORK CONNECTIVITY ANALYSIS SUMMARY")
    print("="*80)
    print("NOTE: Statistics derived from permutation null distribution")
    print("      Only network pairs with observed connections (count > 0) are reported")
    print("="*80)
    
    # Filter for network pairs with observed connections > 0
    stats_with_connections = [entry for entry in network_pair_stats if entry['Observed_Count'] > 0]
    
    # Further filter and sort by FDR-adjusted p-value for better presentation
    stats_with_fdr = [entry for entry in stats_with_connections if 'FDR_Adjusted_P_Value' in entry and not np.isnan(entry['FDR_Adjusted_P_Value'])]
    stats_without_fdr = [entry for entry in stats_with_connections if 'FDR_Adjusted_P_Value' not in entry or np.isnan(entry.get('FDR_Adjusted_P_Value', np.nan))]
    
    # Sort stats with FDR by adjusted p-value (ascending - most significant first)
    stats_with_fdr_sorted = sorted(stats_with_fdr, key=lambda x: x['FDR_Adjusted_P_Value'])
    
    # Combine sorted lists (significant results first, then non-significant)
    all_stats_sorted = stats_with_fdr_sorted + stats_without_fdr
    
    for idx, entry in enumerate(all_stats_sorted, 1):
        network_pair = entry['Network_Pair']
        observed_count = entry['Observed_Count']
        z_score = entry['Z_Score']
        ci_lower = entry['CI_Lower']
        ci_upper = entry['CI_Upper']
        raw_p_value = entry['Raw_P_Value']
        fdr_p_value = entry.get('FDR_Adjusted_P_Value', np.nan)
        is_significant = entry.get('Significant', False)
        
        print(f"\n{idx:2d}. {network_pair}")
        print("-" * (len(network_pair) + 4))
        
        # Connection statistics
        print(f"   Observed Connections: {observed_count:>8.0f}")
        if not np.isnan(z_score):
            print(f"   Z-score:             {z_score:>8.2f}")
        else:
            print(f"   Z-score:             {'N/A':>8s}")
        
        # Confidence intervals
        print(f"   95% CI:              [{ci_lower:>6.1f}, {ci_upper:>6.1f}]")
        
        # Statistical significance
        if not np.isnan(raw_p_value):
            print(f"   Raw p-value:         {raw_p_value:>8.3f}")
            if not np.isnan(fdr_p_value):
                print(f"   FDR-corrected p:     {fdr_p_value:>8.3f}")
                significance_marker = " ***" if is_significant else ""
                print(f"   Significance:        {'Yes' if is_significant else 'No':>8s}{significance_marker}")
            else:
                print(f"   FDR-corrected p:     {'N/A':>8s}")
                print(f"   Significance:        {'N/A':>8s}")
        else:
            print(f"   Raw p-value:         {'N/A':>8s} (no connections observed)")
            print(f"   FDR-corrected p:     {'N/A':>8s}")
            print(f"   Significance:        {'N/A':>8s}")
        
        # Interpretation based on permutation-derived statistics
        if not np.isnan(z_score):
            if observed_count < ci_lower:
                interpretation = "Significantly fewer connections than expected under null"
            elif observed_count > ci_upper:
                interpretation = "Significantly more connections than expected under null"
            else:
                interpretation = "Connections within permutation-derived null range"
            print(f"   Interpretation:      {interpretation}")
        else:
            print(f"   Interpretation:      {'Statistical computation error'}")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    # Count significant pairs
    significant_pairs = sum(1 for entry in network_pair_stats if entry.get('Significant', False))
    total_testable_pairs = len([entry for entry in network_pair_stats if not np.isnan(entry['Raw_P_Value'])])
    total_pairs = len(network_pair_stats)
    
    print(f"Total network pairs analyzed:           {total_pairs:>3d}")
    print(f"Pairs with testable connections:        {total_testable_pairs:>3d}")
    print(f"Pairs with significant connectivity:    {significant_pairs:>3d}")
    
    if total_testable_pairs > 0:
        print(f"Proportion significant (of testable):   {significant_pairs/total_testable_pairs:>7.2%}")
    
    # Range of observed connections
    valid_counts = [entry['Observed_Count'] for entry in network_pair_stats if entry['Observed_Count'] > 0]
    if valid_counts:
        print(f"Range of connection counts:             {min(valid_counts):>3.0f} - {max(valid_counts):>3.0f}")
        print(f"Mean connection count (non-zero):       {np.mean(valid_counts):>7.1f}")
    
    print("\n*** indicates FDR-corrected significance at α = 0.05")
    print("="*80)

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

