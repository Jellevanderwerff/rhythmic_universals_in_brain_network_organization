import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from statsmodels.stats.multitest import multipletests

# Get the directory of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Set the data directory relative to the script location
DATA_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'data', 'network_assignment')

def permutation_within_network_count(adjacency_matrix, node_to_rsn_mapping, unique_networks):
    # Create a shuffled mapping of nodes to networks
    nodes = list(node_to_rsn_mapping.keys())
    shuffled_nodes = np.random.permutation(nodes)
    shuffled_mapping = {node: node_to_rsn_mapping[shuffled_node] for node, shuffled_node in zip(nodes, shuffled_nodes)}
    
    num_networks = len(unique_networks)
    count_shuffled = np.zeros(num_networks)
    
    # Find within-network connections
    for idx, network in enumerate(unique_networks):
        nodes_in_network = [node for node, net in shuffled_mapping.items() if net == network]
        count = 0
        for i in range(len(nodes_in_network)):
            for j in range(i+1, len(nodes_in_network)):
                node1 = nodes_in_network[i]
                node2 = nodes_in_network[j]
                if adjacency_matrix[node1, node2] != 0:
                    count += 1
        count_shuffled[idx] = count
    return count_shuffled

def analyze_within_network_connectivity():
    # List of behaviors/adjacency matrices to process
    behav_list = ['G_resp', 'entropy_diff_norm_q_avg', 'binary_or_ternary_introduced']

    # Define colors for each behavior
    color_list = ['blue', 'green', 'red']  # You can choose any colors you prefer

    # Initialize a dictionary to store densities for each behavior
    all_densities = {}

    for idx, behav in enumerate(behav_list):
        print(f"Processing behavior: {behav}")
        
        # Load the adjacency matrix
        if behav == 'G_resp':
            data = loadmat(os.path.join(DATA_DIR, 'AvAdjacencyMat.rsFC.negative.Gresponse.mat'))
        elif behav == 'entropy_diff_norm_q_avg':
            data = loadmat(os.path.join(DATA_DIR, 'AvAdjacencyMat.rsFC.negative.EntropyDiff.mat'))
        elif behav == 'binary_or_ternary_introduced':
            data = loadmat(os.path.join(DATA_DIR, 'AvAdjacencyMat.rsFC.negative.binary_or_ternary_introduced.mat'))
        
        # Extract the adjacency matrix
        adjacency_matrix = data[list(data.keys())[-1]]
        
        # Ensure the adjacency matrix is square and symmetric
        assert adjacency_matrix.shape[0] == adjacency_matrix.shape[1], "Adjacency matrix must be square."
        adjacency_matrix = np.triu(adjacency_matrix, k=1)  # Use upper triangle to avoid duplicate counting
        adjacency_matrix += adjacency_matrix.T  # Make the matrix symmetric
        
        # Load the network assignment CSV file
        dkas_to_networks = pd.read_csv(os.path.join(DATA_DIR, 'altenrnative_mapping_ShirerKabbara.matlab.csv'))
        
        # Create a mapping of node index to RSN (adjusting for zero-based indexing)
        node_to_rsn_mapping = {}
        unique_networks = dkas_to_networks.columns
        for network in unique_networks:
            nodes = dkas_to_networks[network].dropna().values.astype(int) - 1  # Subtract 1 for zero-based indexing
            for node in nodes:
                node_to_rsn_mapping[node] = network
        
        num_networks = len(unique_networks)
        within_network_connections_count = np.zeros(num_networks)
        total_possible_connections = np.zeros(num_networks)
        
        # Compute the total number of connections within networks
        for idx_net, network in enumerate(unique_networks):
            nodes_in_network = [node for node, net in node_to_rsn_mapping.items() if net == network]
            n_nodes = len(nodes_in_network)
            # Calculate total possible connections for this network
            total_possible = n_nodes * (n_nodes - 1) / 2 if n_nodes > 1 else 0
            total_possible_connections[idx_net] = total_possible
            
            count = 0
            for i in range(n_nodes):
                for j in range(i+1, n_nodes):
                    node1 = nodes_in_network[i]
                    node2 = nodes_in_network[j]
                    if adjacency_matrix[node1, node2] != 0:
                        count += 1
            within_network_connections_count[idx_net] = count
        
        print("Within-network connections count:", within_network_connections_count)
        print("Total possible connections:", total_possible_connections)
        
        # Permutations
        N_PERMUTATIONS = 1000
        permutation_counts = np.zeros((num_networks, N_PERMUTATIONS))
        
        for p in range(N_PERMUTATIONS):
            perm_count = permutation_within_network_count(adjacency_matrix, node_to_rsn_mapping, unique_networks)
            permutation_counts[:, p] = perm_count
        
        # Calculate p-values
        p_values = np.zeros(num_networks)
        for idx_net in range(num_networks):
            observed = within_network_connections_count[idx_net]
            perm_values = permutation_counts[idx_net, :]
            if observed == 0:
                p_values[idx_net] = np.nan
            else:
                # Calculate the p-value with continuity correction
                p_value = (np.sum(perm_values >= observed) + 1) / (N_PERMUTATIONS + 1)
                p_values[idx_net] = p_value
        
        print("P-values before correction:", p_values)
        
        # FDR correction
        non_nan_mask = ~np.isnan(p_values)
        p_values_to_correct = p_values[non_nan_mask]
        
        if len(p_values_to_correct) > 0:
            rejected, p_values_corrected, _, _ = multipletests(p_values_to_correct, method='fdr_bh')
        else:
            p_values_corrected = np.array([])
            rejected = np.array([])
        
        # Calculate densities
        densities = np.zeros(num_networks)
        for idx_net in range(num_networks):
            if total_possible_connections[idx_net] > 0:
                densities[idx_net] = within_network_connections_count[idx_net] / total_possible_connections[idx_net]
            else:
                densities[idx_net] = 0.0  # Handle division by zero
        
        print("Densities:", densities)
        
        # Scale densities for better visualization
        densities_scaled = densities * 100  # Convert to percentages
        
        # Store densities for this behavior
        all_densities[behav] = densities_scaled
        
        # Store other necessary data for plotting
        all_densities[f"{behav}_p_values"] = p_values
        all_densities[f"{behav}_p_values_corrected"] = p_values_corrected
        all_densities[f"{behav}_unique_networks"] = unique_networks.tolist()
        all_densities[f"{behav}_non_nan_mask"] = non_nan_mask
        all_densities[f"{behav}_rejected"] = rejected if len(rejected) > 0 else np.array([])

    # Determine fixed axis limits based on scaled densities
    fixed_min_density = 1.0   # Equivalent to 0.01 before scaling
    fixed_max_density = 5.0   # Equivalent to 0.05 before scaling

    # Optionally, adjust axis limits if densities are outside this range
    all_densities_list = []
    for behav in behav_list:
        densities_scaled = all_densities[behav]
        all_densities_list.extend(densities_scaled)
    
    min_density_scaled = min(all_densities_list)
    max_density_scaled = max(all_densities_list)
    
    # Adjust axis limits if necessary
    if min_density_scaled < fixed_min_density:
        fixed_min_density = max(0.0, min_density_scaled - 0.5)  # Add some padding
    if max_density_scaled > fixed_max_density:
        fixed_max_density = max_density_scaled + 0.5  # Add some padding
    
    print(f"\nAdjusted axis limits for spider charts: {fixed_min_density}% to {fixed_max_density}%")
    
    # Plotting for each behavior
    for idx, behav in enumerate(behav_list):
        print(f"\nGenerating plot for behavior: {behav}")
        densities_scaled = all_densities[behav]
        p_values = all_densities[f"{behav}_p_values"]
        p_values_corrected = all_densities[f"{behav}_p_values_corrected"]
        unique_networks = all_densities[f"{behav}_unique_networks"]
        non_nan_mask = all_densities[f"{behav}_non_nan_mask"]
        rejected = all_densities[f"{behav}_rejected"]
        
        # Choose the color for this behavior
        color = color_list[idx]
        
        # Prepare values and angles
        labels = unique_networks
        values = densities_scaled.tolist()
        num_vars = len(labels)
        
        # Ensure the plot closes by repeating the first value
        values += values[:1]
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]
        
        # Initialize the spider plot
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        
        # Draw the outline of the spider plot
        ax.plot(angles, values, color=color, linewidth=2, linestyle='solid')
        ax.fill(angles, values, color=color, alpha=0.25)
        
        # Set the labels for each axis
        ax.set_xticks(angles[:-1])
        # ax.set_xticklabels(labels, fontsize=10)  # Commented out to remove network names
        
        # Set scaled radial limits
        ax.set_ylim([fixed_min_density, fixed_max_density])
        
        # Add radial gridlines and labels
        ax.set_rlabel_position(0)
        rad_labels = np.linspace(fixed_min_density, fixed_max_density, num=5)
        ax.set_yticks(rad_labels)
        # ax.set_yticklabels(['{:.1f}%'.format(x) for x in rad_labels], fontsize=8)  # Commented out to remove percentage labels
        
        # Set the title
        plt.title(f'Density of Within-Network Connectivity\nBehavior: {behav}', fontsize=16, fontweight='bold')
        
        # Save the figure
        script_name = os.path.splitext(os.path.basename(__file__))[0]
        fig_filename = f"{behav}_{script_name}_spider_chart.pdf"
        fig_path = os.path.join(SCRIPT_DIR, '..', '..', 'results', 'network_assignment', fig_filename)
        plt.savefig(fig_path, format='pdf', bbox_inches='tight')
        plt.close(fig)
        
        # Print p-values and adjusted p-values
        print("\nEmpirical p-values for each network's within-network connections:")
        idx_corrected = 0
        for idx_net, network in enumerate(unique_networks):
            p_val = p_values[idx_net]
            if not np.isnan(p_val):
                if idx_corrected < len(p_values_corrected):
                    adj_p_val = p_values_corrected[idx_corrected]
                    print(f"{network}: p = {p_val:.4f}, adjusted p = {adj_p_val:.4f}")
                    idx_corrected += 1
                else:
                    print(f"{network}: p = {p_val:.4f}")
            else:
                print(f"{network}: p = NaN")
    
    print(f"\nFinal axis limits for spider charts: {fixed_min_density}% to {fixed_max_density}%")

if __name__ == "__main__":
    analyze_within_network_connectivity()