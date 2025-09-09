clear, clc, close all
%% Input parameters %%
modality = {'functional'}; % 'functional' or 'structural_FBC' 
corr_method = {'Spearman'}; % 'Pearson', 'Spearman','partial'
thresh = 0.01; % threshold for feature selection: 0.01, 0.05, etc.
mainmetrics = 'isochrony_introduced'; %'G_resp','entropy_diff_norm_q_avg','isochrony_introduced','binary_or_ternary_introduced'
condition = '6005'; % '4004'(400 ms 4 events)'4005';'6004'; '6005'; 'composite'
figure_plot = {'yes'}; % create figure ('yes') or not ('no')
mask_apply = {'no'}; % Whether to use masking for ROI analyses

% K-fold cross-validation parameters
k_folds = 10; % Number of folds for cross-validation
random_seed = 42; % Set random seed for reproducibility
num_repetitions = 50; % Number of times to repeat the analysis
verbose = 0; % Controls amount of output (0=minimal, 1=detailed) - Set to 0 to suppress fold processing messages

% permutation parameters
permutation={'no'}; % 'yes' or 'no'
no_iterations = 1000;

%% Define the directory paths and load data

[connectomeDirectory, ~, ~, all_mats, all_behav, data] = setupDirectoriesAndLoadData(modality, condition, mainmetrics);

if strcmp(modality, 'structural_FBC')
    % Z-score normalization for each connectivity matrix
    for i = 1:size(all_mats, 3)
        current_mat = all_mats(:,:,i);
        % Only normalize non-zero values to preserve sparsity pattern
        mask = current_mat > 0;
        vals = current_mat(mask);
        vals_norm = (vals - mean(vals)) / std(vals);
        current_mat(mask) = vals_norm;
        all_mats(:,:,i) = current_mat;
    end
    fprintf('Structural connectivity matrices normalized\n');
end

%% mask connectome (optional)
if strcmp(mask_apply{1},'yes')
    mask_indices = [2,3,16,17,21,22,23,26,28,29,30,33,34,36,37,50,51,55,56,57,60,62,63,64,67,68];
    
    % Extract the sub-connectome
    [sub_connectome, valid_mask] = extract_sub_connectome(all_mats, mask_indices);
      
    all_mats = sub_connectome;
else
    mask_indices = [];
end

%% get basic info (number of subjects and nodes)
no_subj = size(all_mats,3);
no_nodes = size(all_mats,1);

%% Connectome-based predictive modeling (CPM) analysis with multiple repetitions of K-fold cross-validation
% Suppress the message about performing repetitions
% fprintf('\n Performing %d repetitions of K-fold cross-validation (k=%d) with seed %d\n', num_repetitions, k_folds, random_seed);

% Initialize arrays to store predictions from multiple repetitions
behav_pred_pos_all = zeros(no_subj, num_repetitions);
behav_pred_neg_all = zeros(no_subj, num_repetitions);
R_pos_all = zeros(num_repetitions, 1);
R_neg_all = zeros(num_repetitions, 1);
P_pos_all = zeros(num_repetitions, 1);
P_neg_all = zeros(num_repetitions, 1);

% Run the k-fold CV multiple times with different random assignments
for rep = 1:num_repetitions
    
    % fprintf('\nRepetition %d of %d\n', rep, num_repetitions);
    
    % Set a different random seed for each repetition (but deterministic)
    rng(random_seed + rep - 1);
    
    % Run the original k-fold cross-validation with suppressed output
    [behav_pred_pos, behav_pred_neg, R_pos, P_pos, R_neg, P_neg, pos_mask, neg_mask, train_sumpos, train_sumneg] = predict_behaviour_kfold_quiet(modality{1}, all_mats, all_behav, no_nodes, no_subj, thresh, corr_method{1}, k_folds, verbose);
    
    % Store predictions
    behav_pred_pos_all(:, rep) = behav_pred_pos;
    behav_pred_neg_all(:, rep) = behav_pred_neg;
    R_pos_all(rep) = R_pos;
    R_neg_all(rep) = R_neg;
    P_pos_all(rep) = P_pos;
    P_neg_all(rep) = P_neg;
    
    
    % fprintf('  Pos. network: R = %.4f (p = %.4f), Neg. network: R = %.4f (p = %.4f)\n', ...
    %    R_pos, P_pos, R_neg, P_neg);
    
    % Store the masks and network strengths from the last repetition
    if rep == num_repetitions
        pos_mask_final = pos_mask;
        neg_mask_final = neg_mask;
        train_sumpos_final = train_sumpos;
        train_sumneg_final = train_sumneg;
    end
end

% Average predictions across repetitions
behav_pred_pos = mean(behav_pred_pos_all, 2);
behav_pred_neg = mean(behav_pred_neg_all, 2);

% Calculate mean and std of individual repetition correlations
R_pos_mean = mean(R_pos_all);
R_neg_mean = mean(R_neg_all);
R_pos_std = std(R_pos_all);
R_neg_std = std(R_neg_all);
P_pos_mean = mean(P_pos_all);
P_neg_mean = mean(P_neg_all);

%% Calculate correlation coefficients using the averaged predictions

[R_pos_true, P_pos_true] = corrcoef(behav_pred_pos, all_behav);
r_value_true_pos = R_pos_true(1, 2);  % This will give you the correlation coefficient
p_value_true_pos = P_pos_true(1, 2);  % This will give you the p-value for the correlation

[R_neg_true, P_neg_true] = corrcoef(behav_pred_neg, all_behav);
r_value_true_neg = R_neg_true(1, 2);  % This will give you the correlation coefficient
p_value_true_neg = P_neg_true(1, 2);  % This will give you the p-value for the correlation

fprintf('\n======= RESULTS =======\n');
fprintf('Average across individual repetitions:\n');
fprintf('  Positive network: R = %.4f ± %.4f (mean p = %.4f)\n', R_pos_mean, R_pos_std, P_pos_mean);
fprintf('  Negative network: R = %.4f ± %.4f (mean p = %.4f)\n', R_neg_mean, R_neg_std, P_neg_mean);

fprintf('\nResults with averaged predictions over %d repetitions:\n', num_repetitions);
fprintf('  Positive network: R = %.4f (p = %.4f)\n', r_value_true_pos, p_value_true_pos);
fprintf('  Negative network: R = %.4f (p = %.4f)\n', r_value_true_neg, p_value_true_neg);

%% Create weighted positive and negative matrices

[pos_averageValues, neg_averageValues] = createWeightedMatrix(all_mats, pos_mask_final, neg_mask_final);

% save weighted matrix in .mat format

% save('AvAdjacencyMat.rsFC.negative.Gresponse.kfold.weighted.mat','neg_averageValues')
% save('AvAdjacencyMat.rsFC.positive.Gresponse.kfold.weighted.mat','pos_averageValues')

%% Find names of the connections that are significant (i.e., the selected network)

[conn_pos, conn_neg] = findSignificantConnections(pos_mask_final, neg_mask_final, connectomeDirectory, mask_apply{1});

%% Create consensus networks based on edge consistency
consistency_threshold = 100; 

[consensus_pos_mask, consensus_neg_mask] = createConsensusNetwork(all_mats, all_behav, no_nodes, no_subj, ...
    thresh, corr_method{1}, k_folds, num_repetitions, consistency_threshold, random_seed, modality{1}, verbose);

%% Find connection names for consensus networks
[consensus_conn_pos, consensus_conn_neg] = findSignificantConnections(consensus_pos_mask, consensus_neg_mask, connectomeDirectory, mask_apply{1});


%% Calculate network strength using consensus masks
consensus_train_sumpos = zeros(no_subj, 1);
consensus_train_sumneg = zeros(no_subj, 1);

for ss = 1:no_subj
    subj_mat = all_mats(:,:,ss);
    subj_mat(isnan(subj_mat)) = 0;
    consensus_train_sumpos(ss) = sum(sum(subj_mat .* consensus_pos_mask)) / 2;
    consensus_train_sumneg(ss) = sum(sum(subj_mat .* consensus_neg_mask)) / 2;
end

%% Plot relationship between consensus network strength and behavior

fprintf('\n===== PLOTTING DECISION =====\n');

% For positive network 
if p_value_true_pos < 0.05 && r_value_true_pos > 0
    fprintf('Plotting positive network consensus strength vs behavior\n');
    fprintf('Decision based on ORIGINAL prediction performance: R=%.4f, p=%.4f\n', ...
        r_value_true_pos, p_value_true_pos);
    
    figure;
    scatter(consensus_train_sumpos, all_behav, 100, 'k', 'filled');
    hold on;
    lsline;
    title('Consensus Positive Network');
    xlabel('Network Strength');
    ylabel(mainmetrics);
    % Calculate correlation for display in the plot
    [R_temp, P_temp] = corrcoef(consensus_train_sumpos, all_behav);
    r_text = sprintf('R = %.4f, p = %.4f', R_temp(1,2), P_temp(1,2));
    text(min(consensus_train_sumpos), max(all_behav), r_text, 'VerticalAlignment', 'top');
    grid on;
else
    fprintf('NOT plotting positive network - original prediction did not meet criteria\n');
    fprintf('Original prediction performance: R=%.4f, p=%.4f\n', r_value_true_pos, p_value_true_pos);
end

% For negative network 
if p_value_true_neg < 0.05 && r_value_true_neg > 0
    fprintf('Plotting negative network consensus strength vs behavior\n');
    fprintf('Decision based on ORIGINAL prediction performance: R=%.4f, p=%.4f\n', ...
        r_value_true_neg, p_value_true_neg);
    
    figure;
    scatter(consensus_train_sumneg, all_behav, 100, 'k', 'filled');
    hold on;
    lsline;
    title('Consensus Negative Network');
    xlabel('Network Strength');
    ylabel(mainmetrics);
    % Calculate correlation for display in the plot
    [R_temp, P_temp] = corrcoef(consensus_train_sumneg, all_behav);
    r_text = sprintf('R = %.4f, p = %.4f', R_temp(1,2), P_temp(1,2));
    text(min(consensus_train_sumneg), max(all_behav), r_text, 'VerticalAlignment', 'top');
    grid on;
else
    fprintf('NOT plotting negative network - original prediction did not meet criteria\n');
    fprintf('Original prediction performance: R=%.4f, p=%.4f\n', r_value_true_neg, p_value_true_neg);
end
%% Plot relationship between predicted and actual behavior values
fprintf('\n===== PLOTTING PREDICTED VS ACTUAL =====\n');

% For positive network predictions 
if p_value_true_pos < 0.05 && r_value_true_pos > 0
    fprintf('Plotting positive network predictions vs actual behavior\n');
    fprintf('Decision based on ORIGINAL prediction performance: R=%.4f, p=%.4f\n', ...
        r_value_true_pos, p_value_true_pos);
    
    figure;
    scatter(behav_pred_pos, all_behav, 100, 'k', 'filled');
    hold on;
    lsline;
    title('Positive Network Predictions');
    xlabel('Predicted Values');
    ylabel(['Actual ' mainmetrics]);
    % Display correlation values in the plot
    r_text = sprintf('R = %.4f, p = %.4f', r_value_true_pos, p_value_true_pos);
    text(min(behav_pred_pos), max(all_behav), r_text, 'VerticalAlignment', 'top');
    grid on;
else
    fprintf('NOT plotting positive network predictions - did not meet criteria\n');
    fprintf('Original prediction performance: R=%.4f, p=%.4f\n', r_value_true_pos, p_value_true_pos);
end

% For negative network predictions 
if p_value_true_neg < 0.05 && r_value_true_neg > 0
    fprintf('Plotting negative network predictions vs actual behavior\n');
    fprintf('Decision based on ORIGINAL prediction performance: R=%.4f, p=%.4f\n', ...
        r_value_true_neg, p_value_true_neg);
    
    figure;
    scatter(behav_pred_neg, all_behav, 100, 'k', 'filled');
    hold on;
    lsline;
    title('Negative Network Predictions');
    xlabel('Predicted Values');
    ylabel(['Actual ' mainmetrics]);
    % Display correlation values in the plot
    r_text = sprintf('R = %.4f, p = %.4f', r_value_true_neg, p_value_true_neg);
    text(min(behav_pred_neg), max(all_behav), r_text, 'VerticalAlignment', 'top');
    grid on;
else
    fprintf('NOT plotting negative network predictions - did not meet criteria\n');
    fprintf('Original prediction performance: R=%.4f, p=%.4f\n', r_value_true_neg, p_value_true_neg);
end

%% Save consensus networks as binary matrices
fprintf('\n===== SAVING CONSENSUS NETWORKS AS BINARY MATRICES =====\n');

% Get current directory and determine base directory
currentDirectory = pwd;
[mainscriptDirectory, ~, ~] = fileparts(currentDirectory);
[baseDirectory, ~, ~] = fileparts(mainscriptDirectory);

% Set up output directory for saving matrices
outputDirectory = fullfile(baseDirectory, 'results', 'binary_matrices');
if ~exist(outputDirectory, 'dir')
    mkdir(outputDirectory);
    fprintf('Created output directory: %s\n', outputDirectory);
end

% Generate filename identifier from condition and metric
filename_id = sprintf('%s_%s', condition, mainmetrics);

% Check which networks are predictive
has_predictive_pos = (p_value_true_pos < 0.05 && r_value_true_pos > 0);
has_predictive_neg = (p_value_true_neg < 0.05 && r_value_true_neg > 0);

% Ensure matrices have the correct dimensions
matrix_size = size(all_mats, 1); % Should be 84x84
fprintf('Creating matrices with dimensions %dx%d to match connectome size\n', matrix_size, matrix_size);

% Process positive consensus network
if has_predictive_pos
    fprintf('\nProcessing predictive positive consensus network:\n');
    
    % Create binary matrix from consensus mask - CONVERT TO DOUBLE HERE
    binary_consensus_pos = double(consensus_pos_mask);
    
    % Verify dimensions
    if size(binary_consensus_pos, 1) ~= matrix_size || size(binary_consensus_pos, 2) ~= matrix_size
        fprintf('  ⚠ Warning: Dimensions mismatch - resizing to %dx%d\n', matrix_size, matrix_size);
        % If dimensions don't match, this is a backup approach
        temp_matrix = zeros(matrix_size, matrix_size);
        min_rows = min(size(binary_consensus_pos, 1), matrix_size);
        min_cols = min(size(binary_consensus_pos, 2), matrix_size);
        temp_matrix(1:min_rows, 1:min_cols) = binary_consensus_pos(1:min_rows, 1:min_cols);
        binary_consensus_pos = temp_matrix;
    end
    
    % Save the matrix
    consPosMatsFile = fullfile(outputDirectory, ['consensus_pos_', filename_id, '.mat']);
    save(consPosMatsFile, 'binary_consensus_pos');
    
    % Report
    fprintf('  Saved consensus positive network with %d connections\n', sum(binary_consensus_pos(:))/2);
    fprintf('  Matrix dimensions: %dx%d\n', size(binary_consensus_pos, 1), size(binary_consensus_pos, 2));
    fprintf('  File: %s\n', consPosMatsFile);
else
    fprintf('\n⚠ Positive network is not predictive (r=%.4f, p=%.4f) - not saving\n', ...
        r_value_true_pos, p_value_true_pos);
end

% Process negative consensus network
if has_predictive_neg
    fprintf('\nProcessing predictive negative consensus network:\n');
    
    % Create binary matrix from consensus mask - CONVERT TO DOUBLE HERE
    binary_consensus_neg = double(consensus_neg_mask);
    
    % Verify dimensions
    if size(binary_consensus_neg, 1) ~= matrix_size || size(binary_consensus_neg, 2) ~= matrix_size
        fprintf('  ⚠ Warning: Dimensions mismatch - resizing to %dx%d\n', matrix_size, matrix_size);
        % If dimensions don't match, this is a backup approach
        temp_matrix = zeros(matrix_size, matrix_size);
        min_rows = min(size(binary_consensus_neg, 1), matrix_size);
        min_cols = min(size(binary_consensus_neg, 2), matrix_size);
        temp_matrix(1:min_rows, 1:min_cols) = binary_consensus_neg(1:min_rows, 1:min_cols);
        binary_consensus_neg = temp_matrix;
    end
    
    % Save the matrix
    consNegMatsFile = fullfile(outputDirectory, ['consensus_neg_', filename_id, '.mat']);
    save(consNegMatsFile, 'binary_consensus_neg');
    
    % Report
    fprintf('  Saved consensus negative network with %d connections\n', sum(binary_consensus_neg(:))/2);
    fprintf('  Matrix dimensions: %dx%d\n', size(binary_consensus_neg, 1), size(binary_consensus_neg, 2));
    fprintf('  File: %s\n', consNegMatsFile);
else
    fprintf('\n⚠ Negative network is not predictive (r=%.4f, p=%.4f) - not saving\n', ...
        r_value_true_neg, p_value_true_neg);
end

if ~has_predictive_pos && ~has_predictive_neg
    fprintf('\n⚠ Neither network is predictive. No binary matrices saved.\n');
end

fprintf('\nConsensus network saving completed.\n');

%% Create weighted consensus networks with average connectivity values
fprintf('\n===== CREATING WEIGHTED CONSENSUS NETWORKS =====\n');

% Calculate the average connectivity matrix across all subjects
avg_connectome = mean(all_mats, 3);

% Create weighted consensus networks by multiplying binary masks with average connectome
if has_predictive_pos
    fprintf('\nCreating weighted positive consensus network:\n');
    
    % Multiply binary mask with average connectome
    weighted_consensus_pos = avg_connectome .* consensus_pos_mask;
    
    % Save the weighted matrix
    weighted_pos_file = fullfile(outputDirectory, ['weighted_consensus_pos_', filename_id, '.mat']);
    save(weighted_pos_file, 'weighted_consensus_pos');
    
    % Report
    fprintf('  Saved weighted consensus positive network\n');
    fprintf('  Non-zero values: %d (meaningful connections)\n', sum(weighted_consensus_pos(:) ~= 0));
    fprintf('  File: %s\n', weighted_pos_file);
    
    % Optional: Print some statistics about the values
    nonzero_vals = weighted_consensus_pos(weighted_consensus_pos ~= 0);
    if ~isempty(nonzero_vals)
        fprintf('  Value statistics: min=%.4f, max=%.4f, mean=%.4f, std=%.4f\n', ...
            min(nonzero_vals), max(nonzero_vals), mean(nonzero_vals), std(nonzero_vals));
    end
else
    fprintf('\n⚠ Positive network is not predictive - not creating weighted version\n');
end

if has_predictive_neg
    fprintf('\nCreating weighted negative consensus network:\n');
    
    % Multiply binary mask with average connectome
    weighted_consensus_neg = avg_connectome .* consensus_neg_mask;
    
    % Save the weighted matrix
    weighted_neg_file = fullfile(outputDirectory, ['weighted_consensus_neg_', filename_id, '.mat']);
    save(weighted_neg_file, 'weighted_consensus_neg');
    
    % Report
    fprintf('  Saved weighted consensus negative network\n');
    fprintf('  Non-zero values: %d (meaningful connections)\n', sum(weighted_consensus_neg(:) ~= 0));
    fprintf('  File: %s\n', weighted_neg_file);
    
    % Optional: Print some statistics about the values
    nonzero_vals = weighted_consensus_neg(weighted_consensus_neg ~= 0);
    if ~isempty(nonzero_vals)
        fprintf('  Value statistics: min=%.4f, max=%.4f, mean=%.4f, std=%.4f\n', ...
            min(nonzero_vals), max(nonzero_vals), mean(nonzero_vals), std(nonzero_vals));
    end
else
    fprintf('\n⚠ Negative network is not predictive - not creating weighted version\n');
end

if ~has_predictive_pos && ~has_predictive_neg
    fprintf('\n⚠ Neither network is predictive. No weighted matrices created.\n');
end

fprintf('\nWeighted consensus network creation completed.\n');

%% Permutation statistics

if strcmp(permutation{1},'yes')
    % For K-fold cross-validation, we need to modify the permutation test
    fprintf('\nPerforming permutation test with averaged predictions\n')
    [pval_pos, pval_neg] = PermutationTest_kfold_avg(r_value_true_pos, r_value_true_neg, all_behav, all_mats, no_nodes, no_subj, thresh, corr_method{1}, k_folds, no_iterations, modality{1}, random_seed, num_repetitions, verbose);
end

clearvars -except p_value_true_pos p_value_true_neg pval_pos pval_neg conn_pos conn_neg consensus_conn_pos consensus_conn_neg
