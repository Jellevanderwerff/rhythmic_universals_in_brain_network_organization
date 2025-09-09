function [consensus_pos_mask, consensus_neg_mask] = createConsensusNetwork(all_mats, all_behav, no_nodes, no_subj, thresh, corr_method, k_folds, num_repetitions, consistency_threshold, random_seed, modality, verbose)
% Creates a consensus network by identifying edges that appear consistently across
% multiple repetitions of k-fold cross-validation
%
% Parameters:
% all_mats            - Full connectome (nodes x nodes x subjects)
% all_behav           - Behavioral data for prediction
% no_nodes            - Number of nodes in connectome
% no_subj             - Number of subjects
% thresh              - p-value threshold for edge selection
% corr_method         - Correlation method ('Pearson', 'Spearman', 'partial')
% k_folds             - Number of folds for cross-validation
% num_repetitions     - Number of times to repeat the analysis
% consistency_threshold - Percentage threshold for edge inclusion (0-100)
% random_seed         - Seed for reproducibility
% modality            - 'functional' or 'structural_FBC'
% verbose             - Verbosity level (0=minimal, 1=detailed)
%
% Returns:
% consensus_pos_mask  - Binary mask of consistent positive edges
% consensus_neg_mask  - Binary mask of consistent negative edges

% Initialize counters for edge selection frequency
edge_count_pos = zeros(no_nodes, no_nodes);
edge_count_neg = zeros(no_nodes, no_nodes);

% Track overall prediction performance
R_pos_all = zeros(num_repetitions, 1);
R_neg_all = zeros(num_repetitions, 1);

fprintf('Running %d repetitions to identify consistent edges...\n', num_repetitions);

% Run multiple repetitions
for rep = 1:num_repetitions
    if verbose > 0
        fprintf('Repetition %d of %d\n', rep, num_repetitions);
    end
    
    % Set reproducible random seed for this repetition
    rng(random_seed + rep - 1);
    
    % For each repetition, we'll store the masks from the k-fold CV
    pos_masks_folds = zeros(no_nodes, no_nodes, k_folds);
    neg_masks_folds = zeros(no_nodes, no_nodes, k_folds);
    
    % Randomly permute subjects for k-fold assignment
    subj_indices = randperm(no_subj);
    
    % Determine fold size
    fold_size = floor(no_subj / k_folds);
    remainder = mod(no_subj, k_folds);
    
    % Create fold indices
    fold_indices = cell(k_folds, 1);
    start_idx = 1;
    for fold = 1:k_folds
        if fold <= remainder
            end_idx = start_idx + fold_size;
        else
            end_idx = start_idx + fold_size - 1;
        end
        fold_indices{fold} = subj_indices(start_idx:end_idx);
        start_idx = end_idx + 1;
    end
    
    % Perform k-fold CV and collect masks
    behav_pred_pos = zeros(no_subj, 1);
    behav_pred_neg = zeros(no_subj, 1);
    
    for fold = 1:k_folds
        if verbose > 0
            fprintf('  Processing fold %d of %d\n', fold, k_folds);
        end
        
        % Get test/train indices for this fold
        test_indices = fold_indices{fold};
        train_indices = setdiff(1:no_subj, test_indices);
        
        % Extract training data
        train_mats = all_mats(:, :, train_indices);
        train_mats(isnan(train_mats)) = 0;
        train_vcts = reshape(train_mats, [], size(train_mats, 3));
        train_behav = all_behav(train_indices);
        
        % Correlate edges with behavior
        if strcmp(corr_method, 'Pearson')
            [r_mat, p_mat] = corr(train_vcts', train_behav);
            r_mat = reshape(r_mat, no_nodes, no_nodes);
            p_mat = reshape(p_mat, no_nodes, no_nodes);
        elseif strcmp(corr_method, 'Spearman')
            [r_mat, p_mat] = corr(train_vcts', train_behav, 'type', 'Spearman');
            r_mat = reshape(r_mat, no_nodes, no_nodes);
            p_mat = reshape(p_mat, no_nodes, no_nodes);
        elseif strcmp(corr_method, 'partial')
            % Get current directory and navigate to base directory
            currentDirectory = pwd;
            [mainscriptDirectory, ~, ~] = fileparts(currentDirectory);
            [baseDirectory, ~, ~] = fileparts(mainscriptDirectory);
            
            % Set directory paths for nuisance regressors
            behaviourDirectory = fullfile(baseDirectory, 'data', 'CPM', 'behaviour');
            filePath = fullfile(behaviourDirectory, ['nuisanceregressors.csv']);
            datanuisance = readtable(filePath);
            age = datanuisance.('age_demeaned');
            sex = datanuisance.('sex');
            
            % Combine age and sex into one matrix, using only training subjects
            covariates = [sex(train_indices), age(train_indices)];
            
            [r_mat, p_mat] = partialcorr(train_vcts', train_behav, covariates);
            r_mat = reshape(r_mat, no_nodes, no_nodes);
            p_mat = reshape(p_mat, no_nodes, no_nodes);
        end
        
        % Create masks for this fold
        pos_mask_fold = zeros(no_nodes, no_nodes);
        neg_mask_fold = zeros(no_nodes, no_nodes);
        
        % Find significant edges
        pos_edges = find(r_mat > 0 & p_mat < thresh);
        neg_edges = find(r_mat < 0 & p_mat < thresh);
        
        pos_mask_fold(pos_edges) = 1;
        neg_mask_fold(neg_edges) = 1;
        
        % Store masks for this fold
        pos_masks_folds(:,:,fold) = pos_mask_fold;
        neg_masks_folds(:,:,fold) = neg_mask_fold;
    end
    
    % Combine masks across folds using OR operation (edge appears in any fold)
    pos_mask_rep = any(pos_masks_folds, 3);
    neg_mask_rep = any(neg_masks_folds, 3);
    
    % Add to edge counters
    edge_count_pos = edge_count_pos + double(pos_mask_rep);
    edge_count_neg = edge_count_neg + double(neg_mask_rep);
    
    % Run the original k-fold CV to get performance metrics
    [behav_pred_pos, behav_pred_neg, R_pos, P_pos, R_neg, P_neg, ~, ~, ~, ~] = predict_behaviour_kfold_quiet(modality, all_mats, all_behav, no_nodes, no_subj, thresh, corr_method, k_folds, 0);
    
    R_pos_all(rep) = R_pos;
    R_neg_all(rep) = R_neg;
end

% Calculate edge consistency (percentage of repetitions)
edge_freq_pos = (edge_count_pos / num_repetitions) * 100;
edge_freq_neg = (edge_count_neg / num_repetitions) * 100;

% Create consensus masks based on threshold
consensus_pos_mask = edge_freq_pos >= consistency_threshold;
consensus_neg_mask = edge_freq_neg >= consistency_threshold;

% Report statistics
fprintf('\n===== CONSENSUS NETWORK STATISTICS =====\n');
fprintf('Consistency threshold: %.1f%% (edges present in this %% of repetitions)\n', consistency_threshold);
fprintf('Positive network: %d consistent edges (%.1f%% of all possible edges)\n', ...
    sum(consensus_pos_mask(:)), (sum(consensus_pos_mask(:))/(no_nodes^2))*100);
% Add a small pause between print statements to prevent overlap
pause(0.01);
fprintf('Negative network: %d consistent edges (%.1f%% of all possible edges)\n', ...
    sum(consensus_neg_mask(:)), (sum(consensus_neg_mask(:))/(no_nodes^2))*100);
% Add another small pause
pause(0.01);
fprintf('Mean prediction performance (avg R between predicted vs actual): Pos = %.4f, Neg = %.4f\n', ...
    mean(R_pos_all), mean(R_neg_all));
end