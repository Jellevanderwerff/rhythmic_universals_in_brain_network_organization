
function [behav_pred_pos, behav_pred_neg, R_pos, P_pos, R_neg, P_neg, pos_mask, neg_mask, train_sumpos, train_sumneg] = predict_behaviour_kfold_quiet(modality, all_mats, all_behav, no_nodes, no_subj, thresh, corr_method, k_folds, verbose)
% K-fold cross-validation implementation of CPM with controlled output verbosity
% Set up directories for partial regression

currentDirectory = pwd;
[mainscriptDirectory, ~, ~] = fileparts(currentDirectory);
[baseDirectory, ~, ~] = fileparts(mainscriptDirectory);

% Initialize variables for predicted values
behav_pred_pos = zeros(no_subj, 1);
behav_pred_neg = zeros(no_subj, 1);
train_sumpos = zeros(no_subj, 1);
train_sumneg = zeros(no_subj, 1);

% Check if k_folds is valid
if k_folds > no_subj
    error('Number of folds cannot exceed number of subjects');
end

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

% Perform k-fold cross-validation
for fold = 1:k_folds
    if verbose > 0
        fprintf('  Processing fold %d of %d\n', fold, k_folds);
    end
    
    % Get test subjects for this fold
    test_indices = fold_indices{fold};
    
    % Get training subjects (all except test subjects)
    train_indices = setdiff(1:no_subj, test_indices);
    
    % Extract training data
    train_mats = all_mats(:, :, train_indices);
    train_mats(isnan(train_mats)) = 0;
    train_vcts = reshape(train_mats, [], size(train_mats, 3));
    train_behav = all_behav(train_indices);
    
    % Correlate all edges with behavior
    if strcmp(corr_method, 'Pearson')
        [r_mat, p_mat] = corr(train_vcts', train_behav);
        r_mat = reshape(r_mat, no_nodes, no_nodes);
        p_mat = reshape(p_mat, no_nodes, no_nodes);
        
    elseif strcmp(corr_method, 'Spearman')
        [r_mat, p_mat] = corr(train_vcts', train_behav, 'type', 'Spearman');
        r_mat = reshape(r_mat, no_nodes, no_nodes);
        p_mat = reshape(p_mat, no_nodes, no_nodes);
        
    elseif strcmp(corr_method, 'partial')
        behaviourDirectory = fullfile(baseDirectory, 'data', 'CPM', 'behaviour');
        filePath = fullfile(behaviourDirectory, ['nuisanceregressors.csv']);
        datanuisance = readtable(filePath);
        age = datanuisance.('age_demeaned');
        sex = datanuisance.('sex');
        %training = datanuisance.('musical_training');
        
        % Combine age and sex into one matrix, using only training subjects
        covariates = [sex(train_indices), age(train_indices)];
        %covariates = [training(train_indices)];
        
        [r_mat, p_mat] = partialcorr(train_vcts', train_behav, covariates);
        r_mat = reshape(r_mat, no_nodes, no_nodes);
        p_mat = reshape(p_mat, no_nodes, no_nodes);
    end
    
    % Set threshold and define masks
    pos_mask = zeros(no_nodes, no_nodes);
    neg_mask = zeros(no_nodes, no_nodes);
    
    % Find significant edges with positive and negative correlations
    pos_edges = find(r_mat > 0 & p_mat < thresh);
    neg_edges = find(r_mat < 0 & p_mat < thresh);
    
    pos_mask(pos_edges) = 1;
    neg_mask(neg_edges) = 1;
    
    % Calculate network strength for training subjects
    train_fold_sumpos = zeros(length(train_indices), 1);
    train_fold_sumneg = zeros(length(train_indices), 1);
    
    for ss = 1:length(train_indices)
        train_fold_sumpos(ss) = sum(sum(train_mats(:, :, ss) .* pos_mask)) / 2;
        train_fold_sumneg(ss) = sum(sum(train_mats(:, :, ss) .* neg_mask)) / 2;
    end
    
    % Build regression model on training subjects
    if strcmp(modality, 'structural_FBC')
        % Manual normalization for structural data
        if all(train_fold_sumpos == 0) || std(train_fold_sumpos) < eps(1)*1000
            % Handle case with zero or near-zero variance
            fit_pos = [0, mean(train_behav)]; % Just use mean behavior as prediction
            mu_pos_mean = 0;
            mu_pos_std = 1;
        else
            % Manual normalization
            mu_pos_mean = mean(train_fold_sumpos);
            mu_pos_std = std(train_fold_sumpos);
            train_fold_sumpos_norm = (train_fold_sumpos - mu_pos_mean) / mu_pos_std;
            fit_pos = polyfit(train_fold_sumpos_norm, train_behav, 1);
        end
        
        if all(train_fold_sumneg == 0) || std(train_fold_sumneg) < eps(1)*1000
            fit_neg = [0, mean(train_behav)];
            mu_neg_mean = 0;
            mu_neg_std = 1;
        else
            mu_neg_mean = mean(train_fold_sumneg);
            mu_neg_std = std(train_fold_sumneg);
            train_fold_sumneg_norm = (train_fold_sumneg - mu_neg_mean) / mu_neg_std;
            fit_neg = polyfit(train_fold_sumneg_norm, train_behav, 1);
        end
    else
        % Original approach for functional connectivity
        fit_pos = polyfit(train_fold_sumpos, train_behav, 1);
        fit_neg = polyfit(train_fold_sumneg, train_behav, 1);
    end
    
    % Test on left-out subjects
    for i = 1:length(test_indices)
        test_idx = test_indices(i);
        test_mat = all_mats(:, :, test_idx);
        test_mat(isnan(test_mat)) = 0;
        
        % Calculate network strength for test subject
        test_sumpos = sum(sum(test_mat .* pos_mask)) / 2;
        test_sumneg = sum(sum(test_mat .* neg_mask)) / 2;
        
        % Store network strengths for all subjects
        train_sumpos(test_idx) = test_sumpos;
        train_sumneg(test_idx) = test_sumneg;
        
        % Predict behavior using appropriate method based on modality
        if strcmp(modality, 'structural_FBC')
            % Apply manual normalization and prediction
            test_sumpos_norm = (test_sumpos - mu_pos_mean) / mu_pos_std;
            test_sumneg_norm = (test_sumneg - mu_neg_mean) / mu_neg_std;
            
            behav_pred_pos(test_idx) = fit_pos(1) * test_sumpos_norm + fit_pos(2);
            behav_pred_neg(test_idx) = fit_neg(1) * test_sumneg_norm + fit_neg(2);
        else
            % Original approach
            behav_pred_pos(test_idx) = fit_pos(1) * test_sumpos + fit_pos(2);
            behav_pred_neg(test_idx) = fit_neg(1) * test_sumneg + fit_neg(2);
        end
    end
end

% Compare predicted and observed scores
[corr_pos, p_pos] = corr(behav_pred_pos, all_behav);
[corr_neg, p_neg] = corr(behav_pred_neg, all_behav);

R_pos = corr_pos;
P_pos = p_pos;
R_neg = corr_neg;
P_neg = p_neg;

if verbose > 0
    fprintf('  Performance metrics:\n');
    fprintf('  Positive network correlation (R): %.4f, p-value: %.4f\n', R_pos, P_pos);
    fprintf('  Negative network correlation (R): %.4f, p-value: %.4f\n', R_neg, P_neg);
end
end