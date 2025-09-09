function [pval_pos, pval_neg] = PermutationTest_kfold_avg(r_value_true_pos, r_value_true_neg, all_behav, all_mats, no_nodes, no_subj, thresh, corr_method, k_folds, no_iterations, modality, random_seed, num_repetitions, verbose)
    % Perform permutation testing to assess the significance of the prediction
    % using averaged predictions across multiple repetitions
    
    % Extract true prediction correlations
    true_prediction_r_pos = r_value_true_pos;
    true_prediction_r_neg = r_value_true_neg;

    % Number of iterations for permutation testing
    prediction_r = zeros(no_iterations, 2);
    prediction_r(1,1) = true_prediction_r_pos;
    prediction_r(1,2) = true_prediction_r_neg;

    % Create estimate distribution of the test statistic via random shuffles of data labels
    for it = 2:no_iterations
        fprintf('\nPermutation iteration %d of %d', it, no_iterations);
        
        % Initialize arrays to store predictions from multiple repetitions
        behav_pred_pos_all = zeros(no_subj, num_repetitions);
        behav_pred_neg_all = zeros(no_subj, num_repetitions);
        
        % Randomly shuffle the behavioral data
        new_behav = all_behav(randperm(no_subj));
        
        % Run multiple repetitions with the same permuted labels
        for rep = 1:num_repetitions
            % Set a different random seed for each repetition (but deterministic)
            rng(random_seed + it*1000 + rep - 1);
            
            % Perform k-fold cross-validation with permuted labels and suppressed output
            [behav_pred_pos, behav_pred_neg, ~, ~, ~, ~, ~, ~, ~, ~] = predict_behaviour_kfold_quiet(modality, all_mats, new_behav, no_nodes, no_subj, thresh, corr_method, k_folds, verbose);
            
            % Store predictions
            behav_pred_pos_all(:, rep) = behav_pred_pos;
            behav_pred_neg_all(:, rep) = behav_pred_neg;
        end
        
        % Average predictions across repetitions
        behav_pred_pos_avg = mean(behav_pred_pos_all, 2);
        behav_pred_neg_avg = mean(behav_pred_neg_all, 2);
        
        % Calculate correlations for permuted data
        [R_pos_perm, ~] = corrcoef(behav_pred_pos_avg, new_behav);
        [R_neg_perm, ~] = corrcoef(behav_pred_neg_avg, new_behav);
        
        prediction_r(it,1) = R_pos_perm(1,2);
        prediction_r(it,2) = R_neg_perm(1,2);
    end

    % Calculate p-values
    sorted_prediction_r_pos = sort(prediction_r(:,1), 'descend');
    position_pos = find(sorted_prediction_r_pos == true_prediction_r_pos);
    if isempty(position_pos)
        % If exact match is not found, find closest position
        [~, position_pos] = min(abs(sorted_prediction_r_pos - true_prediction_r_pos));
    end
    pval_pos = position_pos(1) / no_iterations;

    sorted_prediction_r_neg = sort(prediction_r(:,2), 'descend');
    position_neg = find(sorted_prediction_r_neg == true_prediction_r_neg);
    if isempty(position_neg)
        % If exact match is not found, find closest position
        [~, position_neg] = min(abs(sorted_prediction_r_neg - true_prediction_r_neg));
    end
    pval_neg = position_neg(1) / no_iterations;
    
    % Print results
    fprintf('\nAfter %d permutations, the p-value for the positive network is: %.4f\n', no_iterations, pval_pos);
    fprintf('After %d permutations, the p-value for the negative network is: %.4f\n', no_iterations, pval_neg);
end