function [pval_pos, pval_neg] = PermutationTest(R_pos_true, R_neg_true, all_behav, all_mats, no_nodes, no_subj, thresh, corr_method,no_iterations)
    % Perform permutation testing to assess the significance of the prediction

    % Extract true prediction correlations
    true_prediction_r_pos = R_pos_true(1,2);
    true_prediction_r_neg = R_neg_true(1,2);

    % Number of iterations for permutation testing
    prediction_r = zeros(no_iterations, 2);
    prediction_r(1,1) = true_prediction_r_pos;
    prediction_r(1,2) = true_prediction_r_neg;

    % Create estimate distribution of the test statistic via random shuffles of data labels
    for it = 2:no_iterations
        fprintf('\n Performing iteration %d out of %d', it, no_iterations)
        new_behav = all_behav(randperm(no_subj));
        
        [~, ~, prediction_r(it,1), prediction_r(it,2)] = predict_behaviour_perm(all_mats, new_behav, no_nodes, no_subj, thresh, corr_method);
    end

    % Calculate p-values
    sorted_prediction_r_pos = sort(prediction_r(:,1), 'descend');
    position_pos = find(sorted_prediction_r_pos == true_prediction_r_pos);
    pval_pos = position_pos(1) / no_iterations;

    sorted_prediction_r_neg = sort(prediction_r(:,2), 'descend');
    position_neg = find(sorted_prediction_r_neg == true_prediction_r_neg);
    pval_neg = position_neg(1) / no_iterations;
    
    % Print results
    fprintf('\nAfter %d permutations, the p-value for the positive network is: %.4f\n', no_iterations, pval_pos);
    fprintf('After %d permutations, the p-value for the negative network is: %.4f\n', no_iterations, pval_neg);
end