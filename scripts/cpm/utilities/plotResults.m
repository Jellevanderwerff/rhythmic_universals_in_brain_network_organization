function plotResults(behav_pred_pos, behav_pred_neg, all_behav, r_value_true_pos, p_value_true_pos, r_value_true_neg, p_value_true_neg, mainmetrics, data, train_sumpos, train_sumneg)
    % Plot results of the analysis

    % Plot for positive predictions
    figure(1);
    plot(behav_pred_pos, all_behav, 'r.');
    lsline;
    xlabel('predicted');
    ylabel('true');
    strPos = {['R = ', num2str(r_value_true_pos)], ['P = ', num2str(p_value_true_pos)]};
    text(min(behav_pred_pos), max(all_behav), strPos, 'VerticalAlignment', 'top', 'Color', 'r');

    % Plot for negative predictions
    figure(2);
    plot(behav_pred_neg, all_behav, 'b.');
    lsline;
    xlabel('predicted');
    ylabel('true');
    strNeg = {['R = ', num2str(r_value_true_neg)], ['P = ', num2str(p_value_true_neg)]};
    text(min(behav_pred_neg), max(all_behav), strNeg, 'VerticalAlignment', 'top', 'Color', 'b');

    % Additional plots for specific metrics
    if ismember(mainmetrics, {'G_resp', 'entropy_diff_norm_q_avg', 'iti_ioi_cov_diff_avg', 'simple_ratio_introduced', 'binary_or_ternary_introduced', 'isochrony_introduced'})
        learningIndexes = {'asynchrony_norm_abs_avg', 'tempo_deviation_abs_avg', 'edit_distance_norm_q_avg'};
        NamesLearningIndexes = {'Asynchrony normalized', 'Absolute tempo deviation', 'Transmission Error'};
        
        for i = 1:length(learningIndexes)
            [R_pos, P_pos] = corrcoef(train_sumpos, data.(learningIndexes{i}));
            [R_neg, P_neg] = corrcoef(train_sumneg, data.(learningIndexes{i}));
            
            figure;
            scatter(train_sumpos, data.(learningIndexes{i}), 'r.');
            hold on;
            scatter(train_sumneg, data.(learningIndexes{i}), 'b.');
            h = lsline;
            set(h(1), 'Color', 'b', 'LineWidth', 1.5);
            set(h(2), 'Color', 'r', 'LineWidth', 1.5);
            
            strPos = {['R_pos = ', num2str(R_pos(1, 2))], ['P_pos = ', num2str(P_pos(1, 2))]};
            strNeg = {['R_neg = ', num2str(R_neg(1, 2))], ['P_neg = ', num2str(P_neg(1, 2))]};
            text(min(train_sumpos), max(data.(learningIndexes{i})), strPos, 'VerticalAlignment', 'top', 'Color', 'r');
            text(min(train_sumneg), 0.9*max(data.(learningIndexes{i})), strNeg, 'VerticalAlignment', 'top', 'Color', 'b');
            
            title(NamesLearningIndexes{i});
            xlabel('neural strength');
            ylabel(NamesLearningIndexes{i});
            grid on;
            hold off;
        end
    end

    % Plot relating the strength of significant structural metrics with metrics of interest
    markerSize = 100;

    if p_value_true_pos < 0.05 && r_value_true_pos > 0
        [R_pos, P_pos] = corrcoef(train_sumpos, all_behav);
        figure;
        scatter(train_sumpos, all_behav, markerSize, 'k', 'filled');
        hold on;
        lsline;
        xlabel('neural strength');
        ylabel(mainmetrics);
    end

    if p_value_true_neg < 0.05 && r_value_true_neg > 0
        figure;
        [R_neg, P_neg] = corrcoef(train_sumneg, all_behav);
        scatter(train_sumneg, all_behav, markerSize, 'k', 'filled');
        hold on;
        lsline;
        xlabel('neural strength');
        ylabel(mainmetrics);
    end
end