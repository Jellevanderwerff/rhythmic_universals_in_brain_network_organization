clear, clc

%% Input parameters %%

modality = {'functional'}; % 'functional' or 'structural_FBC' 
corr_method = {'Spearman'}; % 'Pearson', 'Spearman','partial'
thresh = 0.01; % threshold for feature selection: 0.01, 0.05, etc.
mainmetrics = 'G_resp'; %'G_resp', 'entropy_diff_norm_q_avg','asynchrony_norm_abs_avg', 'tempo_deviation_abs_avg','isochrony_introduced','binary_or_ternary_introduced, 'edit_distance_norm_q_avg'
condition = '6005'; % '4004'(400 ms 4 events)'4005';'6004'; '6005'
figure ={'yes'}; % create figure ('yes') or not ('no')

% permutation parameters
permutation={'no'}; % 'yes' or 'no'
no_iterations = 1000;

%% Define the directory paths and load data

[connectomeDirectory, ~, ~, all_mats, all_behav,data] = setupDirectoriesAndLoadData(modality, condition, mainmetrics);

%% get basic info (number of subjects and nodes)
no_subj = size(all_mats,3);
no_nodes = size(all_mats,1);

%% Connectome-based predictive modeling (CPM) analysis 
fprintf('\n Performing the calculation of the true experiment \n')
[behav_pred_pos,behav_pred_neg,R_pos,R_neg,pos_mask,neg_mask,storeLast_pos,storeLast_neg,train_sumpos,train_sumneg]=predict_behaviour(modality,all_mats, all_behav, no_nodes, no_subj, thresh, corr_method)
    
pos_mask_true = pos_mask;
neg_mask_true = neg_mask;

train_sumneg(end+1,:) = storeLast_neg;
train_sumpos(end+1,:) = storeLast_pos;

%% Create weighted positive and negative matrices

[pos_averageValues, neg_averageValues] = createWeightedMatrix(all_mats, pos_mask, neg_mask);

% save weighted matrix in .mat format

% save('AvAdjacencyMat.rsFC.negative.Gresponse.weighted.mat','neg_averageValues')
% save('AvAdjacencyMat.rsFC.negative.EntropyDiff.weighted.mat','neg_averageValues')
% save('AvAdjacencyMat.rsFC.positive.CoV.weighted.mat','pos_averageValues')

%% Calculate correlation coefficients 

[R_pos_true, P_pos_true] = corrcoef(behav_pred_pos, all_behav);

r_value_true_pos = R_pos_true(1, 2);  % This will give you the correlation coefficient
p_value_true_pos = P_pos_true(1, 2);  % This will give you the p-value for the correlation

[R_neg_true, P_neg_true] = corrcoef(behav_pred_neg, all_behav);

r_value_true_neg = R_neg_true(1, 2);  % This will give you the correlation coefficient
p_value_true_neg = P_neg_true(1, 2);  % This will give you the p-value for the correlation


%% Find names of the connections that are significant (i.e., the selected network)

[conn_pos, conn_neg] = findSignificantConnections(pos_mask_true, neg_mask_true, connectomeDirectory);


%% Create figures

if strcmp(figure,'yes')
    plotResults(behav_pred_pos, behav_pred_neg, all_behav, r_value_true_pos, p_value_true_pos, r_value_true_neg, p_value_true_neg, mainmetrics, data, train_sumpos, train_sumneg);
end

%% Permutation statistics

if strcmp(permutation,'yes')
    [pval_pos, pval_neg] = PermutationTest(R_pos_true, R_neg_true, all_behav, all_mats, no_nodes, no_subj, thresh, corr_method,no_iterations);
end

clearvars -except p_value_true_pos p_value_true_neg pval_pos pval_neg
