function [behav_pred_pos,behav_pred_neg,R_pos,R_neg,pos_mask,neg_mask,storeLast_pos,storeLast_neg,train_sumpos,train_sumneg] = predict_behaviour(modality, all_mats, all_behav, no_nodes, no_subj, thresh, corr_method)

% initialise vars for 'predicted' values
behav_pred_pos = zeros(no_subj,1);
behav_pred_neg = zeros(no_subj,1);

%% Divide data into training and test sets for cross-validation

for leftout = 1:no_subj
    
    %fprintf('\n Leaving out subject # %6.3f', leftout)
    
    %leave out subjects from matrices and behaviour
    train_mats = all_mats;
    train_mats(:,:,leftout) = [];
    train_mats(isnan(train_mats)) = 0;

    train_vcts = reshape(train_mats,[],size(train_mats,3));
    
    train_behav = all_behav;
    train_behav(leftout) = [];
   
    
    % correlate all edges with behavior
    if strcmp(corr_method,'Pearson')      
        [r_mat,p_mat] = corr(train_vcts',train_behav);
        r_mat = reshape(r_mat, no_nodes, no_nodes);
        p_mat = reshape(p_mat, no_nodes, no_nodes);
        
    elseif strcmp(corr_method,'Spearman')
        [r_mat,p_mat] = corr(train_vcts',train_behav,'type', 'Spearman');
        r_mat = reshape(r_mat, no_nodes, no_nodes);
        p_mat = reshape(p_mat, no_nodes, no_nodes);
        
    elseif strcmp(corr_method,'partial')
        filePath='nuisanceregressors.csv';
        datanuisance = readtable(filePath);
        age = datanuisance.('age_demeaned');
        sex = datanuisance.('sex');
        
        % Combine age and sex into one matrix
        age(leftout,:) = [];
        sex(leftout,:) = [];
        covariates = [sex, age];          
        

        [r_mat,p_mat] = partialcorr(train_vcts',train_behav, covariates);
        r_mat = reshape(r_mat, no_nodes, no_nodes);
        p_mat = reshape(p_mat, no_nodes, no_nodes);
        
    elseif strcmp(corr_method,'robust_regression')
        % TODO
    end
    
    
    %% set threshold and define masks
    
    pos_mask = zeros(no_nodes, no_nodes);
    neg_mask = zeros(no_nodes, no_nodes);
    
    % pos edges are the indexes where the correlation with behaviour is
    % positive (N.B. they are not positive FC values). Viceversa for
    % negative edges.
    pos_edges = find(r_mat > 0 & p_mat <thresh);
    neg_edges = find(r_mat < 0 & p_mat <thresh);
    
    pos_mask(pos_edges) = 1;
    neg_mask(neg_edges) = 1;
    
    % use the mask to get the sum of all significant edge values --e.g., connectivity strength-- in TRAIN subj (divide then by 2 to control for the
    % fact the the matrices are symmetric)
    
    train_sumpos = zeros(no_subj-1,1);
    train_sumneg = zeros(no_subj-1,1);
    
    for ss = 1:size(train_sumpos)
        train_sumpos(ss) = sum(sum(train_mats(:,:,ss).*pos_mask))/2;
        train_sumneg(ss) = sum(sum(train_mats(:,:,ss).*neg_mask))/2;
    end
    
    % build model on training subjects. Polyfit is fitting a first-degree 
    % polynomial, or a linear model, to the data. polyfit is attempting to find 
    % the line that best fits your data according to the least squares criterion, 
    % which minimizes the sum of the squares of the residuals. The residuals are 
    % the differences between the observed (actual) and fitted (predicted) data values.
    % output: fit_pos(1) = slope; fit_pos(2) = y-intercept
    
    fit_pos = polyfit(train_sumpos, train_behav,1);
    fit_neg = polyfit(train_sumneg, train_behav,1);
    
    % run model to TEST on subject
    
    test_mat = all_mats(:,:,leftout);
    test_mat(isnan(test_mat)) = 0;
    test_sumpos = sum(sum(test_mat.* pos_mask))/2;
    test_sumneg = sum(sum(test_mat.* neg_mask))/2;
    
    % predicted value using the formula: y = mx + b
    behav_pred_pos(leftout) = fit_pos(1)*test_sumpos + fit_pos(2);
    behav_pred_neg(leftout) = fit_neg(1)*test_sumneg + fit_neg(2);
    
    % when leftout = 1, store the last value in the vector train_sumpos and
    % trainsumneg. In this way, you will create a train_neg and train_pos
    % (connection strength of the network) with all 47 values (and not just
    % 46)
    
    if leftout ==1
        storeLast_pos = test_sumpos(end);
        storeLast_neg = test_sumneg(end);
    end
    
end

% compare predicted and observed scores

[R_pos,P_pos] = corr(behav_pred_pos,all_behav);
[R_neg,P_neg] = corr(behav_pred_neg,all_behav);

end

