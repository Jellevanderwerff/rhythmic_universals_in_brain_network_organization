function [pos_averageValues, neg_averageValues] = createWeightedMatrix(all_mats, pos_mask, neg_mask)
    % Create a weighted matrix for nodes that are significant in the analysis.
    % Values in this matrix are the average across subjects.
    
    % Input:
    %   all_mats: 3D matrix of connectivity matrices (nodes x nodes x subjects)
    %   pos_mask: Binary mask for positive network
    %   neg_mask: Binary mask for negative network
    
    % Output:
    %   pos_averageValues: Weighted matrix for positive network
    %   neg_averageValues: Weighted matrix for negative network
    
    % Get the number of subjects
    no_subj = size(all_mats, 3);
    
    % Create weighted matrix for positive network
    pos_mask_true = pos_mask;
    pos_maskedPositions = all_mats .* repmat(pos_mask_true, [1, 1, no_subj]);
    pos_averageValues = mean(pos_maskedPositions, 3);
    
    % Create weighted matrix for negative network
    neg_mask_true = neg_mask;
    neg_maskedPositions = all_mats .* repmat(neg_mask_true, [1, 1, no_subj]);
    neg_averageValues = mean(neg_maskedPositions, 3);
end