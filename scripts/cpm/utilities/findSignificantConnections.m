function [conn_pos, conn_neg] = findSignificantConnections(pos_mask_true, neg_mask_true, connectomeDirectory)
    % Find names of the connections that are significant (i.e., the selected network)
    
    % Get the upper triangular matrix excluding the main diagonal
    upper_tri_pos = triu(pos_mask_true, 1);
    upper_tri_neg = triu(neg_mask_true, 1);
    
    % Get linear indices of the 1s in the upper triangular matrix
    linear_indices_pos = find(upper_tri_pos);
    linear_indices_neg = find(upper_tri_neg);
    
    % Preallocate the array for subscripts
    subscripts_pos = zeros(length(linear_indices_pos), 2);
    subscripts_neg = zeros(length(linear_indices_neg), 2);
    
    % Convert these linear indices to subscripts
    [subscripts_pos(:,1), subscripts_pos(:,2)] = ind2sub(size(pos_mask_true), linear_indices_pos);
    [subscripts_neg(:,1), subscripts_neg(:,2)] = ind2sub(size(neg_mask_true), linear_indices_neg);
    
    % Load Desikan parcellation names
    fileName = 'desikanNodeNames.mat';
    filePath = fullfile(connectomeDirectory, fileName);
    load(filePath, 'desikanNodeNames');
    namesNodes = desikanNodeNames;
    
    % Positive correlation edges
    conn_pos = cell(size(subscripts_pos, 1), 2);
    for nodeNames = 1:size(subscripts_pos, 1)
        conn_pos(nodeNames,:) = {namesNodes{subscripts_pos(nodeNames,1),2}, namesNodes{subscripts_pos(nodeNames,2),2}};
    end
    
    % Negative correlation edges
    conn_neg = cell(size(subscripts_neg, 1), 2);
    for nodeNames = 1:size(subscripts_neg, 1)
        conn_neg(nodeNames,:) = {namesNodes{subscripts_neg(nodeNames,1),2}, namesNodes{subscripts_neg(nodeNames,2),2}};
    end
end