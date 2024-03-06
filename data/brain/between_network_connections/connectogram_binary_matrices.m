% Initialize a 7x7 matrix of zeros
connectivityMatrix = zeros(7,7);

%% Between-network connectivity for entropy
% Set the correlations based on the provided connections
% FPN (4) and DAN (3)
connectivityMatrix(4,3) = 1;
connectivityMatrix(3,4) = 1;

% DAN (3) and DMN (7)
connectivityMatrix(3,7) = 1;
connectivityMatrix(7,3) = 1;

% DMN (7) and LN (5)
connectivityMatrix(7,5) = 1;
connectivityMatrix(5,7) = 1;

save('entropy.functional.between_net.mat','connectivityMatrix')
clear connectivityMatrix
%% Between-network connectivity for g-response
% Initialize a 7x7 matrix of zeros
connectivityMatrix = zeros(7,7);

% VAN (6) and SMN (2)
connectivityMatrix(6,2) = 1;
connectivityMatrix(2,6) = 1;

% FPN (4) and SMN (2)
connectivityMatrix(4,2) = 1;
connectivityMatrix(2,4) = 1;

save('g-resp.functional.between_net.mat','connectivityMatrix')
clear connectivityMatrix

%% Between-network connectivity for binary_or_ternary.functional
% Initialize a 7x7 matrix of zeros
connectivityMatrix = zeros(7,7);

% Set the correlations for the specified connections
% VN (1) and FPN (4)
connectivityMatrix(1,4) = 1;
connectivityMatrix(4,1) = 1;

% VN (1) and VAN (6)
connectivityMatrix(1,6) = 1;
connectivityMatrix(6,1) = 1;

% VN (1) and DAN (3)
connectivityMatrix(1,3) = 1;
connectivityMatrix(3,1) = 1;

% VN (1) and SMN (2)
connectivityMatrix(1,2) = 1;
connectivityMatrix(2,1) = 1;

save('binary_or_ternary.functional.between_net.mat','connectivityMatrix')
clear connectivityMatrix

%% Between-network connectivity for binary_or_ternary.structural
% Initialize a 7x7 matrix of zeros
connectivityMatrix = zeros(7,7);

% FPN (4) to VN (1)
connectivityMatrix(4,1) = 1;
connectivityMatrix(1,4) = 1;

% DAN (3) to VN (1)
connectivityMatrix(3,1) = 1;
connectivityMatrix(1,3) = 1;

% SMN (2) to VN (1)
connectivityMatrix(2,1) = 1;
connectivityMatrix(1,2) = 1;

% FPN (4) to LN (5)
connectivityMatrix(4,5) = 1;
connectivityMatrix(5,4) = 1;

save('binary_or_ternary.structural.between_net.mat','connectivityMatrix')
clear connectivityMatrix