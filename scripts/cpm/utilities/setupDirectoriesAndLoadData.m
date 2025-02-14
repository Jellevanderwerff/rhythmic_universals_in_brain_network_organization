function [connectomeDirectory, behaviourDirectory, scriptsDirectory, all_mats, all_behav,data] = setupDirectoriesAndLoadData(modality, condition, mainmetrics)
    % Get the current directory and navigate to the base directory
    currentDirectory = pwd;
    [mainscriptDirectory, ~, ~] = fileparts(currentDirectory);
    [baseDirectory, ~, ~] = fileparts(mainscriptDirectory);


    % Set up directory paths
    connectomeDirectory = fullfile(baseDirectory, 'data', 'CPM', 'connectomes');
    behaviourDirectory = fullfile(baseDirectory, 'data', 'CPM', 'behaviour');
    scriptsDirectory = fullfile(baseDirectory, 'scripts', 'utilities');

    % Load connectomes
    if strcmp(modality, 'functional')
        fileName = 'desikanFconnectome47.mat';
        connPath = fullfile(connectomeDirectory, fileName);
        load(connPath)
        all_mats = Z;  % Assuming 'Z' is loaded from the file
    elseif strcmp(modality, 'structural_FBC')
        fileName = 'desikan_structural_cortical.mat';
        connPath = fullfile(connectomeDirectory, fileName);
        load(connPath)
        all_mats = connectomes;  % Assuming 'connectomes' is loaded from the file
    else
        error('Invalid modality specified')
    end

    % Load behavioural data
    filePath = fullfile(behaviourDirectory, ['pp_measures_', condition, '.csv']);
    data = readtable(filePath);
    all_behav = data.(mainmetrics);
end