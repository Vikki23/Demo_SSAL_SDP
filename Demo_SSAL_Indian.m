clear all
close all
clc

%% parameter settings
n_class = 5;              % the initial number of training samples per class
TotIte = 20;              % the total number of iterations
us = 15;                  % the number of pseudo-labeled samples added in each iteration
um = 5;                   % the number of new added modes in each iteration for SSAL-SDP
startBT = 0.5;            % the BT scores for triggering the DP augmentation for SSAL-SDP
mflg = 1;
SSAL_type = 'N';

%% load data 
% load image
load data_Indian_pines;
% image size
[no_lines,no_columns,no_bands] = size(data); 
img = ToVector(data);
img = img';
clear data

% load superpixel map (Nc = 400)
load AVI_SLIC_Superpixel400 
superImg = labels';
clear labels

% load ground truth
load gt_Indian_16class
gt_map = zeros(no_lines,no_columns);
gt_map(trainall(:,1)) = trainall(:,2);
trainall = trainall';
no_class = max(trainall(2,:));    % number of classes

%% randomly select the initial training samples 
indexes             = train_test_random_equal_number(trainall(2,:),n_class,no_class*n_class);
train_set           = trainall(:,indexes);
test_set            = trainall;
test_set(:,indexes) = [];  
train_samples       = img(:,train_set(1,:));
train_label         = train_set(2,:);
test_label          = test_set(2,:);

%% SSAL 
% arrange the image struct
img = struct('im',img,'size',[no_lines no_columns]);%,'mycolormap',mycmap

if strcmp(SSAL_type,'N')
    SSL_sampling = struct('us',us,'iter',TotIte); % arrange the SSL_sampling struct
    disp('Start looping for SSAL-N...')
    [mlr_results,mpm_results] = fSSAL_N(img,train_set,test_set,gt_map,[],[],SSL_sampling,mflg); 
elseif strcmp(SSAL_type,'S')
    SSL_sampling = struct('SMap',superImg,'us',us,'iter',TotIte);
    disp('Start looping for SSAL-S...')
    [mlr_results,mpm_results] = fSSAL_S(img,train_set,test_set,gt_map,[],[],SSL_sampling,mflg); 
elseif strcmp(SSAL_type,'SN')
    SSL_sampling = struct('SMap',superImg,'us',us,'iter',TotIte);
    disp('Start looping for SSAL-SN...')
    [mlr_results,mpm_results] = fSSAL_SN(img,train_set,test_set,gt_map,[],[],SSL_sampling,mflg); 
elseif strcmp(SSAL_type,'SDP')
    SSL_sampling = struct('SMap',superImg,'us',us,'iter',TotIte);
    mode_sampling = struct('um',um,'startBT',startBT); % arrange the mode_sampling struct
    disp('Start looping for SSAL-SDP...')
    [mlr_results,mpm_results] = fSSAL_SDP(img,train_set,test_set,gt_map,[],[],SSL_sampling,mode_sampling,mflg); 
else
    error('Wrong SSAL method type!');
end
disp('End of looping.')
    
%% classification results
% mlr results
OA_mlrAccuracy = mlr_results.OA;
AA_mlrAccuracy = mlr_results.AA;
CA_mlrAccuracy = mlr_results.CA;
mlrkappa = mlr_results.kappa;
mlrMap = mlr_results.map;
% mpm results
OA_mpmAccuracy = mpm_results.OA;
AA_mpmAccuracy = mpm_results.AA;
CA_mpmAccuracy = mpm_results.CA;
mpmkappa = mpm_results.kappa;
mpmMap = mpm_results.map;

clearvars -except U u us n_class TotIte...
    OA_mlrAccuracy AA_mlrAccuracy mlrkappa CA_mlrAccuracy mlrMap...
    OA_mpmAccuracy AA_mpmAccuracy mpmkappa CA_mpmAccuracy mpmMap...
    train_size sslbt PseudoAcc TrainSamples
        