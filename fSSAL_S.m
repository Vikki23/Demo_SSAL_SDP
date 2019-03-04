function [varargout] = fSSAL_S(varargin)
%
% Semi-Supervised Learing (SSL): self-learning
% Active Learning (AL): BT - sort the BT values in ascending order
% Spatial Constraint: superpixel map
%
% [mlr_results,mpm_results,train] = fSSAL_S(img,train,test,gt_map,...
%              learning_method,algorithm_parameters,flg,SSL_sampling,mflg);
%
%
% -------------- Brief description ----------------------------------------
%
% This demo implements the algorithm introduced in [1].
%
% In summary:
%
%    1- Based on a training set and the respective labels, learn 
%    the regressors of a multinomial logistic regression (MLR) using 
%    LORSAL and predict the labels of unlabeled samples with MRF
%
%    2- Use the obtained classification map and the superpixel spatial
%    information to generate the candidate set whose samples are of high
%    confidence.
%
%    3- Based on the posterior marginals and class labels, use BT to 
%    acively select pseudo-labeled samples in candidate set.
%
%    4- Update the training set with pseudo-labeled samples.
%
%    5- Goto 1, until some stopping rule is meet
%
%
%
% -------------- Input parameters -----------------------------------------
%
% 1.img     -  hyperspectral dataset:  3 \times 1 struct
%              im:   no_bands \times no_pixels
%              size :  [no_lines,no_columns]
%
% 2.train   -  training set  :  2 \times no_train
%              the first line is the indexes
%              the second line is the labeles
%
% 3.test    -  test set  :  2 \times n_test
%              the first line is the indexes
%              the second line is the labeles
%              default = train;
%
% 4.gt_map  -  ground truth map : no_lines \times no_columns
%              the size is the same with the original image
%              including the labels of both training samples and test samples
%
% 5.learning_method  -  this variable takes values in  {linear, RBF}
%                       default = RBF.
%
% 6.algorithm_parameters  -  settings of LORSAL  :  3 \times 1 struct
%                            lambda - the Laplace parameter controlling the degree of sparsity of the
%                                     regressor components.
%                            beta   -  LORSAL parameter setting the augmented Lagrance weight  and
%                                      algorithm convergency speed (see appendix of [1]; reasonable
%                                      values: beta < lambda)
%                            mu     -  the spatial prior regularization  parameter, controlling
%                                      the degree of spatial smothness. Tune this parameter to
%                                      obtain a good segmentation results (reasobnable values [1, 4]).
%
% 7.SSL_sampling  -  settings of pseudo-labeled sampling :  3 \times 1 struct
%                    SMap      - the superpixel map
%                    us        - pseudo-labeled samples per iteration
%                    iter      - the iteration times in total
%
% 8.mflg - indicator for the BT score type (if zero, calculated by MLR, otherwise MLR-MRF)
%
% --- output parameters ---------------------------------------------------
%
% mlr_results - MLR classification results : 4 \times 1 struct,
%               map        -   classification map
%               OA         -   classification overall accuracy
%               AA         -   classification average accuracy
%               kappa      -   classification kappa statistic
%               CA         -   classification class individual accuracy
%
% mpm_results - MPM classification results : 8 \times 1 struct,
%               map        -   classification map
%               OA         -   classification overall accuracy
%               AA         -   classification average accuracy
%               kappa      -   classification kappa statistic
%               CA         -   classification class individual accuracy
%               train_size -   the size of the training set per iteration
%               sslbt      -   the biggest bt values of pseudo-labeled
%                              samples per iteration
%               PseudoAcc  -   the predicting accuracy of pseudo-labeled
%                              samples per iteration
%
% train - train samples after the iteration process : 2 \times N
%         N is the total number of training samples after iterations
%         including truly labeled and pseudo-labeled samples.
% -------------------------------------------------------------------------
%
% More details in
%
% [1] C. Liu, J. Li and L. He, Superpixel-Based Semisupervised Active 
% Learning for Hyperspectral Image Classification. IEEE Journal of Selected 
% Topics in Applied Earth Observations and Remote Sensing, vol.12, no.1, 
% pp. 357-370, Jan. 2019.
%
%  Copyright: Chenying Liu (sysuliuchy@163.com)
%             Jun Li (lijun8206@hnu.edu.cn)
%             Lin He (helin@scut.edu.cn);  
%
%  For any comments contact the authors


%% parameters
% the 1st parameter is the data set
img = varargin{1};
if ~numel(img),error('the data set is empty');end

% the 2nd parameter is the training set.
train = varargin{2}; % The first line is the indexes, the second is the labeles
if isempty(train), error('the train data set is empty, please provide the training samples');end
no_classes = max(train(2,:));

% the 3rd parameter is the test
if nargin >2
    test = varargin{3};
else
    fprintf('the test set is empty, thus we use the training set as the validation set \n')
    test = train;
end

% the 4th parameter is the ground truth map
if nargin >3
    gt = varargin{4};
else
    fprintf('the ground truth map is empty \n')
    gt = [];
end

% the 5th parameter is the learning_method (RBF/linear)
if nargin >4, learning_method = varargin{5}; else learning_method = 'RBF';end
if isempty(learning_method), learning_method = 'RBF';end

% the 6th parameter is setting of LORSAL
if nargin >5
    algorithm_parameters = varargin{6};
else
    algorithm_parameters.lambda=0.001;
    algorithm_parameters.beta = 0.5*algorithm_parameters.lambda;
    algorithm_parameters.mu = 2;
end
if isempty(algorithm_parameters)
    algorithm_parameters.lambda=0.001;
    algorithm_parameters.beta = 0.5*algorithm_parameters.lambda;
    algorithm_parameters.mu = 2;
end

% the 7th parameter is the setting of SSL
if nargin >6, SSL_sampling = varargin{7}; else SSL_sampling = [];end
if isempty(SSL_sampling), tot_sim = 1; else tot_sim = SSL_sampling.iter+1; end

% the 8th parameter is the MRF indicator for the probabilities
if nargin >7, mflg = varargin{8};else mflg = [];end
if isempty(mflg), mflg = 0;end

%% compute the neighborhood from the grid, in this paper we use the first
% order neighborhood
[~, nList] = getNeighFromGrid(img.size(1),img.size(2));

%% SSAL-S
train_real = [train;SSL_sampling.SMap(train(1,:))];
tsize_real = size(train_real,2);

% start active section iterations
for iter = 1:tot_sim
    % train set in each iteration
    mpm_results.train_size(iter) = size(train,2);
    trainset = img.im(:,train(1,:));
    
    if strcmp(learning_method,'RBF')
        sigma = 0.8;
        % build |x_i-x_j| matrix
        nx = sum(trainset.^2);
        [X,Y] = meshgrid(nx);
        dist=X+Y-2*trainset'*trainset;
        clear X Y
        scale = mean(dist(:));
        % build design matrix (kernel)
        K=exp(-dist/2/scale/sigma^2);
        clear dist
        % set first line to one
        K = [ones(1,size(trainset,2)); K];
    else
        K =[ones(1,size(trainset,2)); trainset];
        scale = 0;
        sigma = 0;
    end
    
    learning_output = struct('scale',scale,'sigma',sigma);
    
    % learn the regressors
    [w,L] = LORSAL(K,train(2,:),algorithm_parameters.lambda,algorithm_parameters.beta);
    
    % compute the MLR probabilites
    p = mlr_probabilities(img.im,trainset,w,learning_output);
    
    % compute the classification results
    [maxp, mlr_results.map(iter,:)] = max(p);  % map:1*21025
    [mlr_results.OA(iter),mlr_results.kappa(iter),mlr_results.AA(iter),mlr_results.CA(iter,:)]=...
        calcError(test(2,:)-1, mlr_results.map(iter,test(1,:))-1,1:no_classes);
    
    %%%%%%% MLR-MLL_LBP
    v0 = exp(algorithm_parameters.mu);
    v1 = exp(0);
    psi = v1*ones(no_classes,no_classes);
    for i = 1:no_classes
        psi(i,i) = v0;
    end
    
    psi_temp = sum(psi);
    psi_temp = repmat(psi_temp,no_classes,1);
    psi = psi./psi_temp;
    p =p';
    
    % belief propagation
    [belief] = BP_message(p,psi,nList,train);
    [maxb,mpm_results.map(iter,:)] = max(belief);
    
    [mpm_results.OA(iter),mpm_results.kappa(iter),mpm_results.AA(iter),...
        mpm_results.CA(iter,:)]= calcError(test(2,:)-1, mpm_results.map(iter,test(1,:))-1,[1:no_classes]);
    
    %%%%%% select pseudo-labeled samples
    
    %% SSL find pseudo-labeled pixels
    
    % obtain BTMap
    if mflg, pactive2 = belief; else pactive2 = p'; end
    pactive2 = sort(pactive2,'descend');
    BTMap = pactive2(1,:)-pactive2(2,:);
    
    % gather the confident within-superpixel samples/candidate set
    SSLSet = []; % the confident superpixel set
    % find the within-superpixel samples of a given truly labeled sample
    for i = 1:tsize_real
        % test if the given sample is located in a selected superpixel
        if i ~=1
            if ismember(train_real(3,i),train_real(3,1:i-1))
                spind = (train_real(3,1:i-1) == train_real(3,i));
                if ismember(train_real(2,i),train_real(2,spind))
                    continue;
                end
            end
        end
        IndInSameSuper = find(SSL_sampling.SMap == train_real(3,i));
        
        % pick confident superpixel samples out (the ones with the same labels)
        IndSameLabInSuper = (mpm_results.map(iter,IndInSameSuper) == train_real(2,i));
        subSSLSet = IndInSameSuper(IndSameLabInSuper);
        
        % remove the superpixel samples which have been already in the training set
        delInd = (ismember(subSSLSet,train(1,:))==1);
        subSSLSet(delInd) = [];
        
        % transform to a row vector
        subSSLSet = subSSLSet';
        
        % refresh the confident superpixel pool
        SSLSet = [SSLSet,[subSSLSet;ones(1,length(subSSLSet))*train_real(2,i)]];
    end
    
    % judge whether the SSLSet is empty
    % if so, jump out of the loop
    if isempty(SSLSet); break; end
    
    % select pseudo-labeled samples from the SSLSet
    SSLSetBT = BTMap(SSLSet(1,:));  % record all the corresponding BT values
    [SSLSetBT_sort,SSLSetBT_sortind] = sort(SSLSetBT,'ascend'); %resort the BT values in descending order
    SSLSet_sort = SSLSet(:,SSLSetBT_sortind); % resort the samples in SSLSet according to their BT values
    if size(SSLSet_sort,2) < SSL_sampling.us
        train = [train SSLSet_sort];
        mpm_results.sslbt(iter) = SSLSetBT_sort(end);
    else
        train = [train SSLSet_sort(:,1:SSL_sampling.us)];
        mpm_results.sslbt(iter) = SSLSetBT_sort(SSL_sampling.us);
    end
    % end of SSL finding pseudo-labeled pixels
    
    % calculate the accuracy of added pseudo-labeled samples
    PseudoTestInd = find(ismember(train(1,tsize_real+1:end),test(1,:))==1);
    PseudoTestInd = PseudoTestInd + tsize_real;
    PseudoTestValue = train(2,PseudoTestInd)-gt(train(1,PseudoTestInd));
    mpm_results.PseudoAcc(iter) = length(find(PseudoTestValue==0))/length(PseudoTestValue);
    
end

%% output
varargout(1) = {mlr_results}; % train_size OA AA kappa CA BT map
varargout(2) = {mpm_results};
if nargout == 3
    varargout(3) = {train};
end

%%
return

% %-----------------------------------------------------------------------%