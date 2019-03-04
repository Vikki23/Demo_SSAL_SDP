function [varargout] = fSSAL_SDP(varargin)
%
% Semi-Supervised Learing (SSL): self-learning
% Active Learning (AL): BT - sort the BT values in ascending order
% Spatial Constraint: superpixel map along with DP augmentation
%
% [mlr_results,mpm_results,train] = fSSAL_SDP(img,train,test,gt_map,...
%              learning_method,algorithm_parameters,flg,SSL_sampling,
%              mode_sampling,mflg);
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
%    information along with the DP augmentation strategy to generate the
%    candidate set whose samples are of high confidence.
%
%    3- Based on the posterior marginals and class labels, use BT to
%    acively select pseudo-labeled samples in candidate set.
%
%    4- Update the training set with pseudo-labeled samples.
%
%    5- Goto 1, until some stopping rule is meet
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
% 8.mode_sampling  -  settings of mode sampling :  2 \times 1 struct
%                     um       - the number of modes selected each time
%                     startBT  - the BT score for triggering the mode
%                                augmentation strategy
%
% 9.mflg - indicator for the BT score type (if zero, calculated by MLR, otherwise MLR-MRF)
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
%
% mode_final - the selected mode set in the DP augmentation : 2 \times N
%              N is the total number of selected modes.
%              The 1st, 2nd, 3rd and 4th rows are the locations, labels,
%              superpixel ids, and the confidences, respectively.
%
% -------------------------------------------------------------------------
%
% More details in
%
% [1] C. Liu, J. Li and L. He, Superpixel-Based Semisupervised Active 
% Learning for Hyperspectral Image Classification. IEEE Journal of Selected 
% Topics in Applied Earth Observations and Remote Sensing, vol.12, no.1, 
% pp. 357-370, Jan. 2019.
%
% Copyright: Chenying Liu (sysuliuchy@163.com)
%             Jun Li (lijun8206@hnu.edu.cn)
%             Lin He (helin@scut.edu.cn);  
%
% For any comments contact the authors


%% parameters
% 1st parameter is the data set
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
if nargin >4,learning_method = varargin{5};else learning_method = 'RBF';end
if isempty(learning_method),learning_method = 'RBF';end 

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
if nargin >6, SSL_sampling = varargin{7};else SSL_sampling = [];end
if isempty(SSL_sampling), tot_sim = 1;else tot_sim = SSL_sampling.iter+1;end

% the 8th parameter is the setting of mode sampling
if nargin >7, mode_sampling = varargin{8}; else mode_sampling = [];end

% the 9th parameter is the MRF indicator for the probabilities
if nargin >8, mflg = varargin{9};else mflg = [];end
if isempty(mflg), mflg = 0;end

%% compute the neighborhood from the grid, in this paper we use the first
% order neighborhood
[~, nList] = getNeighFromGrid(img.size(1),img.size(2)); 

%% SSAL-SDP2
train_real = [train;SSL_sampling.SMap(train(1,:))];
tsize_real = size(train_real,2);
train(3,:) = SSL_sampling.SMap(train(1,:));

% seek the modes in each superpixel using density peaks finding clustering
mode_all = 1:max(SSL_sampling.SMap(:)); % the first row are S-ids, the second are corresponding mode locations
for nsuperImg = 1:max(SSL_sampling.SMap(:)) %find the samples in the same superpixel
    samInASuperP_ind = find(SSL_sampling.SMap == nsuperImg);
    samInASuperP = img.im(:,samInASuperP_ind);
    % calculate the representation abilities of samples (density * distance)
    DPMap = fGenerate_DPMap(samInASuperP',0.9);
    [maxDP,mode0] = max(DPMap);
    mode_all(2,nsuperImg) = samInASuperP_ind(mode0);
%     mode_feat(:,nsuperImg) = mean(samInASuperP,2);
end
% record the locations of all the modes
mode_loc = [];
[mode_loc(1,:),mode_loc(2,:)] = ind2sub(img.size,mode_all(2,:));
% delete the modes of labeled superpixels
trueLabelMInd = ismember(mode_all(1,:),train_real(3,:))==1;
mode_tlabel(1,:) = mode_all(2,trueLabelMInd);
mode_tlabel(3,:) = mode_all(1,trueLabelMInd);
for i = 1:sum(trueLabelMInd)
    ind_temp = train_real(3,:)==mode_tlabel(3,i);
    label_temp = train_real(2,ind_temp);
    mode_tlabel(2,i) = label_temp(1);
end
mode_all(:,trueLabelMInd) = [];
mode_all0 = mode_all;
mode_final = []; % selected mode set

% start active section iterations
startpoint = tot_sim;
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
    [maxp, mlr_results.map(iter,:)] = max(p);
    [mlr_results.OA(iter),mlr_results.kappa(iter),mlr_results.AA(iter),mlr_results.CA(iter,:)]=...
        calcError(test(2,:)-1, mlr_results.map(iter,test(1,:))-1,1:no_classes);

    %%%%%%% MRF
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
    
    [mpm_results.OA(iter),mpm_results.kappa(iter),mpm_results.AA(iter),mpm_results.CA(iter,:)]=...
        calcError(test(2,:)-1, mpm_results.map(iter,test(1,:))-1,[1:no_classes]);
    
    %%%%%% select pseudo-labeled samples
    
    % after watershed, check the training set first
    % delete the unconfident selected modes and the corresponding pseudo-labeled samples
    % the unconfident selected modes are those whose labels are not consistent with the last iteration
    if iter>startpoint
        mode_test = mode_final(2,:) - mpm_results.map(iter,mode_final(1,:));
        mode_delInd = find(mode_test~=0);
        if ~isempty(mode_delInd)
            trainDelMode = ismember(train(3,tsize_real+1:end),mode_final(3,mode_delInd));
            delInd = ([zeros(1,tsize_real) trainDelMode]==1);
            train(:,delInd) = [];
            mode_final(:,mode_delInd) = [];
        end
    end
    
    % obtain BTMap
    if mflg, pactive2 = belief; else pactive2 = p'; end 
    pactive2 = sort(pactive2,'descend');
    BTMap = pactive2(1,:)-pactive2(2,:);
    
    % gather the confident within-superpixel samples/candidate set
    SSLSet = [];
    % find the confident within-superpixel samples of a given truly labeled sample 
    for i = 1:size(train_real,2)
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
        
        % record all the corresponding BT values
        BTSameLabInSuper = BTMap(subSSLSet);  
        
        % refresh the confident superpixel pool
        SSLSet = [SSLSet,[subSSLSet;ones(1,length(subSSLSet))*train_real(2,i);SSL_sampling.SMap(subSSLSet);BTSameLabInSuper]];
    end
    % end of finding the confident samples in labeled superpixels
    
    % find the confident within-superpixel samples of a selected mode (in unlabeled superpixels)
    for i = 1:size(mode_final,2)
		IndInSameSuper = find(SSL_sampling.SMap == mode_final(3,i));  
        
        % pick confident superpixel samples out (the ones with the same labels)
        IndSameLabInSuper = (mpm_results.map(iter,IndInSameSuper) == mode_final(2,i)); 
        subSSLSet = IndInSameSuper(IndSameLabInSuper);
        
        % remove the superpixel samples which have been already in the training set
        delInd = (ismember(subSSLSet,train(1,:))==1); 
		subSSLSet(delInd) = [];
        
        % transform to a row vector
        subSSLSet = subSSLSet'; 
        
        % record all the corresponding BT values
        BTSameLabInSuper = BTMap(subSSLSet);  
        
        % refresh the confident superpixel pool
        SSLSet = [SSLSet,[subSSLSet;ones(1,length(subSSLSet))*mode_final(2,i);SSL_sampling.SMap(subSSLSet);BTSameLabInSuper]];
    end
    % end of finding the confident samples in unlabeled superpixels
    
    % delete the modes from SSLSet
    if ~isempty(mode_final)
        delSSLSetInd = (ismember(SSLSet(1,:),mode_final(1,:))==1);
        SSLSet(:,delSSLSetInd) = [];
    end
    
    % select pseudo-labeled samples from the SSLSet
    % resort the BT values in ascending order
    [BTInSSLSet_sort,BTInSSLSet_sortind] = sort(SSLSet(4,:),'ascend'); 
    if size(SSLSet,2)>0 && size(SSLSet,2) < SSL_sampling.us
        train = [train,SSLSet(1:3,BTInSSLSet_sortind)];
        mpm_results.sslbt(iter) = BTInSSLSet_sort(end);
    elseif size(SSLSet,2) >= SSL_sampling.us
        train = [train,SSLSet(1:3,BTInSSLSet_sortind(1:SSL_sampling.us))];
        mpm_results.sslbt(iter) = BTInSSLSet_sort(SSL_sampling.us);
    end
    % end of SSL finding pseudo-labeled pixels
    
    % judge whether the watershred requirement is met
    if startpoint == tot_sim
        if mpm_results.sslbt(iter)>mode_sampling.startBT
            startpoint = iter;
        end
    end
    
    % mode validating and sampling
    % validating: remove the unconfident modes in cnadidate mode set
    % sampling: select unlabeled superpixels for training after the watershred
    if ~isempty(mode_all)
        % validating
        % validating 1: delete the modes whose predicted labels are not consistent with the last iteration
        if size(mode_all,1)<3
            % if the candidate set has just been updated, record the predicted labels of modes
            mode_all(3,:) = mpm_results.map(iter,mode_all(2,:));
        else
            % delete the modes whose labels are different from their former ones
            mode_all(4,:) = mpm_results.map(iter,mode_all(2,:));
            a = mode_all(3,:)-mode_all(4,:);
            delInd = (a~=0);
            mode_all(:,delInd) = [];
            mode_all(3,:) = mode_all(4,:);
            mode_all(4,:) = [];
        end

        % validating 2: delete the modes whose predicted labels are not the same with most samples in a superpixel
        delInd = [];
        for smode = 1:size(mode_all,2)
            samInASuperP_ind = (SSL_sampling.SMap == mode_all(1,smode)); % the first row of mode_all is the superpixel id
            labInASuperP = mpm_results.map(iter,samInASuperP_ind);
            % count the number of samples in each class
            m=hist(labInASuperP,[1:16]);
            % find the class with the largest number of samples
            [mm,mind] = max(m);
            if mind ~= mode_all(3,smode)
                delInd = [delInd smode];
            end
        end
        mode_all(:,delInd) = [];
        % end of validating
        
        % record the confidences of modes
        mode_all(4,:) = pactive2(1,mode_all(2,:));  

        % sampling: add new modes
        if iter>=startpoint
            % record all the labeled superpixels (both truly and pseudo-labeled)
            if ~isempty(mode_final)
                mode_label = [mode_tlabel,mode_final(1:3,:)];
            else
                mode_label = mode_tlabel;
            end
            
            % claculate the distances between candidates and labeled superpixels
            dis_matr = fCal_Feature_EuclDis(mode_loc,mode_all(1,:),mode_label(3,:));
            min_dis = min(dis_matr);
            clear dis_matr
            [~,mode_sortInd] = sort(min_dis,'descend');

            % rearrange the mode set as the form of training set
            mode_final0(1:2,:) = mode_all(2:3,mode_sortInd); % first row-location; second row-labels
            mode_final0(3,:) = mode_all(1,mode_sortInd); % thrid row: superpixel id
            mode_final0(4,:) = mode_all(4,mode_sortInd); % forth row: confidence
            if size(mode_final0,2) < mode_sampling.um
                mode_final = [mode_final,mode_final0];
                mode_all = [];
            else
                mode_final = [mode_final,mode_final0(:,1:mode_sampling.um)];
                mode_all(:,mode_sortInd(1:mode_sampling.um)) = [];
            end
            clear mode_final0
        end
        % end of sampling
    end
    % end of mode validating and sampling
    
    % refresh the candidate mode set when it is empty or the samples in it are all not confidentenough
    if isempty(mode_all) || max(mode_all(4,:))<0.8
        mode_all = mode_all0;
        mode_delInd = ismember(mode_all(1,:),mode_final(3,:))==1;
        mode_all(:,mode_delInd) = [];
    end
    
    % calculate the accuracy of added pseudo-labeled samples
    PseudoTestInd = find(ismember(train(1,tsize_real+1:end),test(1,:))==1);
    PseudoTestInd = PseudoTestInd + tsize_real;
    PseudoTestValue = train(2,PseudoTestInd)-gt(train(1,PseudoTestInd));
    mpm_results.PseudoAcc(iter) = length(find(PseudoTestValue==0))/length(PseudoTestValue);
    
end

%% output
mpm_results.startpoint = startpoint;
varargout(1) = {mlr_results};
varargout(2) = {mpm_results};
if nargout >2 
    varargout(3) = {train};
end

if nargout > 3
    varargout(4) = {mode_final};
end

%%
return

% %-----------------------------------------------------------------------%