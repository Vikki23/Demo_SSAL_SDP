function [gamma] = fGenerate_DPMap(img,cdp)
% generate the density peak map
%
% input:
%       img(n samples \times m dimensions) ------ data
%       cdp                                ------ cut off rate


nSamples = size(img,1);

% calculate the Euclidean distence matrix
Dist = pdist2(img,img,'euclidean');

% calculate the cutoff distance
% sort the distances in ascending order within each row
Dist_sort = sort(Dist,2); 
% the first colume are zeros (self-distance)
Dist_sort(:,1) = []; 
% calculate the cutoff distance
dc = mean(Dist_sort(:,floor(nSamples*cdp))); 
clear Dist_sort

% calculate the local density vector rho
rho_esti = ceil((dc-Dist)/max(Dist(:)));
rho = sum(rho_esti,2)-1; % -1:exclude the self-distance

% calculate the minimum distance vector delta
for i = 1:nSamples
    % find the indexes of the samples with higher density than the i-th one
    hdenInd = find(rho>rho(i)); 
    if ~isempty(hdenInd)
        hdenDist = Dist(i,hdenInd);
        delta(i,1) = min(hdenDist);
    else  % i is the point of highest density
        delta(i,1) = max(Dist(i,:));
    end
end

gamma = rho.*delta;

