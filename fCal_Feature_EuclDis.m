function dis_matr = fCal_Feature_EuclDis(mode_feat,cand_SID,label_SID)
% % calculate the Euclidean distance matrix between candidate modes and
% labeled modes
cand_feat = mode_feat(:,cand_SID);
label_feat = mode_feat(:,label_SID);
n_candS = size(cand_feat,2);
n_labelS = size(label_feat,2);
dis_matr = zeros(n_labelS,n_candS);
for feat_id = 1:size(mode_feat,1)
    candF = repmat(cand_feat(feat_id,:),n_labelS,1);
    labelF = repmat(label_feat(feat_id,:)',1,n_candS);
    dis_matr = dis_matr+(candF-labelF).^2;
end
dis_matr = sqrt(dis_matr);

end