function [results] = mil_kNN( data, par )

% bags: name inst_name inst_label instace label
X = data.X;
y = data.y;


bag_dist_type = par.bag_dist_type;      % 'max' or 'min'
inst_dist_type = par.inst_dist_type;    % 'euclidean'  or 'cosine'
num_ref = par.num_neighbor;             % number of neighbors
rank_citer = par.rank_citer;            % the rank to cite
