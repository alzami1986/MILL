clear
Shuffled = 1;
Normalization = 1;
insts = load( 'example.data.matrix' );

fid = fopen( 'example.data.label', 'r' );

nbag = 0;
prev_bag_name = '';
ninst = 0;
idx = 0;
while feof(fid) == 0

    line = strtrim(fgets(fid));
    elems = strsplit(' ',line);    %instance_name, bag_name, label

    bag_name = cell2mat(elems(2));
    if strcmp(bag_name, prev_bag_name) == 0     %change of bag
        if (nbag >= 1)
            bags(nbag).instance = insts(idx + 1:idx + ninst, :);
            bags(nbag).label = any(bags(nbag).inst_label);
            idx = idx + ninst;
        end;
        nbag = nbag + 1;
        bags(nbag).name = bag_name;
        prev_bag_name = bag_name;
        ninst = 0;
    end

    ninst = ninst + 1;
    bags(nbag).inst_name(ninst) = elems(1);
    label = cell2mat(elems(3));
    bags(nbag).inst_label(ninst) = strcmp(label,'1');   %the positive label must be set to 1
end;

if (nbag >= 1)
    bags(nbag).instance = insts(idx + 1:idx + ninst,:);
    bags(nbag).label = any(bags(nbag).inst_label);
end;
fclose(fid);

num_data = length(bags);
num_feature = size(bags(1).instance, 2);

% normalize the data set
if (Normalization == 1) 
    bags = MIL_Scale(bags);
end;

% randomize the data
rand('state',sum(100*clock));
if (Shuffled == 1) %Shuffle the datasets
    Vec_rand = rand(num_data, 1);
    [B, Index] = sort(Vec_rand);
    bags = bags(Index);
end;

split = fix( num_data/3 );
testindex  = 1:split;
trainindex = split + 1:num_data;

data.train_bags = bags(trainindex);
data.test_bags = bags(testindex);

% par.bag_dist_type = 'min';      % 'max' or 'min'
% par.inst_dist_type = 'euclidean';    % 'euclidean'  or 'cosine'
% par.num_neighbor = 3;             % number of neighbors
% par.rank_citer = 5; 
% 
% [results] = mil_kNN( data, par );

par.Kernel = 2;
par.KernelParam = 1.0;
par.wi_weight = 1.0;
par.rootDir = pwd;

[results] = mil_mi_SVM( data, par );