function [results] = mil_mi_SVM( data, par )
% inst_MI_SVM adopted from MILL
%Input:
% data.train_bags
% data.test_bags
% each bag is a struct (5x1): name inst_name inst_label instance label

% bags: name inst_name inst_label instace label
train_bags = data.train_bags;
test_bags = data.test_bags;


num_train_bag = length(train_bags);
num_test_bag = length(test_bags);

%set the initial instance labels to bag labels
idx = 0;
for i=1:num_train_bag
    num_inst = size(train_bags(i).instance, 1);
    train_label(idx+1 : idx+num_inst) = repmat(train_bags(i).label, num_inst, 1);
    idx = idx + num_inst;
end

[train_instance, dummy] = bag2instance(train_bags);
[test_instance, dummy] = bag2instance(test_bags);

num_train_inst = size(train_instance, 1);
num_test_inst = size(test_instance, 1);

step = 1;
past_train_label(step,:) = train_label;

while 1
    %num_pos_label = sum(train_label == 1);
    %num_neg_label = sum(train_label == 0);
    %new_para = sprintf(' -NegativeWeight %.10g', (num_pos_label / num_neg_label));
    
    [all_label_predict, all_prob_predict] = lib_SVM(par, train_instance, train_label, [train_instance; test_instance], ones(num_train_inst+num_test_inst, 1));
    train_label_predict = all_label_predict(1 : num_train_inst);
    train_prob_predict = all_prob_predict(1 : num_train_inst);
    test_label_predict = all_label_predict(num_train_inst+1 : num_train_inst+ num_test_inst);
    test_prob_predict = all_prob_predict(num_train_inst+1 : num_train_inst+ num_test_inst);
    
    idx = 0;
    for i=1:num_train_bag
        num_inst = size(train_bags(i).instance, 1);
        
        if train_bags(i).label == 0
            train_label(idx+1 : idx+num_inst) = zeros(num_inst, 1);
        else
            if any(train_label_predict(idx+1 : idx+num_inst))
                train_label(idx+1 : idx+num_inst) = train_label_predict(idx+1 : idx+num_inst);
            else
                [sort_prob, sort_idx] = sort(train_prob_predict(idx+1 : idx+num_inst));
                train_label(idx+1 : idx+num_inst) = zeros(num_inst, 1);
                train_label(idx + sort_idx(num_inst)) = 1;
            end
        end
        idx = idx + num_inst;
    end
    
    difference = sum(past_train_label(step,:) ~= train_label);
    fprintf('Number of label changes is %d\n', difference);
    if difference == 0, break; end;
    
    repeat_label = 0;
    for i = 1 : step
        if all(train_label == past_train_label(i, :))
            repeat_label = 1;
            break;
        end
    end
    
    if repeat_label == 1
        fprintf('Repeated training labels found, quit...\n');
        break;
    end
    
    step = step + 1;
    past_train_label(step, :) = train_label;
    
end

%prediction
test_inst_label = test_label_predict;
test_inst_prob = test_prob_predict;

idx = 0;
test_bag_label = zeros(num_test_bag, 1);
test_bag_prob = zeros(num_test_bag, 1);
for i=1:num_test_bag
    num_inst = size(test_bags(i).instance, 1);
    test_bag_label(i) = any(test_inst_label(idx+1 : idx+num_inst));
    test_bag_prob(i) = max(test_inst_prob(idx+1 : idx+num_inst));
    idx = idx + num_inst;
end
results.test_inst_label = test_label_predict;
results.test_inst_prob = test_prob_predict;
results.test_bag_label = test_bag_label;
results.test_bag_prob = test_bag_prob;

results.BagAcc = MIL_Bag_Evaluate(test_bags, test_bag_label);
results.InstAccu = MIL_Inst_Evaluate(test_bags, test_inst_label);