function  [Y_compute, Y_prob] = lib_SVM(par, X_train, Y_train, X_test, Y_test)

num_class = 2;

switch par.Kernel
    case 0
      s = '';      
    case 1
      s = sprintf('-d %.10g -g 1', par.KernelParam);
    case 2
      s = sprintf('-g %.10g', par.KernelParam);
    case 3
      s = sprintf('-r %.10g', par.KernelParam); 
    case 4
      s = sprintf('-u "%s"', par.KernelParam);
end
root = par.rootDir;
temp_dir = sprintf('%s/temp', root);
if (~exist(temp_dir))
    s = mkdir(root, 'temp');
    if (s ~= 1)
        error('Cannot create temp directory!');
    end
end

temp_train_file = sprintf('%s/temp.train.txt', temp_dir);
temp_test_file = sprintf('%s/temp.test.txt', temp_dir);
temp_output_file = sprintf('%s/temp.output.txt', temp_dir);
temp_model_file = sprintf('%s/temp.model.txt', temp_dir);

        
% set up the commands
% train_cmd = sprintf('! ../svm/svmtrain -b 1 -s 0 -t %d %s -c %f -w1 1 -w0 %f %s %s > log1', p(1), s, p(3), p(4), temp_train_file, temp_model_file);
% test_cmd = sprintf('! ../svm/svmpredict -b 1 %s %s %s > log1', temp_test_file, temp_model_file, temp_output_file);
train_cmd = sprintf('!svm-train -b 1 -s 0 -t %d %s -c %f -w1 1 -w0 %f %s %s > log1', par.Kernel, s, par.KernelParam, par.wi_weight, temp_train_file, temp_model_file);
test_cmd = sprintf('!svm-predict -b 1 %s %s %s > log1', temp_test_file, temp_model_file, temp_output_file);

[num_train_data, num_feature] = size(X_train);   
[num_test_data, num_feature] = size(X_test);   

if (~isempty(X_train)),
    fid = fopen(temp_train_file, 'w');
    for i = 1:num_train_data,
        Xi = X_train(i, :);
        % Write label as the first entry
        s = sprintf('%d ', Y_train(i));
        % Then follow 'feature:value' pairs
        ind = find(Xi);
        s = [s sprintf(['%i:' '%.10g' ' '], [ind' full(Xi(ind))']')];
        fprintf(fid, '%s\n', s);
    end
    fclose(fid);
    % train the svm
    fprintf('Running: %s..................\n', train_cmd);
    eval(train_cmd);
end;

% Prediction
fid = fopen(temp_test_file, 'w');
for i = 1:num_test_data,
  Xi = X_test(i, :);
  % Write label as the first entry
  s = sprintf('%d ', Y_test(i));
  % Then follow 'feature:value' pairs
  ind = find(Xi);
  s = [s sprintf(['%i:' '%.10g' ' '], [ind' full(Xi(ind))']')];
  fprintf(fid, '%s\n', s);
end
fclose(fid);
fprintf('Running: %s..................\n', test_cmd);
eval(test_cmd);

fid = fopen(temp_output_file, 'r');
line = fgets(fid);

Y = fscanf(fid, '%f');
fclose(fid);

Y = (reshape(Y, num_class + 1, num_test_data))';
Y_compute = int16(Y(:, 1));

if isempty(strfind(line, 'labels 1 0'))
    Y_prob = Y(:, 3);
else
    Y_prob = Y(:, 2);
end




        