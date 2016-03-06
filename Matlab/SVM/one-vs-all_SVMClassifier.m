clear;

% load ../cifar-10-batches-mat/data_batch_1.mat % cifar-10 data
% Each row of the array stores a 32x32 colour image
% First 1024 red channels
% Second 1024 green channels
% Third 1024 blue channels
% Each channel could store 0-255 value

%%%%%% Extract Training and Test data for CIFAR-10 ONLY %%%%%%
% numTrain = 1500; numTest = 500;
% F_Train = double(data(1:numTrain,:));
% T_Train = double(labels(1:numTrain));

% F_Test = double(data(numTrain+1:numTrain+numTest, :));
% T_Test = double(labels(numTrain+1:numTrain+numTest));

% imgDataExtractionv1_edge; % get img data w/ edge extraction

imgDataExtractionv1_surf; % get img data w/ surf extraction

numLabels = max(T_Train);
model = cell(numLabels,1);

% [predicted_label, accuracy, decision_values/prob_estimates] = svmtrain(training_label_vector, training_instance_matrix [, 'libsvm_options']);
% [predicted_label, accurracy, decision_values/prob_estimates] = svmpredict(testing_label_vector, testing_instance_matrix, model [, 'libsvm_options']);

for k=1:numLabels
    model{k} = svmtrain(double(T_Train == k), F_Train, '-b 1');
end

% % % % % % % % % % % % % % % % % % % % 
% model contains 9 1x1 cells          %
% Within each cell, there are:        %
% To access each, use 'model{n}.name' %
%     Parameters: [5x1 double]        %
%       nr_class: 2                   %
%        totalSV: 835                 %
%            rho: 5.3153              %
%          Label: []                  %
%     sv_indices: [835x1 double]      %
%          ProbA: []                  %
%          ProbB: []                  %
%            nSV: []                  %
%        sv_coef: [835x1 double]      %
%            SVs: [835x1 double]      %
% % % % % % % % % % % % % % % % % % % %

% Predict Training Data
% model{j}.Label == 1 means the current image is predicted to be in class j
p = zeros(numTest*numSets, numLabels);
for j=1:numLabels
    [pred_train, acc, prob] = svmpredict(double(T_Test == j), F_Test, model{j},'-b 1');
    p(:, j) = prob(:, model{j}.Label==1); % Probability class = k
end

%  max(X, [], 1) - max of each column
%  max(X, [], 2) - max of each row
% Get the max of each row(sample), and return the index(class)
[~, class] = max(p, [], 2);
accuracy = sum(class == T_Test)/(numTest*numSets)

% C = confusionmat(testLabel, pred)  