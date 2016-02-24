clear;
% % % % % % % % % % % % % % % % Food Data Training % % % % % % % % % % % % 
% load 'datav2.mat'; % Food Data                                           
% model = svmtrain(T_Train, F_Train, '-t 0');
% [predict_label, accuracy, dec_values] = svmpredict(T_Test, F_Test, model);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

% load ../cifar-10-batches-mat/data_batch_1.mat % cifar-10 data
% load 'A1.mat';
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

% numInst = size(data,1);
imgDataExtractionv1; % get img data w/ edge extraction
T_Train(T_Train == -1) = 2;
T_Test(T_Test == -1) = 2;
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
p = zeros(numTest, numLabels);
for j=1:numLabels
    [pred_train, acc, prob] = svmpredict(double(T_Test == j), F_Test, model{j},'-b 1');
    p(:, j) = prob(:, model{j}.Label==1); % Probability class = k
end

%  max(X, [], 1) - max of each column
%  max(X, [], 2) - max of each row
% Get the max of each row(sample), and return the index(class)
[~, class] = max(p, [], 2);
accuracy = sum(class == T_Test)/numTest


% % load fisheriris
% [~,~,labels] = unique(labels);   %# labels: 1/2/3
% % data = zscore(meas);              %# scale features

% 
% %# split training/testing
% idx = randperm(numInst);

% trainData = double(data(idx(1:numTrain),:));  testData = double(data(idx(numTrain+1:numTrain+500),:));
% trainLabel = double(labels(idx(1:numTrain))); testLabel = double(labels(idx(numTrain+1:numTrain+500)));



% %# train one-against-all models
% model = cell(numLabels,1);
% for k=1:numLabels
%     model{k} = svmtrain(double(trainLabel==k), trainData, '-c 1 -g 0.2 -b 1');
% end
% 
% %# get probability estimates of test instances using each model
% prob = zeros(numTest,numLabels);
% for k=1:numLabels
%     [~,~,p] = svmpredict(double(testLabel==k), testData, model{k}, '-b 1');
%     prob(:,k) = p(:,model{k}.Label==1);    %# probability of class==k
% end
% 
% %# predict the class with the highest probability
% [~,pred] = max(prob,[],2);
% acc = sum(pred == testLabel) ./ numel(testLabel)    %# accuracy
% C = confusionmat(testLabel, pred)  