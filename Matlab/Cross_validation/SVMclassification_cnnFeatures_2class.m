clear;

% Load features
load '../CNN/GoogleNet/appleFeat.mat';
load '../CNN/GoogleNet/burgerFeat.mat';

% Load VGG-F Features
% load '../CNN/VGG-F/AppleFeat.mat';
% load '../CNN/VGG-F/BurgerFeat.mat';
% load '../CNN/VGG-F/CoffeeFeat.mat';

% load 'pca_feat.mat';
% load 'pca_feat_Googlenet.mat';

numSample =1200;
numSets = 2; % total number of classes

% make sure: numTrain + numTest = numSample/numSets
numTrain = 450;
numTest = 150;

%%%%%%%%%%%%%%%% [trainingSets, validationSets] = partition(imgSets, 0.3, 'randomize');

% Total number of data of each class
numapple = size(appleFeat,1);
numburger = size(burgerFeat,1);


% Random index for random sampling
idxapple = randperm(numapple);
idxburger = randperm(numburger);


% Getting random data sets from different classes
appleSet = double(appleFeat(idxapple(1:numSample/numSets), :));
burgerSet = double(burgerFeat(idxburger(1:numSample/numSets),:));


% Mean Subtraction
appleMean = repmat(mean(appleSet,1), size(appleSet,1),1);
appleSet = appleSet - appleMean;

burgerMean = repmat(mean(burgerSet,1), size(burgerSet,1),1);
burgerSet = burgerSet - burgerMean;


% Combining different classes to form training and testing data
F_Train = [appleSet(1:numTrain, :);
    burgerSet(1:numTrain, :);
    ];

F_Test = [appleSet((numTrain+1):(numSample/numSets), :);
    burgerSet((numTrain+1):(numSample/numSets), :);
    ];

% Storing labels for training data
for k=1:numSets
    T_Train((k-1)*numTrain+1:(k*numTrain)) = k;
end
T_Train = T_Train';

% Storing labels for testing data
for k=1:numSets
    T_Test((k-1)*numTest+1:(k*numTest)) = k;
end
T_Test = T_Test';


% Shuffle the training and testing data within each set
TrainSize = size(F_Train,1);
ordering = randperm(TrainSize);
randF_Train = F_Train(ordering, :);
randT_Train = T_Train(ordering, :);


TestSize = size(F_Test, 1);
ordering = randperm(TestSize);
randF_Test = F_Test(ordering, :);
randT_Test = T_Test(ordering, :);

%%
% #######################
% Automatic Cross Validation 
% Parameter selection using n-fold cross validation
% #######################
optionCV.stepSize = 5;
optionCV.c = 1;
optionCV.gamma = 1/1024;
optionCV.stepSize = 5;
optionCV.bestLog2c = 0;
optionCV.bestLog2g = log2(1/1024);
optionCV.epsilon = 0.005;
optionCV.Nlimit = 100;
optionCV.svmCmd = '-q';

[bestc, bestg, bestcv] = automaticParameterSelection(randT_Train, randF_Train, 3, optionCV);

%%
% #######################
% Classification using N-fold cross validation
% #######################
optionClassif.c = bestc;
optionClassif.gamma = bestg;
optionClassif.NClass = numSets;
optionClassif.svmCmd = '-q';
Ncv_classif = 5;
run = (1:size(randT_Test))';
runCVIndex = mod(run,Ncv_classif)+1;

[predictedLabel, decisValueWinner, totalAccuracy, confusionMatrix, order] = classifyUsingCrossValidation(randT_Test, randF_Test, runCVIndex, Ncv_classif, optionClassif);

%%
% #######################
% Make confusion matrix for the overall classification
% #######################
[confusionMatrixAll,orderAll] = confusionmat(randT_Test,predictedLabel);
figure; imagesc(confusionMatrixAll');
xlabel('actual class label');
ylabel('predicted class label');
title(['confusion matrix for overall classification']);
% Calculate the overall accuracy from the overall predicted class label
accuracyAll = trace(confusionMatrixAll)/(numSets*numTest);
disp(['Total accuracy from ',num2str(Ncv_classif),'-fold cross validation is ',num2str(accuracyAll*100),'%']);

% Compare the actual and predicted class
figure;
subplot(1,2,1); imagesc(randT_Test); title('actual class');
subplot(1,2,2); imagesc(predictedLabel); title('predicted class');


