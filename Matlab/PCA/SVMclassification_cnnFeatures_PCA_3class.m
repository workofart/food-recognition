%% 3-class Classifciation Using PCA GoogLeNet Features
clear;

% Load features
% load '../CNN/GoogleNet/appleFeat.mat';
% load '../CNN/GoogleNet/burgerFeat.mat';
% load '../CNN/GoogleNet/coffeeFeat.mat';

% Load VGG-F Features
% load '../CNN/VGG-F/AppleFeat.mat';
% load '../CNN/VGG-F/BurgerFeat.mat';
% load '../CNN/VGG-F/CoffeeFeat.mat';

% load 'pca_feat.mat';
load 'pca_feat_Googlenet.mat';


% appleFeat = AppleFeat;
% burgerFeat = BurgerFeat;
% coffeeFeat = CoffeeFeat;

appleFeat = AppleFeatPCA;
burgerFeat = BurgerFeatPCA;
coffeeFeat = CoffeeFeatPCA;

numSample =1800;
numSets = 3; % total number of classes

% make sure: numTrain + numTest = numSample/numSets
numTrain = 450;
numTest = 150;

%%%%%%%%%%%%%%%% [trainingSets, validationSets] = partition(imgSets, 0.3, 'randomize');

% Total number of data of each class
numapple = size(appleFeat,1);
numburger = size(burgerFeat,1);
numcoffee = size(coffeeFeat,1);

% Random index for random sampling
idxapple = randperm(numapple);
idxburger = randperm(numburger);
idxcoffee = randperm(numcoffee);

% Getting random data sets from different classes
appleSet = double(appleFeat(idxapple(1:numSample/numSets), :));
burgerSet = double(burgerFeat(idxburger(1:numSample/numSets),:));
coffeeSet = double(coffeeFeat(idxcoffee(1:numSample/numSets),:));

% Mean Subtraction
appleMean = repmat(mean(appleSet,1), size(appleSet,1),1);
appleSet = appleSet - appleMean;

burgerMean = repmat(mean(burgerSet,1), size(burgerSet,1),1);
burgerSet = burgerSet - burgerMean;

coffeeMean = repmat(mean(coffeeSet,1), size(coffeeSet,1),1);
coffeeSet = coffeeSet - coffeeMean;

% Combining different classes to form training and testing data
F_Train = [appleSet(1:numTrain, :);
    burgerSet(1:numTrain, :);
    coffeeSet(1:numTrain, :); 
    ];

F_Test = [appleSet((numTrain+1):(numSample/numSets), :);
    burgerSet((numTrain+1):(numSample/numSets), :);
    coffeeSet((numTrain+1):(numSample/numSets), :);
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


disp('=========================== Randomized =================');
% clear class p; % drop the previous result
numLabels = max(randT_Train);
model = cell(numLabels,1);

for k=1:numLabels
    model{k} = svmtrain(double(randT_Train == k), randF_Train, '-b 1'); %#ok<SVMTRAIN>
end

% Predict Training Data
% model{j}.Label == 1 means the current image is predicted to be in class j
p = zeros(numTest*numSets, numLabels);
for j=1:numLabels
    [pred_train, acc, prob] = svmpredict(double(randT_Test == j), randF_Test, model{j},'-b 1');
    p(:, j) = prob(:, model{j}.Label==1); % Probability class = k
end

%  max(X, [], 1) - max of each column
%  max(X, [], 2) - max of each row
% Get the max of each row(sample), and return the index(class)
[~, class] = max(p, [], 2);
accuracy = sum(class == randT_Test)/(numTest*numSets)

