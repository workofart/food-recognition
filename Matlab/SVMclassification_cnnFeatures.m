clear;

% Load features
% load 'Data/CNN/AppleFeat.mat';
load 'Data/CNN/BurgerFeat.mat';
load 'Data/CNN/CoffeeFeat.mat';

numSample = 2100;
numSets = 3;

% numTrain + numTest == numSample/numSets
numTrain = 500;
numTest = 200;

%%%%%%%%%%%%%%%% [trainingSets, validationSets] = partition(imgSets, 0.3, 'randomize');

% Total number of data of each class
numApple = size(AppleFeat,1);
numBurger = size(BurgerFeat,1);
numCoffee = size(CoffeeFeat,1);

% Random index for random sampling
idxApple = randperm(numApple);
idxBurger = randperm(numBurger);
idxCoffee = randperm(numCoffee);


% Getting random data sets from different classes
AppleSet = double(AppleFeat(idxApple(1:numSample/numSets), :));
BurgerSet = double(BurgerFeat(idxBurger(1:numSample/numSets),:));
CoffeeSet = double(CoffeeFeat(idxCoffee(1:numSample/numSets),:));

% Mean Subtraction
AppleMean = repmat(mean(AppleSet,1), size(AppleSet,1),1);
AppleSet = AppleSet - AppleMean;

BurgerMean = repmat(mean(BurgerSet,1), size(BurgerSet,1),1);
BurgerSet = BurgerSet - BurgerMean;

CoffeeMean = repmat(mean(CoffeeSet,1), size(CoffeeSet,1),1);
CoffeeSet = CoffeeSet - CoffeeMean;

% Combining different classes to form training and testing data
F_Train = [AppleSet(1:numTrain, :);BurgerSet(1:numTrain, :);CoffeeSet(1:numTrain, :)];
F_Test = [AppleSet((numTrain+1):(numSample/numSets), :);BurgerSet((numTrain+1):(numSample/numSets), :);CoffeeSet((numTrain+1):(numSample/numSets), :)];

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

numLabels = max(T_Train);
model = cell(numLabels,1);

for k=1:numLabels
    model{k} = svmtrain(double(T_Train == k), F_Train, '-b 1');
end

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