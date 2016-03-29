clear;
% Image Data Extraction using edge detection %

% Load data OR Run 'dataExtract_surf.m'
% load 'Data/AppleSurf_top9.mat';
% load 'Data/BurgerSurf_top9.mat';
% load 'Data/CoffeeSurf_top9.mat';
dataExtract_surf;

numSample = 2100;
numSets = 3;

% numTrain + numTest == numSample/numSets
numTrain = 500;
numTest = 200;

%%%%%%%%%%%%%%%% [trainingSets, validationSets] = partition(imgSets, 0.3, 'randomize');

% Total number of data of each class
numApple = size(Apple,2);
numBurger = size(Burger,2);
numCoffee = size(Coffee,2);

% Random index for random sampling
idxApple = randperm(numApple);
idxBurger = randperm(numBurger);
idxCoffee = randperm(numCoffee);

% Initializing training&testing data with all zeros
AppleData = zeros(numApple, size(AppleFeatures{1},1));
BurgerData = zeros(numBurger, size(BurgerFeatures{1},1));
CoffeeData = zeros(numCoffee, size(CoffeeFeatures{1},1));

% Storing the training&testing data from extracted features
for i=1:numApple
    AppleData(i, :) = AppleFeatures{i}';
end

for i=1:numBurger
    BurgerData(i, :) = BurgerFeatures{i}';
end

for i=1:numCoffee
    CoffeeData(i, :) = CoffeeFeatures{i}';
end

% Getting random data sets from different classes
AppleSet = double(AppleData(idxApple(1:numSample/numSets), :));
BurgerSet = double(BurgerData(idxBurger(1:numSample/numSets),:));
CoffeeSet = double(CoffeeData(idxCoffee(1:numSample/numSets),:));

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