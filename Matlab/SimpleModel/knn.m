clear all;

% Load features
% load 'CNN/GoogleNet/appleFeat.mat';
% load 'CNN/GoogleNet/burgerFeat.mat';
% load 'CNN/GoogleNet/coffeeFeat.mat';
% load 'CNN/GoogleNet/donutsFeat.mat';
% load 'CNN/GoogleNet/french_friesFeat.mat';
% load 'CNN/GoogleNet/fried_riceFeat.mat';
% load 'CNN/GoogleNet/ice_creamFeat.mat';
% load 'CNN/GoogleNet/ramenFeat.mat';
% load 'CNN/GoogleNet/saladFeat.mat';
% load 'CNN/GoogleNet/sashimiFeat.mat';

load '../Edge/AppleSurf_top9.mat';
load '../Edge/BurgerSurf_top9.mat';



numSample = 1200;
numSets = 2; % total number of classes

% make sure: numTrain + numTest = numSample/numSets
numTrain = 400;
numTest = 200;

%%%%%%%%%%%%%%%% [trainingSets, validationSets] = partition(imgSets, 0.3, 'randomize');

% Total number of data of each class
numapple = size(AppleFeatures,2);
numburger = size(BurgerFeatures,2);
% numcoffee = size(coffeeFeat,1);
% numfrench_fries = size(french_friesFeat,1);
% numice_cream = size(ice_creamFeat,1);
% numdonuts = size(donutsFeat,1);
% numfried_rice = size(fried_riceFeat,1);
% numramen = size(ramenFeat,1);
% numsalad = size(saladFeat,1);
% numsashimi = size(sashimiFeat,1);


% Random index for random sampling
idxapple = randperm(numapple);
idxburger = randperm(numburger);
% idxcoffee = randperm(numcoffee);
% idxfrench_fries = randperm(numfrench_fries);
% idxice_cream = randperm(numice_cream);
% idxdonuts = randperm(numdonuts);
% idxfried_rice = randperm(numfried_rice);
% idxramen = randperm(numramen);
% idxsalad = randperm(numsalad);
% idxsashimi = randperm(numsashimi);



% Getting random data sets from different classes
appleSet = double(cell2mat(AppleFeatures(idxapple(1:numSample/numSets))))';
burgerSet = double(cell2mat(BurgerFeatures(idxburger(1:numSample/numSets))))';
%%%% Need to change the names
% coffeeSet = double(coffeeFeat(idxcoffee(1:numSample/numSets),:));
% french_friesSet = double(french_friesFeat(idxfrench_fries(1:numSample/numSets),:));
% ice_creamSet = double(ice_creamFeat(idxice_cream(1:numSample/numSets),:));
% donutsSet = double(donutsFeat(idxdonuts(1:numSample/numSets),:));
% fried_riceSet = double(fried_riceFeat(idxfried_rice(1:numSample/numSets),:));
% ramenSet = double(ramenFeat(idxramen(1:numSample/numSets),:));
% saladSet = double(saladFeat(idxsalad(1:numSample/numSets),:));
% sashimiSet = double(sashimiFeat(idxsashimi(1:numSample/numSets),:));

% Mean Subtraction
appleMean = repmat(mean(appleSet,1), size(appleSet,1),1);
appleSet = appleSet - appleMean;

burgerMean = repmat(mean(burgerSet,1), size(burgerSet,1),1);
burgerSet = burgerSet - burgerMean;

% coffeeMean = repmat(mean(coffeeSet,1), size(coffeeSet,1),1);
% coffeeSet = coffeeSet - coffeeMean;
% 
% french_friesMean = repmat(mean(french_friesSet,1), size(french_friesSet,1),1);
% french_friesSet = french_friesSet - french_friesMean;
% 
% ice_creamMean = repmat(mean(ice_creamSet,1), size(ice_creamSet,1),1);
% ice_creamSet = ice_creamSet - ice_creamMean;
% 
% donutsMean = repmat(mean(donutsSet,1), size(donutsSet,1),1);
% donutsSet = donutsSet - donutsMean;
% 
% fried_riceMean = repmat(mean(fried_riceSet,1), size(fried_riceSet,1),1);
% fried_riceSet = fried_riceSet - fried_riceMean;
% 
% ramenMean = repmat(mean(ramenSet,1), size(ramenSet,1),1);
% ramenSet = ramenSet - ramenMean;
% 
% saladMean = repmat(mean(saladSet,1), size(saladSet,1),1);
% saladSet = saladSet - saladMean;
% 
% sashimiMean = repmat(mean(sashimiSet,1), size(sashimiSet,1),1);
% sashimiSet = sashimiSet - sashimiMean;

% Combining different classes to form training and testing data
F_Train = [appleSet(1:numTrain, :);
    burgerSet(1:numTrain, :)
%     coffeeSet(1:numTrain, :); 
%     french_friesSet(1:numTrain, :); 
%     ice_creamSet(1:numTrain, :);
%     donutsSet(1:numTrain, :);
%     fried_riceSet(1:numTrain, :);
%     ramenSet(1:numTrain, :);
%     saladSet(1:numTrain, :);
%     sashimiSet(1:numTrain, :)
    ];

F_Test = [appleSet((numTrain+1):(numSample/numSets), :);
    burgerSet((numTrain+1):(numSample/numSets), :)
%     coffeeSet((numTrain+1):(numSample/numSets), :);
%     french_friesSet((numTrain+1):(numSample/numSets), :);
%     ice_creamSet((numTrain+1):(numSample/numSets), :);
%     donutsSet((numTrain+1):(numSample/numSets), :);
%     fried_riceSet((numTrain+1):(numSample/numSets), :);
%     ramenSet((numTrain+1):(numSample/numSets), :);
%     saladSet((numTrain+1):(numSample/numSets), :);
%     sashimiSet((numTrain+1):(numSample/numSets), :)
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

numLabels = max(T_Train);
model = cell(numLabels,1);

k =3;
pred = classifyKnn(F_Train, T_Train, F_Test,k);

[CONF,err] = confusionMatrix(pred,T_Test)
