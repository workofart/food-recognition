% Image Data Extraction using edge detection %
Dataset = '../../FoodData/apple/';   
appleSet = cell([], 1);
k = dir(fullfile(Dataset,'*.jpg'));
k = {k(~[k.isdir]).name};
for j=1:length(k)
    appleSet{j} = imread(horzcat(Dataset,filesep,k{j}));
end
     
     
Dataset = '../../FoodData/burger/';   
burgerSet = cell([], 1);
k = dir(fullfile(Dataset,'*.jpg'));
k = {k(~[k.isdir]).name};
for j=1:length(k)
    burgerSet{j} = imread(horzcat(Dataset,filesep,k{j}));
end
    

Apple = appleSet;
Burger = burgerSet;

% # image size
sz = [100,100];

numSample = 1000;
numSets = 2;

% numTrain + numTest == numSample/numSets
numTrain = 350;
numTest = 150;

%%%%%%%%%%%%%%%% [trainingSets, validationSets] = partition(imgSets, 0.3, 'randomize');

% %# split training/testing
numApple = size(Apple,2);
numBurger = size(Burger,2);

idxApple = randperm(numApple);
idxBurger = randperm(numBurger);

AppleData = zeros(numApple, sz(1)^2);
BurgerData = zeros(numBurger, sz(1)^2);

for k =1:numApple
    temp = reshape(+edge(imresize(rgb2gray(Apple{k}),sz), 'Canny'), [sz(1)^2, 1]);
    AppleData(k, :) = temp';
end

for k =1:numBurger
    temp = reshape(+edge(imresize(rgb2gray(Burger{k}),sz), 'Canny'), [sz(1)^2, 1]);
    BurgerData(k, :) = double(temp)';
end

AppleSet = AppleData(idxApple(1:numSample/numSets), :);
BurgerSet = BurgerData(idxBurger(1:numSample/numSets),:);

F_Train = [AppleSet(1:numTrain, :);BurgerSet(1:numTrain, :)];
F_Test = [AppleSet((numTrain+1):(numSample/numSets), :);BurgerSet((numTrain+1):(numSample/numSets), :)];

for k=1:numSets
    T_Train((k-1)*numTrain+1:(k*numTrain)) = k;
end
T_Train = T_Train';

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

k =100;
pred = classifyKnn(F_Train, T_Train, F_Test,k);

[CONF,err] = confusionMatrix(pred,T_Test)