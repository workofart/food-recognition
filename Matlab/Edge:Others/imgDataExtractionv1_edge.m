% Image Data Extraction using edge detection %

load 'Data/Apple.mat';
load 'Data/Burger.mat';
load 'Data/Coffee.mat';
% # image size
sz = [200,200];

numSample = 2100;
numSets = 3;

% numTrain + numTest == numSample/numSets
numTrain = 500;
numTest = 200;

%%%%%%%%%%%%%%%% [trainingSets, validationSets] = partition(imgSets, 0.3, 'randomize');

% %# split training/testing
numApple = size(Apple,2);
numBurger = size(Burger,2);
numCoffee = size(Coffee,2);

idxApple = randperm(numApple);
idxBurger = randperm(numBurger);
idxCoffee = randperm(numCoffee);

AppleData = zeros(numApple, sz(1)^2);
BurgerData = zeros(numBurger, sz(1)^2);
CoffeeData= zeros(numCoffee, sz(1)^2);

for k =1:numApple
    AppleData(k, :) = reshape(double(edge(Apple{k}, 'Canny')),[sz(1)^2,1])';
end

for k =1:numBurger
    BurgerData(k, :) = reshape(double(edge(Burger{k}, 'Canny')),[sz(1)^2,1]);
end

for k=1:numCoffee
    CoffeeData(k, :) = reshape(double(edge(Coffee{k}, 'Canny')),[sz(1)^2,1]);
end


AppleSet = double(AppleData(idxApple(1:numSample/numSets), :));
BurgerSet = double(BurgerData(idxBurger(1:numSample/numSets),:));
CoffeeSet = double(CoffeeData(idxCoffee(1:numSample/numSets),:));



F_Train = [AppleSet(1:numTrain, :);BurgerSet(1:numTrain, :);CoffeeSet(1:numTrain, :)];
F_Test = [AppleSet((numTrain+1):(numSample/numSets), :);BurgerSet((numTrain+1):(numSample/numSets), :);CoffeeSet((numTrain+1):(numSample/numSets), :)];

for k=1:numSets
    T_Train((k-1)*numTrain+1:(k*numTrain)) = k;
end
T_Train = T_Train';

for k=1:numSets
    T_Test((k-1)*numTest+1:(k*numTest)) = k;
end
T_Test = T_Test';