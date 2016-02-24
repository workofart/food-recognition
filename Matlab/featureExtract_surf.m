% % % % % % % % % % % % Feature Extraction using SURF % % % % % % % % % % 
% Support direct SVM w/o Bag of Features
% Read images from files
% Create image sets, feature sets
% Repeat for every class
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 

clear;
% % % % % % % % % % % % % Apple % % % % % % % % % % % % %
disp('==========Apple=======');
Dataset = '../FoodData/apple/';   

minFeatures = 27;
B = zeros(minFeatures,1);


Apple = cell([], 1);
ApplePoints = cell([], 1);
AppleFeatures = cell([], 1);
AppleFeatureMetrics = cell([], 1);
sz = [1000 1000];

 k = dir(fullfile(Dataset,'*.jpg'));
 k = {k(~[k.isdir]).name};
 
 for j=1:length(k)
     try
%      k{j} % image name, to see which one screwed up
    Apple{j} = imresize(rgb2gray(imread(horzcat(Dataset,filesep,k{j}))), sz); % with resize
    ApplePoints{j} = detectSURFFeatures(Apple{j}); %
    AppleFeatures{j} = extractFeatures(Apple{j}, ApplePoints{j}, 'Upright', true);
    AppleFeatureMetrics{j} = var(AppleFeatures{j}, [], 2);
    [sortvals, sortidx] = sort(AppleFeatureMetrics{j},'descend');
    B = sortidx(1:minFeatures);
    AppleFeatures{j} = AppleFeatures{j}(B, :); % Extract the top minFeatures
    AppleFeatures{j} = reshape(AppleFeatures{j}, [minFeatures*64, 1]);
     catch 
        k{j}
        movefile(fullfile(Dataset,k{j}), '../FoodData/apple/Unqualified');
     end
 end

%  Get the number of features from SURF - Apple
%  numFeatures = zeros(length(k), 1);
%   for j=1:length(k)
%     numFeatures(j) = size(AppleFeatures{j}, 1);
%   end
% prctile(numFeatures, 5) % get the value of the bottom 5th percentile

% % % % % % % % % % % % % Burger % % % % % % % % % % % % %
disp('==========Burger=======');
Dataset = '../FoodData/burger/';

Burger = cell([], 1);
BurgerPoints = cell([], 1);
BurgerFeatures = cell([], 1);
BurgerFeatureMetrics = cell([], 1);

k = dir(fullfile(Dataset,'*.jpg'));
k = {k(~[k.isdir]).name};
for j=1:length(k)
    try
%      k{j} % image name, to see which one screwed up
    Burger{j} = imresize(rgb2gray(imread(horzcat(Dataset,filesep,k{j}))), sz); % with resize
    BurgerPoints{j} = detectSURFFeatures(Burger{j}); %
    BurgerFeatures{j} = extractFeatures(Burger{j}, BurgerPoints{j}, 'Upright', true);
    BurgerFeatureMetrics{j} = var(BurgerFeatures{j}, [], 2);
    [sortvals, sortidx] = sort(BurgerFeatureMetrics{j},'descend');
    B = sortidx(1:minFeatures);
    BurgerFeatures{j} = BurgerFeatures{j}(B, :); % Extract the top minFeatures
    BurgerFeatures{j} = reshape(BurgerFeatures{j}, [minFeatures*64, 1]);
     catch 
        k{j}
        movefile(fullfile(Dataset,k{j}), '../FoodData/burger/Unqualified');
     end
end

%  Get the number of features from SURF - Burger
%  numFeatures = zeros(length(k), 1);
%   for j=1:length(k)
%     numFeatures(j) = size(BurgerFeatures{j}, 1);
%   end
% prctile(numFeatures, 5) % get the value of the bottom 5th percentile


% % % % % % % % % % % % % Coffee % % % % % % % % % % % % %
disp('==========Coffee=======');
Dataset = '../FoodData/coffee/';   

Coffee = cell([], 1);
CoffeePoints = cell([], 1);
CoffeeFeatures = cell([], 1);
CoffeeFeatureMetrics = cell([], 1);

k = dir(fullfile(Dataset,'*.jpg'));
k = {k(~[k.isdir]).name};
 for j=1:length(k)
    try
%      k{j} % image name, to see which one screwed up
    Coffee{j} = imresize(rgb2gray(imread(horzcat(Dataset,filesep,k{j}))), sz); % with resize
    CoffeePoints{j} = detectSURFFeatures(Coffee{j}); %
    CoffeeFeatures{j} = extractFeatures(Coffee{j}, CoffeePoints{j}, 'Upright', true);
    CoffeeFeatureMetrics{j} = var(CoffeeFeatures{j}, [], 2);
    [sortvals, sortidx] = sort(CoffeeFeatureMetrics{j},'descend');
    B = sortidx(1:minFeatures);
    CoffeeFeatures{j} = CoffeeFeatures{j}(B, :); % Extract the top minFeatures
    CoffeeFeatures{j} = reshape(CoffeeFeatures{j}, [minFeatures*64, 1]);
     catch 
        k{j}
        movefile(fullfile(Dataset,k{j}), '../FoodData/coffee/Unqualified');
     end
 end

%  Get the number of features from SURF - Burger
%  numFeatures = zeros(length(k), 1);
%   for j=1:length(k)
%     numFeatures(j) = size(CoffeeFeatures{j}, 1);
%   end
% prctile(numFeatures, 5) % get the value of the bottom 5th percentile

