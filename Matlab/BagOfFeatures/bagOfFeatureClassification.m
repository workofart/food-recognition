% % % % % % % % % % Bag of Features % % % % % % % % % % % %
% Reads the image files
% Randomly splits them into training sets and validation sets
% Performs feature extracting using SURF
% Performs k-means clustering to get 500 features
% Classifies using SVM with default kernels
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 

% % % % % % % % % Image data pre-processing % % % % % % % % %
clear;
imgFolder = fullfile('~', 'Documents', 'FoodData', 'apple');
AppleSet = imageSet(imgFolder);
imgFolder = fullfile('~', 'Documents', 'FoodData', 'burger');
BurgerSet = imageSet(imgFolder);
imgFolder = fullfile('~', 'Documents', 'FoodData', 'coffee');
CoffeeSet = imageSet(imgFolder);
% imgFolder = fullfile('~', 'Documents', 'FoodData', 'donuts');
% DonutsSet = imageSet(imgFolder);
% imgFolder = fullfile('~', 'Documents', 'FoodData', 'french_fries');
% FrenchfriesSet = imageSet(imgFolder);
% imgFolder = fullfile('~', 'Documents', 'FoodData', 'fried_rice');
% FriedriceSet= imageSet(imgFolder);
% imgFolder = fullfile('~', 'Documents', 'FoodData', 'ice_cream');
% IcecreamSet = imageSet(imgFolder);
% imgFolder = fullfile('~', 'Documents', 'FoodData', 'ramen');
% RamenSet = imageSet(imgFolder);
% imgFolder = fullfile('~', 'Documents', 'FoodData', 'salad');
% SaladSet = imageSet(imgFolder);
% imgFolder = fullfile('~', 'Documents', 'FoodData', 'sashimi');
% SashimiSet = imageSet(imgFolder);
% imgSets = [AppleSet, BurgerSet, CoffeeSet, DonutsSet, FrenchfriesSet, FriedriceSet, IcecreamSet, RamenSet, SaladSet, SashimiSet];
imgSets = [AppleSet, BurgerSet, CoffeeSet];

minSetCount = min([imgSets.Count]); % used to get the minimum number of
% samples of all categories
% minSetCount = 500;

% Use the minimum set count as the base line for set creation
imgSets = partition(imgSets, minSetCount, 'randomize');

% Separate Sets into Training and Validation
[trainingSets, validationSets] = partition(imgSets, 0.6, 'randomize');

% % % % % % % % % Create bag of features from training set % % % % % % % % %
bag = bagOfFeatures(trainingSets);

% load 'Data/BagOfFeature/apple_burger_coffee_bag.mat';
% Customize feature extractor for the bag of features framework
% extractorFcn = @v1BagOfFeaturesExtractor;
% bag = bagOfFeatures(imgSets,'CustomExtractor',extractorFcn)

% Display Feature Histogram (500)
img = read(imgSets(1),1);
featureVector = encode(bag, img);
figure
bar(featureVector);
title('Visual Word Occurrences');
xlabel('Visual Word Index');
ylabel('Frequency of occurrence');

% Customize classifier options
% opts = templateSVM('BoxConstraint', 1.1, 'KernelFunction', 'gaussian');

categoryClassifier = trainImageCategoryClassifier(trainingSets, bag);

% Evaluate on Training Set
disp('Evaluate on Training Set');
confMatrix = evaluate(categoryClassifier, trainingSets);

% Evaluate on Validation Set
disp('Evaluate on Validation Set');
confMatrix = evaluate(categoryClassifier, validationSets);

% Compute average accuracy
mean(diag(confMatrix));

% Test the classifier on any given image
% img = imread(fullfile('..', 'FoodData', 'burger', 'test.jpg'));
% [labelIdx, scores] = predict(categoryClassifier, img);
% categoryClassifier.Labels(labelIdx)