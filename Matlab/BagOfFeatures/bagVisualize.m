%% Used to visualize the different bags using histogram

clear;
% load 10set_300x300.mat; % for 10 class
load apple_burger_coffee_bag.mat; % load the pre-constructed bag - 3 class

% Construct Image set
imgFolder = fullfile('~', 'Documents', 'FoodData', 'apple');
AppleSet = imageSet(imgFolder);
imgFolder = fullfile('~', 'Documents', 'FoodData', 'burger');
BurgerSet = imageSet(imgFolder);
imgFolder = fullfile('~', 'Documents', 'FoodData', 'coffee');
CoffeeSet = imageSet(imgFolder);


% Count the number of occurences of each feature in the image
img = read(AppleSet(1), 5);
featureVector = encode(bag, img);

% Plot the histogram of visual word occurrences
figure
bar(featureVector)
title('Visual word occurrences')
xlabel('Visual word index')
ylabel('Frequency of occurrence')