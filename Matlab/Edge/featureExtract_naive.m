% % % % % % % %  Naive Feature Extractor % % % % % % % % % 
% % % % % % % % % % Based on pixels ONLY % % % % % % % % % 


% % % % % % % % % % % % % Apple % % % % % % % % % % % % %
Dataset = '../FoodData/apple/';   

Apple = cell([], 1);
sz = [200 200];

 k = dir(fullfile(Dataset,'*.jpg'));
 k = {k(~[k.isdir]).name};
 for j=1:length(k)
     j
    tempImage = imread(horzcat(Dataset,filesep,k{j}));
    Apple{j} = imresize(double(rgb2gray(tempImage)), sz);
 end

% % % % % % % % % % % % % Burger % % % % % % % % % % % % %
Dataset = '../FoodData/burger/';   

Burger = cell([], 1);

k = dir(fullfile(Dataset,'*.jpg'));
k = {k(~[k.isdir]).name};
for l=1:length(k)
    l
    tempImage = imread(horzcat(Dataset,filesep,k{l}));
    Burger{l} = imresize(double(rgb2gray(tempImage)), sz);
end

% % % % % % % % % % % % % Coffee % % % % % % % % % % % % %
Dataset = '../FoodData/coffee/';   

Coffee = cell([], 1);

k = dir(fullfile(Dataset,'*.jpg'));
k = {k(~[k.isdir]).name};
 for m=1:length(k)
     m
    tempImage = imread(horzcat(Dataset,filesep,k{m}));
    Coffee{m} = imresize(double(rgb2gray(tempImage)), sz);
 end


