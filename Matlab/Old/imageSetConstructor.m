% % % % % % % % % % % % Constructs sets of images % % % % % % % % % % % % 
% Doesn't apply any preprocessing
% Currently supports 10 classes
% Used to support feature extraction
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
function [appleSet, coffeeSet, french_friesSet, ice_creamSet, saladSet, burgerSet, donutsSet, fried_riceSet, ramenSet, sashimiSet] ...
     = imageSetConstructor()
 
%     clear;
% function appleSet = imageSetConstructor()
    Dataset = '../FoodData/apple/';   
    appleSet = cell([], 1);
     k = dir(fullfile(Dataset,'*.jpg'));
     k = {k(~[k.isdir]).name};
     for j=1:length(k)
        appleSet{j} = imread(horzcat(Dataset,filesep,k{j}));
     end
     
    Dataset = '../FoodData/coffee/';   
    coffeeSet = cell([], 1);
    k = dir(fullfile(Dataset,'*.jpg'));
    k = {k(~[k.isdir]).name};
    for j=1:length(k)
        coffeeSet{j} = imread(horzcat(Dataset,filesep,k{j}));
    end
    
    Dataset = '../FoodData/burger/';   
    burgerSet = cell([], 1);
    k = dir(fullfile(Dataset,'*.jpg'));
    k = {k(~[k.isdir]).name};
    for j=1:length(k)
        burgerSet{j} = imread(horzcat(Dataset,filesep,k{j}));
    end
     
    Dataset = '../FoodData/french_fries/';   
    french_friesSet = cell([], 1);
    k = dir(fullfile(Dataset,'*.jpg'));
    k = {k(~[k.isdir]).name};
    for j=1:length(k)
        french_friesSet{j} = imread(horzcat(Dataset,filesep,k{j}));
    end
    
    Dataset = '../FoodData/ice_cream/';   
    ice_creamSet = cell([], 1);
    k = dir(fullfile(Dataset,'*.jpg'));
    k = {k(~[k.isdir]).name};
    for j=1:length(k)
        ice_creamSet{j} = imread(horzcat(Dataset,filesep,k{j}));
    end
    
    Dataset = '../FoodData/salad/';   
    saladSet = cell([], 1);
    k = dir(fullfile(Dataset,'*.jpg'));
    k = {k(~[k.isdir]).name};
    for j=1:length(k)
        saladSet{j} = imread(horzcat(Dataset,filesep,k{j}));
    end
    
    Dataset = '../FoodData/donuts/';   
    donutsSet = cell([], 1);
    k = dir(fullfile(Dataset,'*.jpg'));
    k = {k(~[k.isdir]).name};
    for j=1:length(k)
        donutsSet{j} = imread(horzcat(Dataset,filesep,k{j}));
    end
    
    Dataset = '../FoodData/fried_rice/';   
    fried_riceSet = cell([], 1);
    k = dir(fullfile(Dataset,'*.jpg'));
    k = {k(~[k.isdir]).name};
    for j=1:length(k)
        fried_riceSet{j} = imread(horzcat(Dataset,filesep,k{j}));
    end
    
    Dataset = '../FoodData/ramen/';   
    ramenSet = cell([], 1);
    k = dir(fullfile(Dataset,'*.jpg'));
    k = {k(~[k.isdir]).name};
    for j=1:length(k)
        ramenSet{j} = imread(horzcat(Dataset,filesep,k{j}));
    end
    
    Dataset = '../FoodData/sashimi/';   
    sashimiSet = cell([], 1);
    k = dir(fullfile(Dataset,'*.jpg'));
    k = {k(~[k.isdir]).name};
    for j=1:length(k)
        sashimiSet{j} = imread(horzcat(Dataset,filesep,k{j}));
    end
end