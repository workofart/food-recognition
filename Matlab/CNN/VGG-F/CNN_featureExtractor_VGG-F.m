% % % % % % % % % % % CNN Feature Extraction % % % % % % % % % % % 
% Load pre-trained CNN
% Run imageSetConstructor to get 10 sets of images by class
% Extract the features from the second-last layer of the CNN
% Normalize the extracted features
% Repeat for every set
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 

clear all; close all; clc;
run  ~/Documents/MATLAB/matconvnet-1.0-beta18/matlab/vl_setupnn

% % % % % % % % % % % load the pre-trained CNN % % % % % % % % % % % 
% net = dagnn.DagNN.loadobj(load('imagenet-googlenet-dag.mat')) ;
net = load('imagenet-vgg-f.mat');

% load and preprocess an image
% Dataset = '../FoodData/coffee/';   
[appleSet, coffeeSet, french_friesSet, ice_creamSet, saladSet, burgerSet, donutsSet, fried_riceSet, ramenSet, sashimiSet] = imageSetConstructor();

% % % % % % % % % % % Optional Resizing % % % % % % % % % % % 
% sz = [200 200];

% Initialize counters
correctCounter = 0;
totalCounter = 0;


feat = [];
rgbImgList = {};
 
% % % % % % % % % % Loop through each set to extract features % % % % % % % % % %
 for j=1:length(appleSet)
    
     if size(appleSet{j}, 3) == 3
         im_ = single(appleSet{j});
         im_ = imresize(im_, net.meta.normalization.imageSize(1:2));
         im_ = im_ - net.meta.normalization.averageImage;

         res = vl_simplenn(net,im_);

         featVec = res(20).x;
         featVec = featVec(:);
         feat = [feat; featVec'];
         fprintf('extract %d image \n\n', j);
     else
        im_ = single(repmat(appleSet{j}, [1 1 3]));
        im_ = imresize(im_, net.meta.normalization.imageSize(1:2));
        im_ = im_ - net.meta.normalization.averageImage;
        res = vl_simplenn(net, im_);
        
        featVec = res(20).x;
        featVec = featVec(:);
        feat = [feat; featVec'];
        fprintf('extract %d image \n\n', j);
     end 
 end
%  Normalize extracted features
 appleFeat = normalize1(feat);
 
%  Burger
 for j=1:length(burgerSet)
    
     if size(burgerSet{j}, 3) == 3
         im_ = single(burgerSet{j});
         im_ = imresize(im_, net.meta.normalization.imageSize(1:2));
         im_ = im_ - net.meta.normalization.averageImage;
         % run the CNN
         res = vl_simplenn(net,im_);

         featVec = res(20).x;
         featVec = featVec(:);
         feat = [feat; featVec'];
         fprintf('extract %d image \n\n', j);
     else
        im_ = single(repmat(burgerSet{j}, [1 1 3]));
        im_ = imresize(im_, net.meta.normalization.imageSize(1:2));
        im_ = im_ - net.meta.normalization.averageImage;
        res = vl_simplenn(net, im_);
        
        featVec = res(20).x;
        featVec = featVec(:);
        feat = [feat; featVec'];
        fprintf('extract %d image \n\n', j);
     end 
 end
 burgerFeat = normalize1(feat);