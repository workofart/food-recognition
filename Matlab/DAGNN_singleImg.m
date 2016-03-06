%% Single Image Automatic Classifier (Based on GoogleNet CNN feature extractor and SVM classifier) %%
clear all; close all; clc;

Folder = '../FoodData/salad/';
FileName = 'test.jpg';
img = imread(horzcat(Dataset,filesep,FileName));

numSets = 10; %<------ Number of classes to classify, also check the pre-trained model

run  ~/Documents/MATLAB/matconvnet-1.0-beta18/matlab/vl_setupnn

net = dagnn.DagNN.loadobj(load(fullfile(MatlabDir,'CNN/imagenet-googlenet-dag.mat')));

% Normalize the image
im_ = single(img);
im_ = imresize(im_, net.meta.normalization.imageSize(1:2));
im_ = im_ - net.meta.normalization.averageImage;

% Evaluate the network on the input image
net.eval({'data', im_});
        
% Store the extracted features in a cell
feat = squeeze(net.vars(net.getVarIndex('cls3_pool')).value)';

% Load the pre-trained SVM model
load '10classSVM.mat' %<----- Also check the number of classes

% Predict the image
p = zeros(numSets, numLabels);
for j=1:numLabels
    [pred_train, acc, prob] = svmpredict(double(T_Test == j), F_Test, model{j},'-b 1');
    p(:, j) = prob(:, model{j}.Label==1); % Probability class = k
end
