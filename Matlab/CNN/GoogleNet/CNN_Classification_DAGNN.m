% % % % % % % % % % % CNN Classification % % % % % % % % % % % 
% loads pre-train CNN
% Runs imageSet Constructor to get 10 classes of images into sets
% For every set, perform classification
% Output the best scores vector and prediction vector
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 

clear all; close all; clc;
run  ~/Documents/MATLAB/matconvnet-1.0-beta18/matlab/vl_setupnn

% % % % % % % % % % % load the pre-trained CNN % % % % % % % % % % % 
net = dagnn.DagNN.loadobj(load('imagenet-googlenet-dag.mat')) ;


% load and preprocess every image into a set
[appleSet, coffeeSet, french_friesSet, ice_creamSet, saladSet, burgerSet, donutsSet, fried_riceSet, ramenSet, sashimiSet] = imageSetConstructor();

% Store all image sets into parent set for easy processing
Parent = {appleSet, coffeeSet, french_friesSet, ice_creamSet, saladSet, burgerSet, donutsSet, fried_riceSet, ramenSet, sashimiSet};
SetNames = { 'appleSet', 'coffeeSet', 'french_friesSet', 'ice_creamSet', 'saladSet', 'burgerSet', 'donutsSet', 'fried_riceSet', 'ramenSet', 'sashimiSet'};

% % % % % % % % % % % Optional Resizing % % % % % % % % % % % 
% sz = [200 200];

% Initialize the cells
feat = [];
rgbImgList = {};

for k = 8:length(Parent)
    fprintf('Current Set ----- %s\n', SetNames{k});
    pause;
    % Create a empty prediction cell
    prediction = cell([], 1);
    bestScores = zeros(size(Parent{k},2),1);

    % % % % % % % % % % Loop through each set to extract features % % % % % % % % % %
     for j=1:length(Parent{k})
             im_ = single(Parent{k}{j});
             im_ = imresize(im_, net.meta.normalization.imageSize(1:2));
             im_ = im_ - net.meta.normalization.averageImage;

             % Run the CNN
             net.eval({'data', im_});
             scores = net.vars(net.getVarIndex('prob')).value ;
             scores = squeeze(gather(scores)) ;

             [bestScore, best] = max(scores);
             prediction{j} = net.meta.classes.description{best};
             bestScores(j) = best;
    %          featVec = res(20).x;
    %          featVec = featVec(:);
    %          feat = [feat; featVec'];
             fprintf('extract %d image \n\n', j);
     end
    fprintf('Mode of %s: %d\n', SetNames{k},mode(bestScores));
    fprintf('Predicted Class of %s: %s\n', SetNames{k}, net.meta.classes.description{mode(bestScores)});
    pause;
end
%  Optional
% subBestScores = [];
% threshold = 950;
% % Slice out the values greater than 'threshold'
% for i = 1:length(bestScores)
%     if (bestScores(i) > threshold)
%         subBestScores = [subBestScores;bestScores(i)];
%     end
% end
