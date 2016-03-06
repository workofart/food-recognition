% % % % % % %  CNN Classification % % % % % % % % % % % % % % % %
% loads pre-train CNN
% Runs imageSet Constructor to get 10 classes of images into sets
% For every set, perform classification
% Output the best scores vector and prediction vector
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
clear all; close all; clc;

% setup MatConvNet
run  ~/Documents/MATLAB/matconvnet-1.0-beta18/matlab/vl_setupnn

% load the pre-trained CNN
net = load('imagenet-vgg-f.mat') ;

% Load images into respective sets
[appleSet, coffeeSet, french_friesSet, ice_creamSet, saladSet, burgerSet, donutsSet, fried_riceSet, ramenSet, sashimiSet] = imageSetConstructor();


% Store all image sets into parent set for easy processing
Parent = {appleSet, coffeeSet, french_friesSet, ice_creamSet, saladSet, burgerSet, donutsSet, fried_riceSet, ramenSet, sashimiSet};
SetNames = { 'appleSet', 'coffeeSet', 'french_friesSet', 'ice_creamSet', 'saladSet', 'burgerSet', 'donutsSet', 'fried_riceSet', 'ramenSet', 'sashimiSet'};

for k = 8:length(Parent)
    fprintf('Current Set ----- %s\n', SetNames{k});
    pause;
    % Create a empty prediction cell
    prediction = cell([], 1);
    bestScores = zeros(size(Parent{k},2),1);

    for j = 1:length(Parent{k})
        % load and preprocess an image
        im_ = single(Parent{k}{j}) ; % note: 0-255 range
        im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
        im_ = im_ - net.meta.normalization.averageImage ;

        % Run the CNN
        res = vl_simplenn(net, im_) ;

        % Get the prediction scores of each prediction     
        scores = squeeze(gather(res(end).x)) ;
        [bestScore, best] = max(scores) ;
        % Store the predictions and best scores into a vector for reference
        prediction{j} = net.meta.classes.description{best};
        bestScores(j) = best;
        fprintf('extract %d image \n\n', j);
    end

    fprintf('Mode of %s: %d\n', SetNames{k},mode(bestScores));
    fprintf('Predicted Class of %s: %s\n', SetNames{k}, net.meta.classes.description{mode(bestScores)});
    
    %%%%%%%%%%%% Add stuff to output %%%%%%%%%%%
    
    % Optional filter a specific range of values that has the most prediction
%     subBestScores = [];
%     threshold = 925;
    % Slice out the values greater than 'threshold'
%     for i = 1:length(bestScores)
%         if (bestScores(i) > threshold)
%             subBestScores = [subBestScores;bestScores(i)];
%         end
%     end
    pause;
end
