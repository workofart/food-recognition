

%% Total Layers, parameters, vars %%
% # Layers = 153 ----- orange
% # Param = 128 ---- blue
% # Vars = 154 ----- yellow
% % % % % % % % % % % % % %


%% How to Interpret Layers %%
% For every layer, the input image is applied with filters based on the
% params, and outputs a filtered image

%% ----- Start of Program ---- %%
MatlabDir = '~/Dropbox/CS4490/Matlab';
cd ~/Dropbox/CS4490/Matlab;
clear all; close all; clc;
run  ~/Documents/MATLAB/matconvnet-1.0-beta18/matlab/vl_setupnn

[appleSet, coffeeSet, french_friesSet, ice_creamSet, saladSet, burgerSet, donutsSet, fried_riceSet, ramenSet, sashimiSet] = imageSetConstructor();
% load 'appleset.mat';

net = dagnn.DagNN.loadobj(load(fullfile(MatlabDir,'CNN/imagenet-googlenet-dag.mat')));


%% Capturing intermediate parameters/features %%
net.vars(net.getVarIndex('cls3_pool')).precious = true; % enable logging of features before fully connected layer

% Create a feature vector equal to the size of each set
% appleFeat = zeros(size(appleSet, 2),1024);
% burgerFeat = zeros(size(burgerSet, 2),1024);
% coffeeFeat = zeros(size(coffeeSet, 2),1024);
% french_friesFeat = zeros(size(french_friesSet, 2),1024);
donutsFeat = zeros(size(donutsSet, 2),1024);

%% Loop through images in the set %%
for j=1:length(donutsSet)
        % Preprocess Images
        im_ = single(donutsSet{j});
        im_ = imresize(im_, net.meta.normalization.imageSize(1:2));
        im_ = im_ - net.meta.normalization.averageImage;
         
        % Evaluate the network on the input image
        net.eval({'data', im_});
        
        % Store the extracted features in a cell
        donutsFeat(j, :) = squeeze(net.vars(net.getVarIndex('cls3_pool')).value)'; 
        
        % Print status
        fprintf('extract %d image \n\n', j);
end

%% Capturing different intermediate results %%
% net.vars(net.getVarIndex('conv1')).precious = true;
% net.vars(net.getVarIndex('norm1')).precious = true;
% net.vars(net.getVarIndex('conv2')).precious = true;
% net.vars(net.getVarIndex('icp2_in')).precious = true;
% net.vars(net.getVarIndex('icp3_out')).precious = true;
% net.vars(net.getVarIndex('icp4_out')).precious = true;
% net.vars(net.getVarIndex('icp5_out')).precious = true;
% net.vars(net.getVarIndex('icp6_out')).precious = true;
% net.vars(net.getVarIndex('icp9_out')).precious = true;


%%%%%%%%%%%% Display Patches %%%%%%%%%%%%%
% sz = size(net.vars(net.getVarIndex('conv1')).value,3);
% sz = size(net.vars(net.getVarIndex('norm1')).value,3);
% sz = size(net.vars(net.getVarIndex('conv2')).value,3);
% sz = size(net.vars(net.getVarIndex('icp2_in')).value,3);
% sz = size(net.vars(net.getVarIndex('icp3_out')).value,3);
% sz = size(net.vars(net.getVarIndex('icp4_out')).value,3);
% sz = size(net.vars(net.getVarIndex('icp5_out')).value,3);
% sz = size(net.vars(net.getVarIndex('icp6_out')).value,3);
% sz = size(net.vars(net.getVarIndex('icp9_out')).value,3);
% sz = size(net.vars(net.getVarIndex('cls3_pool')).value,3);


% row = floor(sz^0.5);

% for i = 1:row
%     for j = 1:row
%         imgCell{i,j} = net.vars(net.getVarIndex('cls3_fc')).value(:,:,(i-1)*row+j);
%     end
% end
% bigImage = cell2mat(imgCell);
% imshow(bigImage);


%%%%%%%%%%% Get the 2nd-last layer features %%%%%%%%%
% features2 = squeeze(net.vars(net.getVarIndex('cls3_fc')).value);
% features2 = squeeze(net.vars(net.getVarIndex('cls3_pool')).value);
% predProb = squeeze(net.vars(net.getVarIndex('prob')).value);


% All the 1000 class names
% net.meta.classes.description

% Print out the size of each params layer
% for i=1:size(net.params,2)
    
% end