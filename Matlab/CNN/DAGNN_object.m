clear all; close all; clc;
run  ~/Documents/MATLAB/matconvnet-1.0-beta18/matlab/vl_setupnn


net = dagnn.DagNN();

% Create a convolution filter
convBlock = dagnn.Conv('size', [3 3 256 16], 'hasBias', true) ;
% Add the convolution filter to the network as one layer (name: conv1)
net.addLayer('conv1', convBlock, {'x1'}, {'x2'}, {'filters', 'biases'}) ;

% Create a reLu layer
reluBlock = dagnn.ReLU() ;
% Add the reLu layer to the network (name: relu1)
net.addLayer('relu1', reluBlock, {'x2'}, {'x3'}, {}) ;

% FanOut - how many network layers have the particular varaible/parameter
% as output and input. If Fanout ? 1, parameter sharing is present

% Save the DAGNN object
netStruct = net.saveobj(); % convert obj into a structure
save('myDAGNN.mat', '-struct', 'netStruct');
clear netStruct;

% Load the saved DAGNN object
netStruct = load('myfile.mat') ;
net = dagnn.DagNN.loadobj(netStruct) ;
clear netStruct ;

% Intialize parameters and variables
net.initParams();

% Create a random input for testing forward/backprop
input = randn(10,15,256,1,'single');

% Evaluate the network on an input pair
% Input pair: 'variableName', variableValue
net.eval({'x1', input});

% Store the leaf variables into an output variable
i = net.getVarIndex('x3');
output =net.vars(i).value;

dzdy = rand(size(output), 'single'); % projection vector
net.eval({'x1', input}, {'x3', dzdy});

p = net.getParamIndex('filters');
dzdfilters = net.vars(p).der;
