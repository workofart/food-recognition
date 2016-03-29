%% Compute PCA and % explained variance 
clear;

% load 'CNN/VGG-F/AppleFeat.mat';
% load 'CNN/VGG-F/BurgerFeat.mat';
% load 'CNN/VGG-F/CoffeeFeat.mat';

load 'CNN/GoogleNet/appleFeat.mat';
load 'CNN/GoogleNet/burgerFeat.mat';
load 'CNN/GoogleNet/coffeeFeat.mat';

AppleFeat = appleFeat;
BurgerFeat = burgerFeat;
CoffeeFeat = coffeeFeat;

% Eigen Version

[M, N] = size(AppleFeat); % M - trials, N - dimensions

mn = mean(AppleFeat, 2); % mean of all dimensions of each trial
data = AppleFeat - repmat(mn, 1, N); % subtract mean from original data

% calculate covariance matrix
covar = (data'* data) / (N-1);

[PC, V] = eig(covar); % Diagonal matrix of eigenvalues and full matrix V
                      % whose columns are corresponding eigenvectors
                      % A * V = V * D

% Sort variances in decreasing order
[e, i] = sort(diag(V), 'descend');

% [junk, rindices] = sort(-1*V);
% AppleFeat = PC(:, 1:350);
weights = (e./sum(e))*100; % store the weights of the principal components
explained1 = sum(weights(1:627)); % the first 350 principal components contributed 90.55% of the variance


% newX = data * PC(:, 1:end);
% signals = PC' * data;

[M, N] = size(BurgerFeat); % M - trials, N - dimensions

mn = mean(BurgerFeat, 2); % mean of all dimensions of each trial
data = BurgerFeat - repmat(mn, 1, N); % subtract mean from original data

% calculate covariance matrix
covar = (data'* data) / (N-1);

[PC, V] = eig(covar); % Diagonal matrix of eigenvalues and full matrix V
                      % whose columns are corresponding eigenvectors
                      % A * V = V * D

% Sort variances in decreasing order
[e, i] = sort(diag(V), 'descend');

% [junk, rindices] = sort(-1*V);
% BurgerFeat = PC(:, 1:350);
weights = (e./sum(e))*100; % store the weights of the principal components
explained2 = sum(weights(1:627)); % the first 350 principal components contributed 90.55% of the variance

[M, N] = size(CoffeeFeat); % M - trials, N - dimensions

mn = mean(CoffeeFeat, 2); % mean of all dimensions of each trial
data = CoffeeFeat - repmat(mn, 1, N); % subtract mean from original data

% calculate covariance matrix
covar = (data'* data) / (N-1);

[PC, V] = eig(covar); % Diagonal matrix of eigenvalues and full matrix V
                      % whose columns are corresponding eigenvectors
                      % A * V = V * D

% Sort variances in decreasing order
[e, i] = sort(diag(V), 'descend');

% [junk, rindices] = sort(-1*V);
% CoffeeFeat = PC(:, 1:350);
weights = (e./sum(e))*100; % store the weights of the principal components
explained3 = sum(weights(1:627)); % the first 350 principal components contributed 90.55% of the variance

















