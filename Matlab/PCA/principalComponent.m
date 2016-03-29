%% Performs PCA on the loaded features
clear;

% classes = {'apple', 'burger', 'coffee', 'donuts', 'french_fries', 'fried_rice', 'ice_cream', 'ramen', 'salad' 'sashimi'};
% classes = {'apple', 'burger', 'coffee'};

load 'CNN/GoogleNet/appleFeat.mat';
load 'CNN/GoogleNet/burgerFeat.mat';
load 'CNN/GoogleNet/coffeeFeat.mat';

% load 'CNN/VGG-F/AppleFeat.mat';
% load 'CNN/VGG-F/BurgerFeat.mat';
% load 'CNN/VGG-F/CoffeeFeat.mat';


% Eigen Version
% [M, N] = size(appleFeat);
% 
% mn = mean(appleFeat, 2);
% data = appleFeat - repmat(mn, 1, N);
% 
% % calculate covariance matrix
% covar = 1 / (N-1) * data * data';
% 
% [PC, V] = eig(covar);
% 
% V = diag(V);
% 
% % Sort variances in decreasing order
% [junk, rindices] = sort(-1*V);
% V = V(rindices);
% 
% PC = PC(:, rindices);
% 
% signals = PC' * data;


% SVD version
% appleFeat = AppleFeat'; % transpose so that MxN (M dimensions, N trials)
% 
% [M, N] = size(appleFeat);
% mn = mean(appleFeat, 2);
% data = appleFeat - repmat(mn, 1, N);
% 
% 
% % Calculate Covariance Matrix
% covar = 1 / (N-1) * data * data';
% 
% % Find eigenvectors and eigenvalues
% [PC, V] = eig(covar);
% 
% % Extract diagonal of matrix as vector
% V = diag(V);
% 
% % Sort variances in decreasing order
% [junk, rindices] = sort(-1*V);
% V = V(rindices);
% 
% PC = PC(:, rindices);
% 
% signals = PC' * data;
% 
% appleFeatPC = signals


% Each column of score corresponds to one principal component.
% The vector, latent, stores the variances of the four principal components.
% 
[coeff,score,latent,~,appleExplained,~] = pca(AppleFeat);
Xcentered = score*coeff';
AppleFeatPCA = score(:, 1:700);

[coeff,score,latent,~,burgerExplained,~] = pca(BurgerFeat');
Xcentered = score*coeff';
BurgerFeatPCA = score(:, 1:700);

[coeff,score,latent,~,coffeeExplained,~] = pca(CoffeeFeat');
Xcentered = score*coeff';
CoffeeFeatPCA = score(1:700, :);













