% takes input C, T column vector of the true class of samples
% the ouput is a matrix (2x2)
% CONF(i, j) - number of examples of class i that are classified as class j
% Diagnal are the correct classifications, off-diagonal are
% mis-classifications
% error rate is calculated by using the find function
function [CONF,err] = confusionMatrix(C,T)

% i true class classified as j
pos1 = find(T == 1); % i = 1, positions
pos2 = find(T == 2); % i = 2, positions
pos_1 = find(C == 1); % j = 1, positions
pos_2 = find(C == 2); % j = 2, positions

C11 = length(intersect(pos1,pos_1)); % how many matches for i = 1, j = 1
C12 = length(intersect(pos1,pos_2)); % how many matches for i = 1, j = 2
C21 = length(intersect(pos2,pos_1)); % how many matches for i = 2, j = 1
C22 = length(intersect(pos2,pos_2)); % how many matches for i = 2, j = 2

CONF = [C11 C12; C21 C22]; % create the CONF matrix with the corresponding sums

err = (C12+C21)/(C11+C12+C21+C22);

