function [C] = PerceptronBatchNormalized(F_Train,T_Train,F_Test, k)

% Normalize the sample features to be on the same scale
F_Train_N = (F_Train-repmat(mean(F_Train), size(F_Train,1),1))*diag(1./std(F_Train));
    
% Normalize the test features to be on the same scale
F_Test_N = (F_Test-repmat(mean(F_Train), size(F_Test,1),1))*diag(1./std(F_Train));


alpha = 1/k;

totalSamples = size(F_Train_N,1); % size of the training data
totalFeatures = size(F_Train_N,2); % total number of features
totalTests = size(F_Test_N,1); % size of the test data (used as verification)
trueClass = T_Train; % the true class of the training data

Z = [ones(totalSamples,1), F_Train_N]; % append the ones vector the front
                                    % and transpose to create the Z matrix

Z_test = [ones(totalTests,1), F_Test_N];
A = repmat((1/totalFeatures), 1, totalFeatures+1); % initial weights

% replace all examples from class -1 by -Z
Z((trueClass == 2),:) = -Z((trueClass == 2),:);

minClass = totalSamples; % initialize total misclassified examples as total samples


% repeat training for k iterations
for i = 1:k
    
    g = A*Z';
    temp = Z(g' < 0,:);
    misClass = size(temp,1);
    err = sum(temp);
    A = A + alpha.*err;
    
    if (misClass < minClass)
        minClass = misClass;
        bestWeight = A;
    end
    
end

% Use the trained weights to predict the test data
C = bestWeight*Z_test';
C(C>0) = 1;
C(C<0) = 2;
C = C';

end