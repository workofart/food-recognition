function [C] = LinearMSE(F_Train,T_Train, F_Test)
    
    totalSamples = size(F_Train,1); % size of the training data
    totalTests = size(F_Test,1); % size of the test data (used as verification)
    trueClass = T_Train; % the true class of the training data
    

    % Create a B matrix that contain all the b-parameters, initially set to
    B = trueClass; % create a vector 1500 by 1
    
    Z = [ones(size(F_Train,1),1), F_Train]; % append the ones vector the front and transpose to create the Z matrix    
    
    A = pinv(Z) * B;
  
    %     Calculate the predicted classes
    C = [ones(size(F_Test,1),1), F_Test] * A;
    
    % replace the predict values with corresponding classes
    C(C > 0) = 1;
    C(C < 0) = 2;
    
end
   
    
    

