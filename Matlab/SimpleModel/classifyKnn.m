function  [C] = classifyKnn(F_Train, T_Train, F_Test,k) 

    totalSamples = size(F_Train,1); % size of the training data
    totalTests = size(F_Test,1); % size of the test data (used as verification)

    trueClass = T_Train; % the true class of the training data

    for i = 1:totalTests
        newSample = F_Test(i,:);

        testMatrix = repmat(newSample, totalSamples, 1); % try to find which class the test data is part of but we need to compare it to every single one

        absDiff = abs(F_Train - testMatrix).^2;

        dist = sum(absDiff,2);

        [Y, I] = sort(dist); % sort in ascending order and store the sorted matrix in Y and the original index in I

        neighborsInd = I(1:k); % only take the k-nearest neighbor

        neighbors = trueClass(neighborsInd); % get the true class of the 3 nearest neighbor

        class1 = find(neighbors == 1); % return the index of the 3 nearest neighbors that belongs to class 1

        class2 = find(neighbors == 2); % return the index of the 3 nearest neighbors that belongs to class 2

        joint = [size(class1, 1); size(class2, 1)];

        [value, class] = max(joint); % get the maximum # of neighbors and get the class which it belongs to
        
        % Convert class 2 to class -1
%         if class == 2
%             class = -1;
%         end
        C(i) = class;
    end
end

