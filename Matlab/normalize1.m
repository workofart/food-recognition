% % % % % % % % % % Utility function to normalize matrix X % % % % % % % % 
% Returns the same matrix
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
function [X] = normalize1(X)
    for i = 1:size(X, 1)
        if(norm(X(i, :)) == 0)
            
        else
            X(i, :) = X(i, :)./norm(X(i,:));
        end
    end 
end