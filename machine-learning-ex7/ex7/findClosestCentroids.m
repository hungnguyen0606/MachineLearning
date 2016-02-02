function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);
N = size(idx, 1);
% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

m = size(centroids,2);
for i=1:N
    idx(i) = 1;
    s = 0;
    for k=1:m
        s = s + (X(i,k)-centroids(1,k))^2;
    end
    minn = s;
    
    for j=2:K
        s = 0;
        for k=1:m
            s = s + (X(i,k)-centroids(j,k))^2;
        end
        
        if s < minn
            idx(i) = j;
            minn = s;
        end
    end
end





% =============================================================

end

