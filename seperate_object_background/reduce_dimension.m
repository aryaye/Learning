function [index] = reduce_dimension(M)
% input M is a M1*64 matrix 
% the goal is to find the index of the second largest number of each row
% return a M1*1 matrix
[c,~]=size(M);
index = zeros(1,c);
for i = 1:c
    [~,B] = sort(M(i,:), 'descend');
    index(i)= B(2);
end
end

