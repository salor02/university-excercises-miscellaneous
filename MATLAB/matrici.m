function [outputArg1,outputArg2] = matrici(D,b)
%MATRICI Summary of this function goes here
%   Detailed explanation goes here

%check if matrix is diagonal
if(isdiag(D) == false)
        error("D non diagonale");
end

d=diag(D);
res=b./d;

%
n = length(b);
res = zeros(n,1);

for i=1:n
    res(i) = b(i)/D(i,i);
end
%




