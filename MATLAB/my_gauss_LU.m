% Funzione che calcola la fattorizzazione LU di una matrice A
% utilizzando l'algoritmo di fattorizzazione di Gauss
% Ritorna le matrici L e U tali che L*U = A

function [L,U] = my_gauss_LU(A)
    
    num = length(A);
    %L = zeros(num);
    %U = zeros(num);

    for k = 1:num-1
        %for i = k+1:num
        %    A(i,k) = A(i,k)/A(k,k);
        %    for j = k+1:num
        %        A(i,j) = A(i,j) * A(i,k);
        %    end
        %end
        A(k+1:num,k) = A(k+1:num,k) / A(k,k);
        A(k+1:num,k+1:num) = A(k+1:num,k+1:num) - A(k+1:num,k) * A(k,k+1:num);
    end

    L = tril(A,-1); 
    L = L + eye(num); 
    U = triu(A);
           
end