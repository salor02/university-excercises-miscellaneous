% Funzione che calcola la fattorizzazione LU di una matrice A
% utilizzando l'algoritmo di fattorizzazione di Gauss
% Ritorna le matrici L e U tali che L*U = A

function [L,U] = my_gauss_LU(A)
    
    num = length(A);
    %L = zeros(num);
    %U = zeros(num);

    for k = 1:num-1
        if(abs(A(k,k))< eps) % A(k,k) = 0
            error("\n Fattorizzazione non eseguibile \n");
        end

        %for i = k+1:num
        %    A(i,k) = A(i,k)/A(k,k);
        %    for j = k+1:num
        %        A(i,j) = A(i,j) * A(i,k);
        %    end
        %end

        %equivalente al ciclo sopra
        A(k+1:num,k) = A(k+1:num,k) / A(k,k); %scrive i moltiplicatori nella colonna k

        % sottrae alla matrice degli elementi precedenti una matrice formata
        % da m(i) * a(k,j)
        A(k+1:num,k+1:num) = A(k+1:num,k+1:num) - A(k+1:num,k) * A(k,k+1:num); 
    end

    L = tril(A,-1); 
    L = L + eye(num); 
    U = triu(A);
           
end