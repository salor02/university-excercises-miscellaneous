% Calcola la fattorizzazione di Gauss con pivoting parziale: ovvero 
% restituisca tre matrici L,U,P (L triangolare inferiore, U triangolare 
% superiore) tali che L*U = P*A. 

function [L,U,P] = my_gauss_LU_piv(A)
    
    num = length(A);
    P = eye(num);

    for k = 1:num-1
        %calcolo elemento massimo colonna
        [~,r] = max (abs(A(k:num,k)));
        r = r + k-1; %serve perchè max da l'indice del vettore passato
        %se massimo è diverso da elemento diagonale
        if (r~=k)

            %scambio righe matrice A
            tmp = A(r,:);
            A(r,:) = A(k,:);
            A(k,:) = tmp;

            %scambio righe P
            tmp = P(r,:);
            P(r,:) = P(k,:);
            P(k,:) = tmp;
        end

        if(abs(A(k,k))< eps) % A(k,k) = 0
            error("\n Fattorizzazione non eseguibile \n");
        end

        A(k+1:num,k) = A(k+1:num,k) / A(k,k); %scrive i moltiplicatori nella colonna k

        % sottrae alla matrice degli elementi precedenti una matrice formata
        % da m(i) * a(k,j)
        A(k+1:num,k+1:num) = A(k+1:num,k+1:num) - A(k+1:num,k) * A(k,k+1:num); 
    end

    L = tril(A,-1); 
    L = L + eye(num); 
    U = triu(A);

end