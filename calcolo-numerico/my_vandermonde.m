% Si scriva una funzione my_vandermonde.m che, dati n+1 punti del piano R^2, restituisca la matrice di Vandermonde V di dimensione (n+1)x(n+1).
% 
% La funzione prende in ingresso:
% - un vettore colonna di nodi (ascisse dei punti di interpolazione) x,
% 
% Si costruisca la matrice V per colonna:
% - inizializzare V a una matrice di 1 (utilizzando il comando ones)
% - tranne la prima, ogni colonna si ottiene moltiplicando (elemento per elemento, ovvero puntualmente .*) la colonna precedente per il vettore dei nodi.
% V(:,j) = V(:,j-1) .* x, per ogni j=2,..,n+1.
% Si noti che n+1 è la lunghezza del vettore in input.

function V = my_vandermonde(x_nodi)
    N = length(x_nodi);

    V = ones(N);

    for j = 2 : N
        V(:, j) = V(:, j-1) .* x_nodi;
    end
end