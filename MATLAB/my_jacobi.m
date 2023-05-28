% Si scriva una funzione my_jacobi.m che calcoli la soluzione del sistema lineare A*x=b utilizzando il metodo iterativo di Jacobi.
% 
% La funzione prende in ingresso:
% - la matrice dei coefficienti A,
% - il vettore dei termini noti b,
% - il vettore "punto iniziale" x0,
% - il parametro (intero) di salvaguardia sul numero di iterazioni Kmax,
% - la tolleranza per valutare i criteri di arresto tau.
% 
% La funzione restituisce il vettore "soluzione" x e il numero di iterate eseguite k.

function [x, num_it] = my_jacobi(A, b, x0, Kmax, tau)

    num_it = 1;
    x = zeros(length(x0),1);
    x_prec = x0;

    %decomposizione di Jacobi
    M = diag(A);
    N = (A - diag(M)) * -1;
    
    while num_it < Kmax
        r = (norm(b - A * x) / norm(b));

        x = (b + N * x_prec) ./ M;

        dist = (norm(x - x_prec)) / norm(x);

        if dist < tau && r < tau
            break
        end
        
        x_prec = x;
        num_it = num_it + 1;
    end
end