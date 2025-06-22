% Si scriva una funzione my_GSeidel.m che calcoli la soluzione del sistema lineare A*x=b utilizzando il metodo iterativo di Gauss-Seidel.
% 
% La funzione prende in ingresso:
% - la matrice dei coefficienti A,
% - il vettore dei termini noti b,
% - il vettore "punto iniziale" x0,
% - il parametro (intero) di salvaguardia sul numero di iterazioni Kmax,
% - la tolleranza per valutare i criteri di arresto tau.
% 
% La funzione restituisce il vettore "soluzione" x e il numero di iterate eseguite k.

function [x, num_it] = my_gauss_seidel(A, b, x0, Kmax, tau)
    num_it = 1;
    x = zeros(length(x0),1);
    x_prec = x0;

    %decomposizione di Gauss Seidel
    M = tril(A); %D - E
    N = (A - M) * -1; %F
    
    while num_it < Kmax
        r = norm(b - A * x) / norm(b);

        x = my_ltri_sol(M,b + N * x_prec);

        dist = (norm(x - x_prec)) / norm(x);

        if dist < tau && r < tau
            break
        end
        
        x_prec = x;
        num_it = num_it + 1;
    end
end