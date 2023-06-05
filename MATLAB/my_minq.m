% Scrivere una function (my_minq.m) che risolve il problema dei minimi quadrati nel caso generale lineare, non degenere.
% 
% INPUT: x un vettore di m elementi (ascisse dei dati)
% 	  y un vettore di m elementi (ordinate dei dati)
% 	  f vettore di funzioni di base
% 	  un booleano: se true effettua il controllo sul rango 
% OUTPUT: ritornare alpha.

function alpha = my_minq(x, y, f, check_rank)

    n = length(f);
    m = length(x);

    if n > m
        error("Numero di parametri da calcolare maggiore del numero di dati forniti");
    end

    A = ones(m, n); %matrice delle incognite

    for k = 1 : n
        A(:, k) = f{k}(x);
    end

    if check_rank == true
        if rank(A) ~= n
            error("Il rango della matrice A non è uguale al numero dei parametri da calcolare");
        end
    end

    [Q, R] = qr(A); %fattorizzazione di A
    R = R(1:n,1:n);

    %calcolo y segnato
    y_tilde = Q' * y;
    y_seg = y_tilde(1:n);

    %risoluzione del sistema triangolare superiore
    alpha = my_utri_sol(R, y_seg);

end