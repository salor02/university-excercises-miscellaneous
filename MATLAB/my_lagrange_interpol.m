% Si scriva una funzione my_lagrange_interpol.m che, dati n+1 punti del
% piano R^2, valuti il polinomio interpolante di grado n costruito rispetto
% alla base di Lagrange in un ulteriore punto dato in input x_eval.
% 
% La funzione prende in ingresso: - un vettore colonna di nodi (ascisse)
% dei punti, - un vettore colonna di ordinate dei punti, - il punto
% (x_eval) in cui si valuta il polinomio.
% 
% La funzione restituisce il numero reale (y_eval) in cui è valutato il
% polinomio interpolante.
% 
% Si utilizzi un vettore per memorizzare i valori degli n+1 elementi della
% base di Lagrange calcolati in x_eval.
% 
% Si utilizzi una variabile omega_n per memorizzare il prodotto delle n+1
% differenze (x_eval - nodo_i), in modo da ottimizzare il calcolo dei
% numeratori degli elementi della base di Lagrange.

function y_eval = my_lagrange_interpol(x_nodi, y_nodi, x_eval)
    
    N = length(x_nodi);

    %calcolo di omega_n per ottimizzazione
    omega_n = 1;

    for i = 1 : N
        omega_n = omega_n * (x_eval - x_nodi(i));
    end

    l = zeros(N,1);

    for k = 1 : N %calcolo della base di Lagrange e valutazione
        d = 1;

        for j = 1 : N %calcolo denominatore
            if j ~= k
                d = d * (x_nodi(k) - x_nodi(j));
            end
        end

        %valutazione polinomio lk
        l(k) = omega_n / ((x_eval - x_nodi(k)) * d);
    end

    %somma dei vari lk moltiplicati per i coefficienti
    y_eval = y_nodi' * l;
end