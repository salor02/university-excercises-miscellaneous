% La funzione prende in ingresso:
% - gli estremi a e b che definiscono l0intervallo [a,b]
% - il numero n di punti da costruire.
% 
% La funzione restituisce il vettore colonna dei nodi di Chebyschev desiderati.

function ceb_nodes = my_ceb_nodes(a, b, n_pti)

    den = 2 * (n_pti);

    ceb_nodes = zeros(n_pti, 1);

    for i = 1 : n_pti
        ceb_nodes(i) = cos( ((2 * (i - 1) + 1) * pi) / den);
    end

    ceb_nodes = ((b - a)/2) * ceb_nodes + ((a + b)/2);
end