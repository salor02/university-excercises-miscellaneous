% Si scriva una funzione my_bisection.m che calcoli la soluzione approssimata del problema unidimensionale f(x)=0 utilizzando il metodo di bisezione.
%
% La funzione prende in ingresso:
% - una stringa func_name contenete il nome del file in cui è descritta la funzione f(x), ad esempio se la funzione è contenuta nel file my_function.m la stringa in input sarà "my_function",
% - il numero reale a che definisce l'estremo sinistro dell'intervallo di ricerca,
% - il numero reale b che definisce l'estremo destro dell'intervallo di ricerca,
% - la tolleranza tau per valutare il criterio di arresto.
%
% La funzione restituisce il numero reale x che approssima uno zero di f e il numero di iterate eseguite N.

function [c, N] = my_bisection(func_name, a, b, tau)
    
    fa = feval(func_name, a);
    fb = feval(func_name, b);

    %controllo validità teorema valor medio
    if fa * fb >= 0
        error ("Teorema del valor medio non soddisfatto");
    end

    %calcolo numero iterate
    N = ceil(log2((b - a) / tau));

    for k = 1:N
    
        %calcolo e valutazione del punto medio
        c = a + (b - a)/2;
        fc = feval(func_name, c);

        if fc == 0
            return
        end
        
        %bisezione
        if fa * fc < 0
            b = c;
            fb = fc;
        else
            a = c;
            fa = fc;
        end
    end
end