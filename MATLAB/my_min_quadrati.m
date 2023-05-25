% minimi quadrati
% verificare m>=n e rango(a) = n
% definire a come vandermorth
% fattorizzazione qr matrice A
% calcolo y segnato = Qty 
% estrarre solo prime n componenti di y
% risolvere sistema triangolare sup R alfa = y segnato

function alfa = my_min_quadrati(x,y,n)
    m = length(x);

    if(m<n)
        error("numero di dati minore del grado del polinomio");
    end

    A = ones(m,n);
    for i = 2:n
        A(:,i) = A(:,i-1).*x;
    end

    if(rank(A) ~= n)
        error("rango di A diverso da n");
    end

    [Q,R] = qr(A);
    R = R(1:n,1:n);
    
    yseg = Q' * y;
    yseg = yseg(1:n);

    alfa = my_utri_sol(R,yseg);
end