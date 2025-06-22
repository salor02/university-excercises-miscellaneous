% "my_diag_sol" prende in input una matrice 
% diagonale D e un vettore colonna b e calcola la soluzione del sistema 
% lineare D*x=b utilizzando l'operatore divisione-puntuale;

function sol = my_diag_sol(D, b)
    if(isdiag(D)==false) 
        error("La matrice in input non è diagonale");
    end
    D_diag = diag(D);
    sol = b ./ D_diag;
end