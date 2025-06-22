% "my_utri_sol" prende in input una matrice quadrata triangolare superiore 
% U e un vettore colonna b e calcola la soluzione del sistema lineare U*x=b
% mediante sostituzione all'indietro.

function x = my_utri_sol(U,b)
    if(istriu(U)==false)
        error("Given matrix is not upper triangular");
    end
    
    num_row = length(b);
    x = zeros(num_row,1);

    %apply formula for each row
    for j = num_row : -1 : 1
        x(j) = (b(j) - U(j, j+1:num_row) * x(j+1:num_row)) / U(j,j);
    end
end