% "my_ltri_sol" prende in input una matrice quadrata triangolare inferiore 
% L e un vettore colonna b e calcola la soluzione del sistema lineare L*x=b
% mediante sostituzione in avanti;

function x = my_ltri_sol(L,b)
    if(istril(L)==false)
        error("Given matrix is not lower triangular")
    end

    num_row = length(L);
    x = zeros(num_row,1); %init sol array

    for j = 1 : num_row 
        %sum = 0; %sum of L(i,j)*x(j) terms at each iteration
        %for j = 1 : (i-1)
        %    sum = sum + L(i,j)*x(j);
        %end
        
        %equivalent to the upper for since it's a row - col product
        x(j) = (b(j) - L(j, 1:j-1) * x(1:j-1)) / L(j,j); %apply the lower matrix formula
    end
end