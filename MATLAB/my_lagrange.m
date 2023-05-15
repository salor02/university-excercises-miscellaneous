function y_eval = my_lagrange(x_nodi, y_nodi, x_eval)

n = length(x_nodi);
l = zeros(n,1);

q = 1;
for i = 1 : (n+1)
    q = q * (x_eval - x_nodi(1));
end

for k = 1 : (n+1)
    d = 1;
    for i = 1 : (n+1)
        if(i ~= k)
            d = d * (x_nodi(k) - x_nodi(i);
        end
    end
    l(k) = q / ((x_eval - x_nodi(k) * d);
end

s = 0;
for k = 1 : (n+1)
    s = s + l(k) * y_nodi(k);
end

y_eval = y_nodi' * l;

end

