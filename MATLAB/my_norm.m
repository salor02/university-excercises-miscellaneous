function res = my_norm(v)
    %res = sum(v .^ 2) ^ (1/2);
    res = (v' * v) ^ (1/2);
end