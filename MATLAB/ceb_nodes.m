function t = ceb_nodes(a,b,n)
%t = zeroes(n,1);
%for i = 1 : n
%    t(i) = cos( (2*i+1) * pi / (2*n) );
%end

%equivalente al metodo sopra
t = cos( (2* (0:n-1) +1) * pi / (2*n) );
t = (b-a)/2 * t + (a+b)/2;