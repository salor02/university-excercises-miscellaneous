clear variables
close all

%estremi intervallo
a = 0;
b = 10;

%generare i dati senza rumore
m = 20;
x = rand(m,1) * (b-a) + a;
y = my_function(x);

%sporcare i dati
volume = 1;
y = y + volume * randn(m,1);

%regressione
grad_mod = 1;
alpha = my_minq_poli(x, y, grad_mod+1, true);

%stampa
figure
hold on
plot(x,y,'ro');

n_plot = 1000;
x_plot = linspace(a,b,n_plot);
y_plot = polyval(alpha(length(alpha):-1:1), x_plot);
plot(x_plot, y_plot, 'b');

disp(alpha);
