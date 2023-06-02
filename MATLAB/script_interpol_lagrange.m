clear variables;
close all;

%inizializzazione figura
figure;
hold on;

%definizione intervallo
a = -2;
b = 2;

%stampa grafico funzione
n_plot = 10000;
x_plot = linspace(a, b, n_plot);
y_plot = my_function(x_plot);
plot(x_plot, y_plot);

%campionamento di f
n_pti_camp = 20;
x_camp = linspace(a, b, n_pti_camp);
y_camp = my_function(x_camp);
plot(x_camp, y_camp, 'ro');

%calcolo interpolazione con Lagrange
y_lagr = zeros(length(x_plot),1);
for k = 1 : length(x_plot)
    y_lagr(k) = my_lagrange_interpol(x_camp, y_camp', x_plot(k));
end

%stampa grafico lagrange
plot(x_plot, y_lagr, 'g');

