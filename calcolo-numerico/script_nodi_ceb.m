clear variables;
close all;

%inizializzazione figura
figure;
hold on;
tiledlayout(2,2); 
lagr = nexttile; hold on;
ceb = nexttile; hold on;
err_lagr = nexttile; hold on;
err_ceb = nexttile; hold on;

%definizione intervallo
a = -2;
b = 2;

%stampa grafico funzione nei 2 riquadri superiori
n_plot = 10000;
x_plot = linspace(a, b, n_plot);
y_plot = my_function(x_plot);
plot(lagr, x_plot, y_plot);
plot(ceb, x_plot, y_plot);

%campionamento di f
%n_pti_camp = 20;
for n_pti_camp = 2 : 100
    %generazione punti campionamento equispaziati
    x_camp = linspace(a, b, n_pti_camp);
    y_camp = my_function(x_camp);
    plot_camp_lagr = plot(lagr, x_camp, y_camp, 'ro');

    %generazione punti campionamento con Chebychev
    x_ceb = my_ceb_nodes(a, b, n_pti_camp)';
    y_ceb = my_function(x_ceb);
    plot_camp_ceb = plot(ceb, x_ceb, y_ceb, 'ro');
    
    %calcolo interpolazione con Lagrange (nodi equispaziati)
    y_lagr = zeros(length(x_plot),1);
    for k = 1 : length(x_plot)
        y_lagr(k) = my_lagrange_interpol(x_camp', y_camp', x_plot(k));
    end

    %calcolo interpolazione con Lagrange (nodi Chebycev)
    y_lagr_ceb = zeros(length(x_plot),1);
    for k = 1 : length(x_plot)
        y_lagr_ceb(k) = my_lagrange_interpol(x_ceb', y_ceb', x_plot(k));
    end
    
    %stampa grafico lagrange
    plot_lagr = plot(lagr, x_plot, y_lagr, 'g');
    plot_lagr_ceb = plot(ceb, x_plot, y_lagr_ceb, 'g');

    %calcolo errore
    y_err = y_lagr' - y_plot;
    plot_err_lagr = plot(err_lagr, x_plot, y_err, 'r');

    y_err_ceb = y_lagr_ceb' - y_plot;
    plot_err_lagr_ceb = plot(err_ceb, x_plot, y_err_ceb, 'r');

    %serve ad andare avanti premendo un tasto
    try
        w = waitforbuttonpress;
        delete(plot_camp_lagr);
        delete(plot_camp_ceb);
        delete(plot_lagr);
        delete(plot_lagr_ceb);
        delete(plot_err_lagr);
        delete(plot_err_lagr_ceb);
    catch ME
        break;
    end
end

