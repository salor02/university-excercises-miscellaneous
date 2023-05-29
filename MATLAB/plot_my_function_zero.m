% Scrivere una funzione che prende in ingresso:
% - una stringa func_name contenete il nome del file in cui è descritta la funzione f(x),
% - il numero reale a che definisce l'estremo sinistro dell'intervallo di stampa,
% - il numero reale b che definisce l'estremo destro dell'intervallo di stampa,
% - il numero npt di punti di campionamento di [a,b],
% - il valore z dello zero che avete calcolato con il metodo di bisezione.
% 
% La funzione deve stampare 
% 1) il grafico di f(x) nell'intervallo [a,b],
% 2) l'asse x ristretto all'intervallo [a,b],
% 3) il punto (z,0) che rappresenta l'approssimazione dello zero di f che avete ottenuto con il metodo di bisezione.

function plot_my_function_zero(func_name, a, b, npt, z)

    %inizializzazione figura
    close all;
    figure;
    hold on;

    %disegno grafico funzione
    x_plot = linspace(a, b, npt);
    y_plot = feval(func_name, x_plot);
    plot(x_plot, y_plot);

    %disegno asse x
    plot(x_plot,zeros(npt,1),'r');

    %disegno punto calcolato da metodo bisezione
    plot(z,0,'go');

    axis equal;


end