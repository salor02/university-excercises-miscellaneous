import gurobipy as gp
from gurobipy import GRB
import csv 
from os import path
import math

def read_csv(file_path, sep = ','):

    X, Y, P, W = [], [], [], []
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader) 
        for row in reader:
            X.append(float(row[0]))
            Y.append(float(row[1]))
            P.append(int(row[2]))
            W.append(int(row[3]))
            #legge fino al punto 300
            if reader.line_num == 300:
                break
    return X, Y, P, W

# callback: si verifica che per ogni item selezionato la distanza massima da qualsiasi altro item selezionato sia 0.7,
# se viene trovata una distanza maggiore di 0.7 allora viene aggiunto il vincolo che impedisce ad entrambi i punti della coppia 
# di essere selezionati
def distanza_lazy(model, where):
    if where == GRB.Callback.MIPSOL:
        x = model.cbGetSolution(model._x)
        for i in range(model._num_items):
            if x[i] > 0.5: #se l'item è stato selezionato
                for j in range(model._num_items):
                    if x[j] > 0.5: #se l'altro item della coppia è stato selezionato avvia la verifica della distanza
                        dist = math.sqrt((model._X[i] - model._X[j])**2 + (model._Y[i] - model._Y[j])**2) #distanza euclidea
                        if dist > 0.7:
                            #viene aggiunto il vincolo lazy: non possono essere selezionati entrambi i punti contemporaneamente
                            model.cbLazy(model._x[i] + model._x[j] <= 1)

def main():
    X, Y, P, W = read_csv('data.csv')
    num_items = len(X)

    m = gp.Model("knapsack")

    #abilita lazy constraints
    m.Params.LazyConstraints = 1

    #memorizzazione variabili per poterle accedere nella callback
    m._num_items = num_items
    m._X = X
    m._Y = Y

    #una variabile per item, se item selezionato allora = 1, altrimenti = 0
    x = m.addVars(num_items, vtype=GRB.BINARY, name="x")
    m._x = x

    #vincolo capacità knapsack, c = 100, la sommatoria del prodotto dei punti selezionati per il loro peso deve essere <= 100
    m.addConstr(gp.quicksum(W[i]*x[i] for i in range(num_items)) <= 100)

    #funzione obiettivo, si massimizza la sommatoria del prodotto dei punti selezionati per il loro profitto
    m.setObjective(gp.quicksum(P[i]*x[i] for i in range(num_items)), GRB.MAXIMIZE)

    m.optimize(distanza_lazy)

    #selected items è la lista degli indici dei punti selezionati
    selected_items = [i for i in range(num_items) if x[i].x > 0.5]
    print("Item selezionati:")

    #vengono stampati sia l'indice del punto selezionato, sia le coordinate
    for item in selected_items:
        print(f'Item [{item}] ({X[item]},{(Y[item])})')
    print("Profitto totale:", sum(P[i] for i in selected_items))
    print("Peso totale:", sum(W[i] for i in selected_items))

if __name__ == "__main__":
    main()  
