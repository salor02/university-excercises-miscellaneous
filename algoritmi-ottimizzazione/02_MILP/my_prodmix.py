import gurobipy as gp
from gurobipy import GRB
from os import path

Products, revenue, maxProd = gp.multidict ({
        'Product A': [ 30, 300],
        'Product B': [ 28, 180],
        'Product C': [ 24, 500]
        })
Components, cost, maxSupply = gp.multidict ({
        'Comp 1': [2,  800],
        'Comp 2': [4,  900],
        'Comp 3': [3, 1300]        
        })

BOM = {
    ('Product A', 'Comp 1'): 2,
    ('Product A', 'Comp 2'): 5,
    ('Product A', 'Comp 3'): 0,
    ('Product B', 'Comp 1'): 4,
    ('Product B', 'Comp 2'): 0,
    ('Product B', 'Comp 3'): 6,
    ('Product C', 'Comp 1'): 0,
    ('Product C', 'Comp 2'): 2,
    ('Product C', 'Comp 3'): 1
}

def printSolution(m):
    print(f'Maximized profit: {m.objVal}')
    for var in m.getVars():
        print(f'[{var.VarName}] {var.x}')

def main():
    #model
    m = gp.Model("my_prodmix")
    m.reset()

    #definizione delle variabili e vincoli sul loro valore
    xP = m.addVars(Products, vtype=GRB.INTEGER, lb=0, ub=maxProd, name=Products)
    xC = m.addVars(Components, vtype=GRB.INTEGER, lb=0, ub=maxSupply, name=Components)
    
    #definizione della funzione obiettivo
    m.setObjective(xP.prod(revenue) - xC.prod(cost), GRB.MAXIMIZE)

    #definizione vincoli
    m.addConstrs( gp.quicksum(BOM[p,c] * xP[p] for p in Products) == xC[c] for c in Components )

    m.write(path.join('Output','my_prodmix.lp'))
    m.optimize()
    printSolution(m)
    

if __name__ == "__main__":
    main()  
