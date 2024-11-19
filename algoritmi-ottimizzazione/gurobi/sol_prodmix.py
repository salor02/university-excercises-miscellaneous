# -*- coding: utf-8 -*-
"""
Created on 2021/03/17 - updated 2022/03/01

@author: Mauro Dell'Amico'
"""
import gurobipy as gp
from gurobipy import GRB
## production mix
#

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
    if m.status == GRB.Status.OPTIMAL:
        print('\nProfit : %g\n' % m.objVal)
        for x in m.getVars():
            #if x[i].x > 0.0001:
                print('%10s = %g' % (x.VarName,  x.X))
    else:
        print('No solution')

def printSlack(m):
    if m.status == GRB.Status.OPTIMAL:
        for cnst in m.getConstrs():
            print('Constr. %s = %g'%(cnst.constrName,cnst.slack) )
    else:
        print('No solution')
        
def main():
    #model
    m = gp.Model("Mix")
     
    # Create decision variables 
    xP = m.addVars(Products,vtype=GRB.INTEGER, ub=maxProd, name=Products)
    xC = m.addVars(Components,vtype=GRB.INTEGER, name=Components,ub=maxSupply)
    
    # The objective 
    m.setObjective(xP.prod(revenue) - xC.prod(cost), GRB.MAXIMIZE)
    
    # components availability
    for c in Components:
        m.addConstr(gp.quicksum(BOM[p, c] * xP[p] for p in Products) == xC[c],name=c)
        
    m.write("example02.lp")
    
    # Solve
    m.optimize()
    
    printSolution(m)
    #printSlack(m)

if __name__ == "__main__":
    main()    