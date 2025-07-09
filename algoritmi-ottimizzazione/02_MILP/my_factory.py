# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 18:09:15 2021

@author: Mauro
"""
import gurobipy as gp
from gurobipy import GRB
import os
import csv 

def read_csv(file_path, sep = ','):
    """

    Args:
        :param file_path: (str) path to the input file.
        :param separator: (str) field separator
    :return:
    """
    products, machines, periods = [], [], []
    numMachines = []

    # Open the  file
    with open(file_path, 'r') as file:   
        # Reads the products (first row:  ...)
        csv_r = csv.reader(file, delimiter=sep)  
        line = next(csv_r)
        products = line[1:]  # removes the last col
        print("Products = ", products)
        line = next(csv_r)    
        profits = {j : int(line[1+index]) for index, j in enumerate(products)}
        print("Profits = ", profits)
        # Reads the machines 
        machines = []
        numMachines = []
        A = {(m, j): 0 for m in machines for j in products}
        line = next(csv_r)
        while len(line[0]) > 0:
            machines.append(line[0])
            for index, p in enumerate(products):
                A[line[0],p] = float(line[index+1])
            numMachines.append(int(line[-1]))
            line = next(csv_r)          
                   
        print("Machines = ", machines)
        print("NumMachines = ", numMachines)
        # max sales
        periods = []
        MAXS = {(t, j): 0 for t in periods for j in products}
        line = next(csv_r)
        line = next(csv_r)
        while len(line[0]) > 0:
            periods.append(line[0])
            for index, p in enumerate(products):
                MAXS[line[0],p] = float(line[index+1])
            line = next(csv_r)          
        # unavailable machines
        MC = {(m, t): 0 for m in machines for t in periods}   
        line = next(csv_r)
        line = next(csv_r)
        line = next(csv_r)
        while len(line[0]) > 0:
            try:
                 for index, m in enumerate(machines):
                     MC[m,line[0]] = numMachines[index] - int(line[index+1])
                 line = next(csv_r)     
            except StopIteration:
                 break
    
        print("Available machines",MC)

    return products, machines, periods, A, profits, MAXS, MC



def make_model(products, machines, periods, A, profits, MAXS, MC, model_name='ProductionMix'):
    m = gp.Model(model_name)
    m.setParam("MIPGap", 0.00001)

    x = m.addVars(periods, products, name="x",  vtype=GRB.INTEGER, lb=0 )
    s = m.addVars(periods, products, name="s",  vtype=GRB.INTEGER, lb=0 )
    i = m.addVars(periods, products, name="i",  vtype=GRB.INTEGER, lb=0 )

    m.addConstrs((i[periods[periods.index(t)-1],j] + x[t,j] - s[t,j] == i[t,j] for t in periods[1:] for j in products))
    m.addConstrs((i[t,j]  -x[t,j] + s[t,j] == 0 for t in periods[0:1] for j in products))

    m.addConstrs((i[t,j] <= 100 for t in periods for j in products))
    m.addConstrs((i[periods[len(periods)-1],j] == 50 for j in products))

    m.addConstrs((s[t,j] <= MAXS[t,j] for t in periods for j in products))

    m.addConstrs((gp.quicksum(A[i,j]*x[t,j] for j in products) <= 16*24*MC[i,t] for i in machines for t in periods))

    m.setObjective(gp.quicksum(profits[j]*s[t,j] - 0.5*i[t,j] for j in products for t in periods), GRB.MAXIMIZE)

    return m
###########################
# MAIN
###########################
if __name__ == '__main__':
    # Read the data from the input file
    products, machines, periods, A, profits, MAXS, MC = read_csv(os.path.join('Input', 'Factory.csv'),';')

    # Make the model and solve
    model = make_model(products, machines, periods, A, profits, MAXS, MC)
    
    model.optimize()
    
    model.write(os.path.join('Output', '04_Factory.lp'))

    if model.status == GRB.Status.OPTIMAL:      
        print(f'\nProfit: {model.objVal:.6f}\n')
        # Print single variables
        for i in model.getVars():
            if i.x > 0.0001:
                print(f'Var. {i.varName:22s} = {i.x:6.1f}')
                
        print()
        
        # Print some constraints related values
        for i in machines:
            for t in periods:
                 s = 'PT[' + t + ','+i+']'
                 cnt = model.getConstrByName(s)
                 print(f'{cnt.ConstrName:32s} slack = {cnt.slack:10.3f}  RHS = {cnt.RHS:6.0f}')  #DUAL = {cnt.Pi:7.3f}')
            print()
    else:
        print(">>> No feasible solution")
        
    