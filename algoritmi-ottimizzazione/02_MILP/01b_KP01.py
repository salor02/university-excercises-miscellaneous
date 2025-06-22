# -*- coding: utf-8 -*-
"""
Created on Sat Sep  17 12:18:57 2021

@author: Mauro
"""

import sys
from csv import reader
import gurobipy as gp
from gurobipy import GRB




def read_csv_points(file_path, sep=',', has_headers=True, dtype=float):
    """
    Read a csv file containing 2D points.

    :param file_path: path to the csv file containing the points
    :param sep: csv separator (default ',')
    :param has_headers: whether the file has headers (default True)
    :return: list of points
    """
    with open(file_path, 'r') as f:
        csv_r = reader(f, delimiter = sep)
        line = next(csv_r)
        C = int(line[0])
        if has_headers:
            headers = next(csv_r)
            print('Headers:', headers)
        names = []
        profits = {}
        weights =  {}
        for line in csv_r:
            names.append(line[0])
            profits[line[0]] = int(line[1])
            weights[line[0]] = int(line[2])
            
    return names, profits, weights,C

        
def printKP(indices, profits, weights):
    print('===================')
    for i in indices:
        print('%2s %5d %5d'% (i, profits[i], weights[i]) )
              

def printSolution(m, x, names, weights):
    if m.status == GRB.Status.OPTIMAL:
        print('\nProfit: %g Empty space %d' % (m.objVal, m.slack[0]))
        print('\n%3s %3s %5s' % ('var','val','weight'))
        for i in names:
            if x[i].x > 0.5:
                print('%3s %3g %5d' % (i,  x[i].x, weights[i]))
    else:
        print('No solution')
     
def main():
    m = gp.Model("kp")
    
    names, profits, weights, C = read_csv_points("input/KP01_1.csv", ';', True, int)

    #sortKP(names, profits, weights)        
    printKP(names, profits, weights)        

    # Create decision variables 
    x = m.addVars(names,vtype=GRB.BINARY)
    
    # The objective 
    m.setObjective(x.prod(profits), GRB.MAXIMIZE)
    #m.setObjective(gp.quicksum(x[i]*profits[i] for i in names), GRB.MAXIMIZE)
    
    m.addConstr(x.prod(weights) <= C)
    #m.addConstr(gp.quicksum(x[i]*weights[i] for i in names) <= C)
    
    # Solve
    m.optimize()

    printSolution(m, x, names, weights)
    
if __name__ == "__main__":
    main() 

