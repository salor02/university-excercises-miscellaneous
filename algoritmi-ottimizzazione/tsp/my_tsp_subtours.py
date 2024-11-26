# Adapted by Mauro Dell'Amico from 
# Copyright 2020, Gurobi Optimization, LLC
#
# Solve a symmetric traveling salesman problem on a randomly generated set of
# points using Subtour elimination constraints.   The base MIP model only includes
# 'degree-2' constraints, requiring each node to have exactly
# two incident edges.  Solutions to this model may contain subtours -
# tours that don't visit every city.  
# The subtour elimination constraint adds new constraints to cut them off.

import sys
import random
import time
from itertools import combinations
import gurobipy as gp
from gurobipy import GRB

import tsp_utils as tspu

def main():
    RANDOM = False # if True generates a random instance
    TSPLIB = True # if true reads a TSPLIB instance (name in argv[1])
        
    if TSPLIB:
        if len(sys.argv) < 2:
            print('Usage: tsp.py nameOfTsplibInstanceFile')
            sys.exit(1)
        #n, dist = tspu.readTSPLIB_atsp("istanze/br17")
        dist=tspu.randomDiGraph(10, 0.5, 100)
        n, points, dist, optTour, optCost = tspu.readTSPLIB(sys.argv[1])
    else:
        #random instance 
        if len(sys.argv) < 2:
            print('Usage: tsp.py npoints')
            sys.exit(1)  
            
        n = int(sys.argv[1])        
        # Create n random points 
        if RANDOM :
            random.seed(92489)
            points, dist  = tspu.randomEuclGraph(n,100)
        
    #defines model
    m = gp.Model()

    # Create variables
    edges = [(i,j) for (i,j) in dist.keys()]
    visit_order = [0 for i in range(n)]
    vars_edges = m.addVars(edges, obj=dist, vtype=GRB.BINARY, name='e')
    vars_visit_order = m.addVars(range(1,len(visit_order)), lb=0, vtype=GRB.INTEGER, name='visit order')

    # Add degree-2 constraint
    
    m.addConstrs(vars_edges.sum(i, '*') == 1 for i in range(n))
    m.addConstrs(vars_edges.sum('*', j) == 1 for j in range(n))

    for i in vars_visit_order:
        for j in vars_visit_order:
            if j != 1:
                m.addConstr(vars_visit_order[j] - vars_visit_order[i] >= 1 - sys.maxsize * (1 - vars_edges[i,j]))
    
    # Optimize model
    
    tourLength = 0 #length of the shortest tour in the current solution    
    cutCount = 0
    start = time.process_time()
    
    m.optimize()
    m.write("my_tsp.lp")
    vals = m.getAttr('x', vars_edges)
    selected_edges = gp.tuplelist((i, j) for i, j in vals.keys() if vals[i, j] == 1)
    tourLength = len(selected_edges)
    m.write("myatsp.lp")

    end = time.process_time() 
    

    
    print('Added cuts : %d ' % cutCount)
    print("Points: ",points)
    print('Optimal tour: %s' % str(selected_edges))
    print('Optimal cost: %g Time = %g' % (m.objVal, end-start))
    print('')
    tspu.plot_selectedEdges2D(points, edges, selected_edges, save_fig='Try.png')

if __name__ == '__main__':
	main()