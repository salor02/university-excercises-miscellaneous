# Adapted by Mauro Dell'Amico from 
# Copyright 2020, Gurobi Optimization, LLC

# Solve a symmetric traveling salesman problem on a randomly generated set of
# points using lazy constraints.   The base MIP model only includes
# 'degree-2' constraints, requiring each node to have exactly
# two incident edges.  Solutions to this model may contain subtours -
# tours that don't visit every city.  The lazy constraint callback
# adds new constraints to cut them off.

import sys
#import math
import random
import time
from itertools import combinations
#import matplotlib.pyplot as plt
#from csv import reader
import gurobipy as gp
from gurobipy import GRB

import tsp_utils as tspu

QUIET = 0
# Callback - use lazy constraints to eliminate sub-tours
def subtourelim(model, where):
    #we execute the function only when the MIP solver has found a new integer solution
    if where == GRB.Callback.MIPSOL:
        # make a list of edges selected in the solution
        #in a callBack we cannot access directly the value of the variables. We must use cbGetSolution()
        x = model.cbGetSolution(model._vars)
        selected_edges = gp.tuplelist((i, j) for i, j in model._vars.keys()
                                if x[i, j] > 0.5)
#
        # find the shortest cycle in the selected edge list
        tour = subtour(selected_edges, model._n)
        if len(tour) < model._n:
            # add subtour elimination constr. for every pair of cities in tour
            #if not QUIET:
                #print('\n>>> Subtour eliminated  %s\n' % str(tour))        
            model.cbLazy(gp.quicksum(model._vars[i, j]
                                     for i, j in combinations(tour, 2) if i < j) +
                         gp.quicksum(model._vars[j, i]
                                     for i, j in combinations(tour, 2) if i > j)
                         <= len(tour)-1)


def subtour(edges,n):
    # finds the shortest subtour in the list of edges 
    # N.B. the graph induced by the "edges" must contain only cycles
    visited = [False for i in range(n)]
    shortestTour = range(n+1)  # initial length has 1 more city
    for (i,j) in edges :
        if not visited[i]:
            isave = i
            narcs = 1
            visited[i]=True
            Tour = [i]
            while j != isave:
                neighbor = [kk for (jj,kk) in edges.select(j,'*') if kk != i ]
                neighbor.extend([kk for (kk, jj) in edges.select('*', j) if kk != i])
                k = neighbor[0]
                visited[j] = True
                Tour.append(j)
                i = j
                j = k
                narcs = narcs + 1
            if narcs < len(shortestTour):
                shortestTour = Tour
    return shortestTour


############################### MAIN ##################################################
def main():
    QUIET = False
    
    RANDOM = False # if True generates a random instance
    TSPLIB = True # if true reads a TSPLIB instance (name in argv[1])
        
    if TSPLIB:
        if len(sys.argv) < 2:
            print('Usage: tsp.py nameOfTsplibInstanceFile\n\n')
            sys.exit(1)  
        n, points, dist, optTour, optCost = tspu.readTSPLIB(sys.argv[1])
    else:
        #random instance 
        if len(sys.argv) < 2:
            print('Usage: tsp.py npoints\n\n')
            sys.exit(1)  
            
        n = int(sys.argv[1])        
        # Create n random points 
        if RANDOM :
            random.seed(989)
            points,dist  = tspu.randomEuclGraph(n,100)
    
    # distance is a dictionary where the keys are the edges and the values the corresponding distances
    
    if QUIET:
        gp.setParam('OutputFlag', 0)
    else:
        gp.setParam('OutputFlag', 1)
    
    m = gp.Model()
    
    #store the number of vertices n in model "m" to be used in the callback (otherwise it is not accessible)
    m._n = n
    
    # Create variables
    edges = [(i,j) for (i,j) in dist.keys() if i < j ]
    vars = m.addVars(edges, obj=dist, vtype=GRB.BINARY, name='e')
    #we assume that the graph is non-oriented 

    # You could use Python looping constructs and m.addVar() to create
    # these decision variables instead.  The following would be equivalent
    # to the preceding m.addVars() call...
    #
    # vars = tupledict()
    # for i,j in dist.keys():
    #   vars[i,j] = m.addVar(obj=dist[i,j], vtype=GRB.BINARY,
    #                        name='e[%d,%d]'%(i,j))
    
    
    # Add degree-2 constraint
    
    m.addConstrs(vars.sum(i, '*') + vars.sum('*',i) == 2 for i in range(n))
    m.write("tsp.lp")
    
    # Using Python looping constructs, the preceding would be...
    #
    # for i in range(n):
    #   m.addConstr(sum(vars[i,j] for j in range(n)) == 2)
    
    # Optimize model
    
    #store variables in model "m"
    m._vars = vars
    #instructs Gurobi tu use Lazy constraints
    m.Params.lazyConstraints = 1
    
    start = time.process_time()
    #subtourelim is the function implementing the lazy constraints
    m.optimize(subtourelim)
    end = time.process_time()
    
    vals = m.getAttr('x', vars)
    selected_edges = gp.tuplelist((i, j) for i, j in vals.keys() if vals[i, j] > 0.5)
    
    
    tour = subtour(selected_edges,n)
    #checks that the current solution is an Hamiltonian circuit              
    assert len(tour) == n
    
    print('')
    print("Points: ",points)
    print('Optimal tour: %s' % str(tour))
    print('Optimal cost: %g Time = %g' % (m.objVal, end-start))
    print('')
    
    tspu.plot_selectedEdges2D(points, edges, selected_edges, save_fig='Try.png')

if __name__ == '__main__':
	main()