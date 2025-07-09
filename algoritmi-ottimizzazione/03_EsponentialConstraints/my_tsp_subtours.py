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

QUIET = 0

def subtour(edges,n):
    visited = [False for i in range(n)]
    shortestTour = range(n+1)
    for (i,j) in edges :
        if not visited[i]:
            isave = i
            narcs = 1
            visited[i] = True
            tour = [i]

            while j != isave:
                neighbor = [next_node for (current_node, next_node) in edges.select(j,'*') if next_node != i ]
                neighbor.extend([next_node for (next_node, current_node) in edges.select('*',j) if next_node != i])
                k = neighbor[0]
                visited[j] = True
                tour.append(j)
                i = j
                j = k
                narcs = narcs + 1
            if narcs < len(shortestTour):
                shortestTour = tour
    return shortestTour


def main():
    QUIET = False
   
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
    
    # distance is a dictionary where the keys are the edges and the values the corresponding distances
    if QUIET:
        gp.setParam('OutputFlag', 0)
    else:
        gp.setParam('OutputFlag', 1)
        
    #defines model
    m = gp.Model()

    # Create variables
    edges = [(i,j) for (i,j) in dist.keys() if i < j ] #li prende in quest'ordine per convenzione
    vars = m.addVars(edges, obj=dist, vtype=GRB.BINARY, name='e')
    
    # You could use Python looping constructs and m.addVar() to create
    # these decision variables instead.  The following would be equivalent
    # to the preceding m.addVars() call...
    #
    # vars = tupledict()
    # for i,j in dist.keys():
    #   vars[i,j] = m.addVar(obj=dist[i,j], vtype=GRB.BINARY,
    #                        name='e[%d,%d]'%(i,j))

    # Add degree-2 constraint
    
    # serve a specificare che ogni vertice ha un arco che esce e uno che entra, la somma è necessaria perchè per come
    # si è costruita la variabile edges senza il secondo termine escluderemmo tutti gli archi i cui vertici hanno 
    # un indice inferiore ad i.
    m.addConstrs(vars.sum(i, '*') + vars.sum('*',i) == 2 for i in range(n))
    
    # Using Python looping constructs, the preceding would be...
    #
    # for i in range(n):
    #   m.addConstr(sum(vars[i,j] for j in range(n)) == 2)
    
    
    # Optimize model
    
    tourLength = 0 #length of the shortest tour in the current solution    
    cutCount = 0
    start = time.process_time()
    
    while (tourLength < n):
        #solve the current problem
        m.optimize()
        m.write("tsp.lp")
        #find the smallest tour
        vals = m.getAttr('x', vars)
        selected_edges = gp.tuplelist((i, j) for i, j in vals.keys() if vals[i, j] > 0.5)
        tour = subtour(selected_edges,n)
        tourLength = len(tour)
        #
        if tourLength < n:
            # add subtour elimination constr. for every pair of cities in tour
            if not QUIET:
                print('\n>>> Subtour eliminated  %s\n' % str(tour))
                print(tour.index)
                tspu.plot_selectedEdges2D(points, edges, selected_edges, title="Subtours", figsize=(12, 12), save_fig=None)

#            m.addConstr(gp.quicksum(vars[i, j] for i, j in combinations(tour, 2))
#                           <= len(tour)-1)   
            m.addConstr(gp.quicksum(vars[i, j] for i in tour for j in tour if j > i)
                           <= len(tour)-1)               
            cutCount = cutCount + 1
        m.write("stsp.lp")

    end = time.process_time() 
    

    tour = subtour(selected_edges,n)
    #checks that the current solution is an Hamiltonian circuit      
    assert len(tour) == n
    
    print('Added cuts : %d ' % cutCount)
    print("Points: ",points)
    print('Optimal tour: %s' % str(tour))
    print('Optimal cost: %g Time = %g' % (m.objVal, end-start))
    print('')
    tspu.plot_selectedEdges2D(points, edges, selected_edges, save_fig='Try.png')

if __name__ == '__main__':
	main()