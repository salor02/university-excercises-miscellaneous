
import math
import time
from csv import reader
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt

def ScutSeparation(edges, n, S, Smax):
 
    visited = [False for i in range(n)]
    for (i,j) in edges :
        if not visited[i]:
            isave = i
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
    seq = [] 
    max_seq = []  

    for vertex in Tour:
        if vertex in S:
            seq.append(vertex)
        else:
            if len(seq) > len(max_seq):
                max_seq = seq
            seq = []  

    if len(seq) > len(max_seq):
        max_seq = seq

    if len(max_seq) > Smax:
        return max_seq
    else:
        return []  


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

def plot_selectedEdges2D(points, edges, selectededges=[], title="", figsize=(12, 12), save_fig=None):
    """
    Plot a graph in 2D emphasizing a set of selected edges .

    :param points: list of points.
    : param edges: all the edges of the graph
    :param selectededges: list of selected edges
    :param title: title of the figure.
    :param figsize: width and height of the figure
    :param save_fig: if provided, path to file in which the figure will be save.
    :return: None
    """

    plt.figure(figsize=figsize)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title, fontsize=15)
    plt.grid()
    x = [pnt[0] for pnt in points]
    y = [pnt[1] for pnt in points]
    plt.scatter(x, y, s=60)

    maxx = max(x)
    maxy=max(y)
    # Add label to points
    for i, label in enumerate(points):
        plt.annotate('{}'.format(i), (x[i]+0.001/maxx, y[i]+0.001/maxy), size=25)

   # Add the edges
    for (i,j) in edges:
        if i < j:
            plt.plot([points[i][0], points[j][0]], [points[i][1], points[j][1]], 'b', alpha=.5)
    # plots the selected edges (sub)tours
    for (i, j) in selectededges:
       plt.plot([points[i][0], points[j][0]], [points[i][1], points[j][1]], 'r', alpha=1.,linewidth=3)
       
    if save_fig:
        plt.savefig(save_fig)
    else:
        plt.show()

def read_csv_points(file_path, sep=',', has_headers=True, dtype=float):
    """
    Read a csv file containing 2D points.

    :param file_path: path to the csv file containing the points
    :param sep: csv separator (default ',')
    :param has_headers: whether the file has headers (default True)
    :param dtype: data type of values in the input file (default float)
    :return: list of points
    """
    with open(file_path, 'r') as f:
        csv_r = reader(f, delimiter=sep)

        if has_headers:
            headers = next(csv_r)
            print('Headers:', headers)

        points = [tuple(map(dtype, line)) for line in csv_r]
        print(points)

    return points

def EuclDist(points):
    """
    generates a dictionary of Euclidean distances between pairs of points    

    Parameters
    ----------
    points : list of pair of coordinates

    """
    # Dictionary of Euclidean distance between each pair of points
    dist = {(i, j):
            math.sqrt(sum((points[i][k]-points[j][k])**2 for k in range(2)))
            for i in range(len(points)) for j in range(len(points))}
            #for i in range(len(points)) for j in range(i)}
    return dist     

def main():
   
    points = read_csv_points('data.csv',';',True, dtype=int)
    n = len(points)
    S = []
    for i,p in enumerate(points) :
        if i % 2 == 0:
            S.append(i)
    Smax = 3
    dist = EuclDist(points)

    gp.setParam('OutputFlag', 0)
    #defines model
    m = gp.Model()

    # Create variables
    edges = [(i,j) for (i,j) in dist.keys() if i < j ]
    vars = m.addVars(edges, obj=dist, vtype=GRB.BINARY, name='e')

    # Add degree-2 constraint
    m.addConstrs(vars.sum(i, '*') + vars.sum('*',i) == 2 for i in range(n))
      
    # Optimize model
    subtourCuts = 0
    SCuts = 0
    start = time.process_time()
    added = True
    violated_sequences = []
    while (added):
        added = False
        #find an hamiltonian circuit
        tourLength = 0
        while (tourLength < n):
            #solve the current problem
            m.optimize()
            #find the smallest tour
            vals = m.getAttr('x', vars)
            selected_edges = gp.tuplelist((i, j) for i, j in vals.keys() if vals[i, j] > 0.5)
            tour = subtour(selected_edges,n)
            tourLength = len(tour)
            #
            if tourLength < n:
                # add subtour elimination constr. for every pair of cities in tour
                print('\n>>> Subtour eliminated  %s\n' % str(tour))
                m.addConstr(gp.quicksum(vars[i, j] for i in tour for j in tour if j > i)  <= len(tour) - 1)
                subtourCuts += 1
        ###################
        #check for the "sequence" constraint and add a constraint if necessary
        seqS = ScutSeparation(selected_edges,n, S, Smax)
        seqSlength = len(seqS)
        start_index = tour.index(seqS[0]) if seqS else -1  
        print(seqS,start_index)
        unique_seq = (tuple(seqS), start_index)

        if seqSlength > Smax and unique_seq not in violated_sequences:
            print('\n>>> Scut: Sequence found that violates Smax at position %d: %s\n' % (start_index, str(seqS)))
            m.addConstr(gp.quicksum(vars[i, j] for i in seqS for j in seqS if j > i) <= Smax)
            violated_sequences.append(unique_seq)  
            added = True
            SCuts += 1
            m.optimize()
        
    m.write("tsp++.lp")
    end = time.process_time() 
    
    tour = subtour(selected_edges,n)
    #checks that the current solution is an Hamiltonian circuit      
    assert len(tour) == n
    
    print('Added subtour elimination constraints : %d ' % subtourCuts)
    print('Added S cuts : %d ' % SCuts)
    print("Points: ",points)
    print('Optimal tour: %s' % str(tour))
    print('Optimal cost: %g Time = %g' % (m.objVal, end-start))
    print('')
    plot_selectedEdges2D(points, edges, selected_edges, save_fig='Try.png')

if __name__ == '__main__':
	main()