# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 15:00:58 2022

@author: Mauro
"""
import math
import random
from csv import reader
import matplotlib.pyplot as plt
import tsplib95 



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

def randomEuclGraph (n, maxcoord):
    """
    generates an instance of an Euclidean graph
    Parameters
    ----------
    n number of vertices
    maxcoord maximum value of a coordinate of a vertex
    """
    points = [(random.randint(0, maxcoord), random.randint(0, maxcoord)) for i in range(n)]
    dist = EuclDist(points)
    
    return points, dist

def randomDiGraph (n, p, maxcost):
    """
    generates an instance of a random digraph with probability p
    Parameters
    ----------
    n number of vertices
    p = probability of an arc between two vertices
    maxcost maximum cost of an arc
    """
    dist = {}
    for i in range(n):
        for j in range(n):
            r = random.random()
            if r <= p:
                dist[(i,j)] = int(r * maxcost)
    return dist        

    
def randomGraphGeo (n, d):
    """
    generates an instance of a Geometric graph U_{n,d}, 
    generated drawing from an uniform distribution n points in a unit square, 
    associating a vertex with each point and adding edge [u, v] to the graph 
    iff the euclidean distance between u and v is less or equal to d.
    Parameters
    ----------
    n number of vertices
    d max distance between two connected vertices 
    """
    points = [(random.random(), random.random()) for i in range(n)]
    dist = {}
    for i in range(len(points)-1):
        for j in range(i+1,len(points)): 
            
            dij = math.sqrt(sum((points[i][k]-points[j][k])**2 for k in range(2)))
            if dij < d:
                dist.update({(i,j) : dij})
                dist.update({(j,i) : dij})
    
    return points, dist

def randomGraph2D (n, p):
    """
    generates an instance of a Random graph in 2D G_{n,p}, 
    generated drawing from an uniform distribution n points in a unit square, 
    associating a vertex with each point and adding edge [u, v] with probability p 
    Parameters
    ----------
    n number of vertices
    d max distance between two connected vertices 
    """
    points = [(random.random(), random.random()) for i in range(n)]
    dist = {}
    for i in range(len(points)-1):
        for j in range(i+1,len(points)): 
            prob = random.random()
            dij = math.sqrt(sum((points[i][k]-points[j][k])**2 for k in range(2)))
            if prob < p:
                dist.update({(i,j) : dij})
                dist.update({(j,i) : dij})
    return points, dist    

def computeCost(tour,dist):
    """
    computes the cost of a given tour (list of nodes) using the distances in the distances
    ----------
    tour  list of nodes in the tour
    dist  dictionary with keys = arcs and values = cost
    """
    start = i = tour[0]
    mycost = 0
    for j in tour[1:]:        
        mycost += dist[i,j]
        #print(i,j,dist[i,j],mycost)
        i = j
    mycost += dist[(j,start)]
    return mycost

def readTSPLIB(file_path):
    problem = tsplib95.load(file_path+'.tsp')
    n = problem.dimension
    nodes = list(problem.get_nodes())
    points = tuple(problem.node_coords.values())
    if len(points) == 0:
        points = tuple(problem.display_data.values())
    # shift nodes to start from 0
    nodes = [x-1 for x in nodes]
    #shift nodes to start from 0
    edges = list(problem.get_edges())
    edges = [(i-1,j-1) for (i,j) in edges]
    dist = {(i,j) : 0 for  (i,j) in edges}
    for (i,j) in edges:
        dist[i,j] = problem.get_weight(i+1, j+1)
    opt = tsplib95.load(file_path+'.opt.tour')
    optTour = opt.tours[0]
    optTour = [x-1 for x in optTour]
    optCost = computeCost(optTour, dist)
    
    return n, points, dist, optTour, optCost

def readTSPLIB_atsp(file_path):
    problem = tsplib95.load(file_path + '.atsp')
    n = problem.dimension
    edges = list(problem.get_edges())
    edges = [(i , j ) for (i, j) in edges]
    dist = {(i, j): 0 for (i, j) in edges}
    for (i, j) in edges:
        dist[i, j] = problem.get_weight(i , j )
    return n, dist

def pippo():
    a =[]
    return a