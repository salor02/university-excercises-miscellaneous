import gurobipy as gp
from gurobipy import GRB
from os import path
import csv

def read_diet_csv(file_path, sep = ','):
    """
    Reads a csv with data for a diet problem
    Args:
        :param file_path: (str) path to the input file.
    :return:
    """
    nutrients, foods = [], []

    # Open and uses the csv file
    with open(file_path, 'r') as file:
        csv_r = csv.reader(file, delimiter=sep)
        next(csv_r)
        foods = next(csv_r)
        #selects the food labels
        foods = foods[1:-3]
        #read the nutrients labels
        line = next(csv_r)
        while line[0].lower() != 'cost':
            nutrients.append(line[0])
            line = next(csv_r)
        # defines the matrix and costs
        #initialize to 0
        A = {(n, f): 0 for n in nutrients for f in foods}  
        costs = {f : 0 for f in foods}
        min_req = {n: 0 for n in nutrients}
        max_req = {n: 0 for n in nutrients}
        #
        file.seek(0)
        next(csv_r) #header
        next(csv_r) #foods' labels
        for row,line in enumerate(csv_r):
            if line[0].lower() != 'cost':
                #nutrient in each food
                for col,f in enumerate(foods):
                    A[nutrients[row],f] = float(line[col+1])
                #min and max request for current nutrient
                min_req[nutrients[row]] = float(line[len(foods)+1]) 
                max_req[nutrients[row]] = float(line[len(foods)+2])
            else:
                # the cost row
                for col,f in enumerate(foods):
                    costs[f] = float(line[col+1])
                    
    return nutrients, foods, costs, A, min_req, max_req

def printSolution(m):
    print(f'Minimized cost for food purchasing: {m.objVal}')
    for var in m.getVars():
        print(f'[{var.VarName}] {var.x}')

def main():
    
    N, F, c, A, minR, maxR = read_diet_csv(path.join('Input', 'Diet.csv'),';')
    
    #model
    m = gp.Model("my_diet")
    m.reset()

    #definizione delle variabili e vincoli sul loro valore
    x = m.addVars(F, vtype=GRB.CONTINUOUS, lb=0, name=F)
    
    #definizione della funzione obiettivo
    m.setObjective(x.prod(c), GRB.MINIMIZE)

    #definizione vincoli
    m.addConstrs( gp.quicksum(A[n,f] * x[f] for f in F) >= minR[n] for n in N )
    m.addConstrs( gp.quicksum(A[n,f] * x[f] for f in F) <= maxR[n] for n in N )

    m.write(path.join('Output','my_diet.lp'))
    m.optimize()
    printSolution(m)

if __name__ == "__main__":
    main()    