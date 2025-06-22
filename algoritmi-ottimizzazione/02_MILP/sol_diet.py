# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 19:45:17 2020

@author: Mauro
"""
import gurobipy as gp
from gurobipy import GRB
import os
import xlrd
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

def read_xlsx(file_path, sheet_name='Table'):
    """

    Args:
        :param file_path: (str) path to the input file.
        :param sheet_name: (str) name of the excel sheet to read
    :return:
    """
    nutrients, foods = [], []

    # Open the .xlsx file
    book = xlrd.open_workbook(file_path)
    sh = book.sheet_by_name(sheet_name)

    # Reads the nutrients (first column: Calcium, Iron, ...)
    row = 2
    nutrient = str(sh.cell_value(row, 0))
    while nutrient.lower() != 'cost':
        try:
            nutrients.append(nutrient)
            row = row + 1
            nutrient = str(sh.cell_value(row, 0))
        except IndexError:
            raise RuntimeError("Cannot find a cost row!")
    print("Nutrients = ", nutrients)

    # Reads the foods (first row: Orange, Beans, ...)
    col = 1
    while True:
        try:
            foods.append(sh.cell_value(1, col))
            col = col + 1
        except IndexError:
            break
    foods = foods[:-3]  # removes the last three columns
    print("Foods = ", foods)

    # Reads the costs
    costs = {f: float(sh.cell_value(row, idx + 1))
             for idx, f in enumerate(foods)}
    print(costs)

    # Reads the matrix of nutrients for each food
    A = {(n, f): 0 for n in nutrients for f in foods}
    for row_idx, n in enumerate(nutrients):
        for col_idx, f in enumerate(foods):
            A[n, f] = float(sh.cell_value(row_idx + 2, col_idx + 1))

    # Reads the requirements
    min_req = {n: 0 for n in nutrients}
    max_req = {n: 0 for n in nutrients}
    for row_idx, n in enumerate(nutrients):
        min_req[n] = float(sh.cell_value(row_idx + 2, len(foods) + 1))
        max_req[n] = float(sh.cell_value(row_idx + 2, len(foods) + 2))

    return nutrients, foods, costs, A, min_req, max_req


def make_primal(nutrients, foods, costs, A, min_req, max_req,
                name='Diet_Primal'):
    """

    Args:
        :param nutrients:
        :param foods:
        :param costs:
        :param A:
        :param min_req:
        :param max_req:
        :param name:
    :return:
    """
    m = gp.Model(name)
    # Variables
    x = m.addVars(foods,vtype=GRB.CONTINUOUS, name=foods)
    # Constraints
    m.addConstrs((gp.quicksum(A[n,f] * x[f] for f in foods)  >= min_req[n])
                  for n in nutrients)
    m.addConstrs((gp.quicksum(A[n,f] * x[f] for f in foods)  <= max_req[n])
                  for n in nutrients)    
    # Objective
    m.setObjective(gp.quicksum(costs[f] * x[f] for f in foods) , GRB.MINIMIZE)
    return m

if __name__ == '__main__':
    # flags to select input mode and model
    READ_CSV  = True
    READ_XLSX = False    
    
    if READ_CSV:
        nutrients, foods, costs, A, min_req, max_req = read_diet_csv(os.path.join('Input', 'Diet.csv'),';')
    
    if READ_XLSX:
        nutrients, foods, costs, A, min_req, max_req = \
                read_xlsx(os.path.join('Input', 'Diet.xls'))

         
    print('\n>>>Solving primal model')
    mp = make_primal(nutrients, foods, costs, A, min_req, max_req)
    mp.reset()
    mp.setParam("LogToConsole", 0)
    # mp.setParam("Method", 1)
    mp.optimize()
    mp.write(os.path.join('Output', '03_DietPrimal.lp'))
    
    if mp.status != GRB.Status.OPTIMAL:
        print("\n NO FEASIBLE SOLUTION !")
    else:
        print('\nPrimal Cost: %g\n' % mp.objVal)
        #
        print('Variables...')
        x = mp.getVars()
        for f in foods:
            print('%17s = %10f,  red. cost = %9g   stato = %d' % (
            f, x[foods.index(f)].X, x[foods.index(f)].RC, x[foods.index(f)].vBasis))
         
        print('Constraints...')
        cnt = mp.getConstrs()
        for n in nutrients:
            print('%17s) slack = %10g  rhs = %8g  dual value = %g' % (
            n, cnt[nutrients.index(n)].Slack, cnt[nutrients.index(n)].RHS,
            cnt[nutrients.index(n)].Pi))
    
