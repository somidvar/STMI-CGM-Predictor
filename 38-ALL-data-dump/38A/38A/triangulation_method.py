from sympy import *
import numpy as np
from numpy import genfromtxt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error




f0 = open('38A_results_true.csv', 'r')

f1 = open('38C_results_true.csv', 'r')

f2 = open('38C_pred_crosspatient.csv', 'w')

a = genfromtxt(f0, delimiter=',')
c = genfromtxt(f1, delimiter=',')

A38A = []
C38C = []

for row in a:
	one_row = []
	for item in row[:-1]:
		one_row.append(item)
	A38A.append(one_row)
	

for row in c:
	one_row = []
	for item in row[:-1]:
		one_row.append(item)
	C38C.append(one_row)

A38A = np.array(A38A)
C38C = np.array(C38C)



leave_one_meal_out = KFold(n_splits=9)


#each meal has a weight vector
for train_index, test_index in leave_one_meal_out.split(A38A):
	
	x1, x2, x3, x4, x5, x6, x7, x8 = symbols('x1 x2 x3 x4 x5 x6 x7 x8')
	
	x_vector = [x1, x2, x3, x4, x5, x6, x7, x8]
	
	equations = []
	
	#solve 8 equations (8 features), to get 8 weights
	for i in range(8):
		
		one_equation = 0
		
		for item in np.multiply(x_vector, A38A[train_index, i]):
			
			one_equation += item
		
		one_equation -= A38A[test_index, i]
		
		equations.append(one_equation)
		
			
		
	a = linsolve(equations, [x1, x2, x3, x4, x5, x6, x7, x8])
	for item in a:
		weights_38A = list(tuple(item))
		
		
	
	#test on other patient's meal
	summation = np.array([0, 0, 0, 0, 0, 0, 0, 0])
	
	#index for weights (train_index can't be used the index for weights)
	i = 0
	
	for one_train_index in train_index:
		
		summation = np.add(summation, np.dot(C38C[one_train_index, :], weights_38A[i]))	
		i += 1
		
	#print(C38C[test_index, :])	
	#print(summation)	
		
	for item in summation:
		
		f2.write(str(item))
		f2.write(',')
	
	f2.write('\n')
	

	
	
	
	
























