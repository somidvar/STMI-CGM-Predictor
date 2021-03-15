from pandas import ExcelFile
import numpy as np
import scipy.stats
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, LogisticRegression, BayesianRidge, SGDRegressor, LassoLars, ARDRegression, PassiveAggressiveRegressor, TheilSenRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold
import sys
from sklearn.metrics import mean_squared_error


#half peak span.
def half_peak_span(x):
	
	max_value = max(x)
	
	above_half_peak_idx = []
	
	for index, value in enumerate(x):
		
		if value >= max_value/2:
			
			above_half_peak_idx.append(index)
			
	return (above_half_peak_idx[-1] - above_half_peak_idx[0]) * 15

	
#peak index
def peak_index(x):

	max_value = max(x)
	
	for index, value in enumerate(x):
	
		if value == max_value:
			
			return index
			
			break

			
Participant_ID = '38B'

xls = ExcelFile('%s_nine_standard_meals.xlsx' %(Participant_ID))

df_original = xls.parse(xls.sheet_names[0])


#feature space
feature_matrix = []


#label space
label_half_peak_span = []
label_3hrs_later = []
label_peak = []
label_AUC = []
label_peak_index = []
label_mean = []
label_std = []
label_skewness = []



for idx, row in df_original.iterrows():


	#feature
	one_feature = [row['Protein (g)'], row['CHO (g)'], row['Fat (ml)'], row['Total E (kcal)']]
	
	feature_matrix.append(one_feature)

	line = df_original.iloc[idx].dropna().tolist()
	
	print(line)


	#half_peak_span
	label_half_peak_span.append(half_peak_span(line[7:]))
	#3hrs later BG
	label_3hrs_later.append(line[18])
	#peak
	label_peak.append(max(line[7:]))
	#AUC
	label_AUC.append(np.trapz(line[7:]))
	#peak index
	label_peak_index.append(peak_index(line[7:]))
	#mean
	label_mean.append(np.mean(line[7:]))
	#std
	label_std.append(np.std(line[7:]))
	#skewness
	label_skewness.append(scipy.stats.skew(line[7:]))
	

#print(label_half_peak_span, label_3hrs_later, label_peak, label_AUC, label_peak_index, label_mean, label_std, label_skewness)

label_matrix = []


label_matrix.append(np.array(label_half_peak_span))
label_matrix.append(np.array(label_3hrs_later))
label_matrix.append(np.array(label_peak))
label_matrix.append(np.array(label_AUC))
label_matrix.append(np.array(label_peak_index))
label_matrix.append(np.array(label_mean))
label_matrix.append(np.array(label_std))
label_matrix.append(np.array(label_skewness))

	
feature_matrix = np.array(feature_matrix)
label_matrix = np.array(label_matrix)
label_matrix = label_matrix.T




clf = sys.argv[1]
if clf == 'linr':
	lm = LinearRegression()
elif clf == 'svr':
	lm = SVR()
elif clf == 'sgd':
	lm = SGDRegressor()
elif clf == 'bg':
	lm = BayesianRidge()
elif clf == 'll':
	lm = LassoLars()
elif clf == 'ard':
	lm = ARDRegression()
elif clf == 'par':
	lm = PassiveAggressiveRegressor()
elif clf == 'tsr':
	lm = TheilSenRegressor()
elif clf == 'knn':
	lm = KNeighborsRegressor()


regr = MultiOutputRegressor(lm)


pred_matrix = []
f0=open('results_pred.csv', 'w')
f1=open('results_true.csv', 'w')

#leave one meal out
loo = KFold(n_splits=9)
for train_index, test_index in loo.split(feature_matrix):

	regr.fit(feature_matrix[train_index], label_matrix[train_index])
	
	pred_vector = regr.predict(feature_matrix[test_index])
	

	#print ('pred', pred_vector)
	#print ('true', label_matrix[test_index][0], '\n')
	
	#print ('meal ', test_index, mean_squared_error(pred_vector, label_matrix[test_index]))
	
	
	for item in pred_vector[0]:
		f0.write(str(item))
		f0.write(',')
	f0.write('\n')
		
		
	for item in label_matrix[test_index][0]:
		f1.write(str(item))
		f1.write(',')
	f1.write('\n')
	
	pred_matrix.append(pred_vector)
	
	
	
	
pred_matrix = np.array(pred_matrix)


#pred_matrix_normed = pred_matrix / pred_matrix.max(axis = 0)
#label_matrix_normed = label_matrix / label_matrix.max(axis = 0)



		

	
	
	
	
	
	
	
	
	
	
	
	
	