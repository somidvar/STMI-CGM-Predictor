from pandas import ExcelFile
from sklearn.linear_model import LinearRegression, LogisticRegression, BayesianRidge, SGDRegressor, LassoLars, ARDRegression, PassiveAggressiveRegressor, TheilSenRegressor
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import numpy as np
import scipy
from sklearn.model_selection import KFold
import sys
from sklearn.metrics import mean_squared_error, accuracy_score
from math import sqrt
import matplotlib.pyplot as plt



Participant_ID = '38A'


xls = ExcelFile('%s_nine_standard_meals.xlsx' %(Participant_ID))


df_original = xls.parse(xls.sheet_names[0])


#half peak span.
def half_peak_span(x):
	
	max_value = max(x)
	
	above_half_peak_idx = []
	
	for index, value in enumerate(x):
		
		if value >= max_value/2:
			
			above_half_peak_idx.append(index)
			


	return (above_half_peak_idx[-1] - above_half_peak_idx[0]) * 15

	
def peak_index(x):

	max_value = max(x)
	
	for index, value in enumerate(x):
	
		if value == max_value:
			
			return index
			
			break
	
clf = sys.argv[1]
if clf == 'linr':
	lm_carb = LinearRegression()
	lm_protein = LinearRegression()
	lm_fat = LinearRegression()
elif clf == 'svr':
	lm_carb = SVR()
	lm_protein = SVR()
	lm_fat = SVR()
elif clf == 'sgd':
	lm_carb = SGDRegressor()
	lm_protein = SGDRegressor()
	lm_fat = SGDRegressor()
elif clf == 'bg':
	lm_carb = BayesianRidge()
	lm_protein = BayesianRidge()
	lm_fat = BayesianRidge()
elif clf == 'll':
	lm_carb = LassoLars()
	lm_protein = LassoLars()
	lm_fat = LassoLars()
elif clf == 'ard':
	lm_carb = ARDRegression()
	lm_protein = ARDRegression()
	lm_fat = ARDRegression()
elif clf == 'par':
	lm_carb = PassiveAggressiveRegressor()
	lm_protein = PassiveAggressiveRegressor()
	lm_fat = PassiveAggressiveRegressor()
elif clf == 'tsr':
	lm_carb = TheilSenRegressor()
	lm_protein = TheilSenRegressor()
	lm_fat = TheilSenRegressor()
##############################################################	
elif clf == 'svm':	
	lm_carb = SVC()
	lm_protein = SVC()
	lm_fat = SVC()
elif clf == 'mlp':	
	lm_carb = MLPClassifier()
	lm_protein = MLPClassifier()
	lm_fat = MLPClassifier()
elif clf == 'knn':	
	lm_carb = KNeighborsClassifier()
	lm_protein = KNeighborsClassifier()
	lm_fat = KNeighborsClassifier()
elif clf == 'gp':	
	lm_carb = GaussianProcessClassifier()
	lm_protein = GaussianProcessClassifier()
	lm_fat = GaussianProcessClassifier()
elif clf == 'dt':	
	lm_carb = DecisionTreeClassifier()
	lm_protein = DecisionTreeClassifier()
	lm_fat = DecisionTreeClassifier()
elif clf == 'rf':	
	lm_carb = RandomForestClassifier()
	lm_protein = RandomForestClassifier()
	lm_fat = RandomForestClassifier()
elif clf == 'ab':	
	lm_carb = AdaBoostClassifier()
	lm_protein = AdaBoostClassifier()
	lm_fat = AdaBoostClassifier()
elif clf == 'xg':	
	lm_carb = XGBClassifier()
	lm_protein = XGBClassifier()
	lm_fat = XGBClassifier()
elif clf == 'lr':
	lm_carb = LogisticRegression(multi_class='multinomial', solver='lbfgs')
	lm_protein = LogisticRegression(multi_class='multinomial', solver='lbfgs')
	lm_fat =LogisticRegression(multi_class='multinomial', solver='lbfgs')
	
	
	
	
	
	
feature_matrix = []

for idx, row in df_original.iterrows():
	line = df_original.iloc[idx].dropna().tolist()
	
	one_feature = []
	
	#half_peak_span
	one_feature.append(half_peak_span(line[7:]))
	#3hrs later BG
	one_feature.append(line[18])
	#peak
	one_feature.append(max(line[7:]))
	#AUC
	one_feature.append(np.trapz(line[7:]))
	#peak index
	one_feature.append(peak_index(line[7:]))
	#mean
	one_feature.append(np.mean(line[7:]))
	#std
	one_feature.append(np.std(line[7:]))
	#skewness
	one_feature.append(scipy.stats.skew(line[7:]))
	
	feature_matrix.append(one_feature)
	

label_protein = []

for item in df_original['Protein (g)'].tolist():
	
	label_protein.append(float(item))
	
	# item = str(item)
	
	# if item == '15':
		# label_protein.append('low')
	# if item == '30':
		# label_protein.append('Mid')
	# if item == '60':
		# label_protein.append('high')
	
	
label_carb = []

for item in df_original['CHO (g)'].tolist():

	label_carb.append(float(item))
	
	# item = str(item)
	
	# if item == '42.5':
		# label_carb.append('low')
	# if item == '85.0':
		# label_carb.append('Mid')
	# if item == '170.0':
		# label_carb.append('high')
	
	
	
label_fat = []

for item in df_original['Fat (ml)'].tolist():

	label_fat.append(float(item))
	
	# item = str(item)

	# if item == '13':
		# label_fat.append('low')
	# if item == '26':
		# label_fat.append('Mid')
	# if item == '52':
		# label_fat.append('high')
		
		

	
feature_matrix = np.array(feature_matrix)
label_protein = np.array(label_protein)
label_carb = np.array(label_carb)
label_fat = np.array(label_fat)
	
	
carb_pred = []
carb_true = []
protein_pred = []
protein_true = []
fat_pred = []
fat_true = []


	
#leave one meal out
loo = KFold(n_splits=9)
for train_index, test_index in loo.split(feature_matrix):
	
	lm_carb.fit(feature_matrix[train_index], label_carb[train_index])
	lm_protein.fit(feature_matrix[train_index], label_protein[train_index])
	lm_fat.fit(feature_matrix[train_index], label_fat[train_index])
	
	y_carb_pred = lm_carb.predict(feature_matrix[test_index])
	y_protein_pred = lm_protein.predict(feature_matrix[test_index])
	y_fat_pred = lm_fat.predict(feature_matrix[test_index])

	carb_pred.append((y_carb_pred[0]))
	carb_true.append(label_carb[test_index][0])
	protein_pred.append((y_protein_pred[0]))
	protein_true.append(label_protein[test_index][0])
	fat_pred.append((y_fat_pred[0]))
	fat_true.append(label_fat[test_index][0])
	
print ('----------------------------------------')
print ('carb_pred: ', carb_pred)
print ('carb_true: ', carb_true)	
print ('carb coeff: ', np.corrcoef(carb_pred, carb_true))
#print  (sqrt(mean_squared_error(carb_true, carb_pred)))
#print (accuracy_score(carb_true, carb_pred))
print ('protein_pred: ', protein_pred)
print ('protein_true: ', protein_true)
print ('protein coeff: ', np.corrcoef(protein_pred, protein_true))
#print  (sqrt(mean_squared_error(protein_true, protein_pred)))
#print  (accuracy_score(protein_true, protein_pred))
print ('fat_pred: ', fat_pred)
print ('fat_true: ', fat_true)
print ('fat coeff: ', np.corrcoef(fat_pred, fat_true))
#print  (sqrt(mean_squared_error(fat_true, fat_pred)))
#print  (accuracy_score(fat_true, fat_pred))


f, (ax1, ax2, ax3) = plt.subplots(1,3)
f.suptitle(clf)
ax1.scatter(carb_pred, carb_true)
ax2.scatter(protein_pred, protein_true)
ax3.scatter(fat_pred, fat_true)
#order = np.array(['low', 'Mid', 'high'])
#plt.xticks(range(len(order)), order)


#give coefficient
ax1.set_title('carb coeff  '+str(np.corrcoef(carb_pred, carb_true)[0][1]))
ax2.set_title('protein coeff  '+str(np.corrcoef(protein_pred, protein_true)[0][1]))
ax3.set_title('fat coeff  '+str(np.corrcoef(fat_pred, fat_true)[0][1]))


#give label each point
for i in range(9):
	ax1.annotate(i+1, (carb_pred[i], carb_true[i]))
	ax2.annotate(i+1, (protein_pred[i], protein_true[i]))
	ax3.annotate(i+1, (fat_pred[i], fat_true[i]))
	
plt.show()





